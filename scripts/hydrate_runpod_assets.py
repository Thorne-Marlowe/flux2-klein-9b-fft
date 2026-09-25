#!/usr/bin/env python3
"""Hydrate pinned Base-9B model and Phase 4.1 dataset assets; never train."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bootstrap_runpod as bootstrap
from scripts import dataset_transport


FORMAT = "klein-runpod-asset-hydration"
SCHEMA_VERSION = 1
_RECORD_DIRECTORY = ".klein-hydration"
_SHA256 = re.compile(r"[0-9a-f]{40}")
_WORKSPACE_DIRECTORIES = ("models", "datasets", "runs", "evidence", "cache")


class HydrationError(RuntimeError):
    """A requested external asset cannot be safely hydrated or reused."""


def _canonical_bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _error(message: str, error: Exception | None = None) -> HydrationError:
    # Hub exceptions may contain URLs or service diagnostics.  Never relay them.
    return HydrationError(message)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HydrationError(f"Configuration field {name!r} must be a non-empty string.")
    return value.strip()


def _workspace_path(value: object, name: str, workspace: Path) -> Path:
    text = _require_string(value, name)
    path = Path(text)
    if not path.is_absolute():
        raise HydrationError(f"Configuration field {name!r} must be an absolute path under workspace_root.")
    path = path.resolve(strict=False)
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise HydrationError(f"Configuration field {name!r} must be under workspace_root.") from error
    return path


def load_config(path: Path) -> dict[str, Any]:
    """Load the small explicit asset configuration without accepting secrets."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HydrationError("Asset configuration must be valid UTF-8 JSON.") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "workspace_root", "model", "dataset"}:
        raise HydrationError("Asset configuration has unsupported or missing fields.")
    if document["schema_version"] != SCHEMA_VERSION:
        raise HydrationError("Unsupported asset configuration schema version.")
    workspace_value = _require_string(document["workspace_root"], "workspace_root")
    workspace = Path(workspace_value).absolute()
    if workspace.is_symlink() or not workspace.is_dir():
        raise HydrationError("workspace_root must be an existing non-linked directory.")
    workspace = workspace.resolve(strict=True)
    model = document["model"]
    dataset = document["dataset"]
    if not isinstance(model, dict) or set(model) != {"repository", "revision", "destination"}:
        raise HydrationError("model configuration must contain repository, revision, and destination.")
    if not isinstance(dataset, dict) or set(dataset) != {"repository", "revision", "package_destination", "destination"}:
        raise HydrationError("dataset configuration must contain repository, revision, package_destination, and destination.")
    return {
        "workspace_root": workspace,
        "model": {"repository": _require_string(model["repository"], "model.repository"),
                  "revision": _require_string(model["revision"], "model.revision"),
                  "destination": _workspace_path(model["destination"], "model.destination", workspace)},
        "dataset": {"repository": _require_string(dataset["repository"], "dataset.repository"),
                    "revision": _require_string(dataset["revision"], "dataset.revision"),
                    "package_destination": _workspace_path(dataset["package_destination"], "dataset.package_destination", workspace),
                    "destination": _workspace_path(dataset["destination"], "dataset.destination", workspace)},
    }


def _token(environment: dict[str, str] | None = None) -> str:
    token = (environment or os.environ).get("HF_TOKEN", "").strip()
    if not token:
        raise HydrationError("Hydration requires HF_TOKEN through the runtime environment; interactive Hugging Face login is not used.")
    return token


def _record_path(workspace: Path, asset: str) -> Path:
    return workspace / _RECORD_DIRECTORY / f"{asset}.json"


def _read_document(path: Path, description: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise HydrationError(f"Existing {description} is not a regular file; inspect it manually.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HydrationError(f"Existing {description} is malformed; inspect it manually.") from error
    if not isinstance(document, dict):
        raise HydrationError(f"Existing {description} is malformed; inspect it manually.")
    return document


def _write_document_last(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise HydrationError("Hydration record directory must not be a symbolic link.")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(document))
            handle.flush()
        os.replace(temporary, path)
    except OSError as error:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise HydrationError("Could not publish hydration completion record.") from error


def _request_matches(record: dict[str, Any], *, asset: str, repository: str,
                     requested_revision: str, resolved_revision: str, destinations: dict[str, Path]) -> bool:
    if record.get("format") != FORMAT or record.get("schema_version") != SCHEMA_VERSION:
        return False
    if record.get("asset") != asset or record.get("completion") != "complete":
        return False
    if record.get("repository") != repository or record.get("requested_revision") != requested_revision:
        return False
    if record.get("resolved_revision") != resolved_revision:
        return False
    return all(record.get(name) == str(path) for name, path in destinations.items())


def _intent_document(asset: str, repository: str, requested_revision: str, resolved_revision: str) -> dict[str, Any]:
    return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "asset": asset,
            "repository": repository, "requested_revision": requested_revision,
            "resolved_revision": resolved_revision, "completion": "partial"}


def _assert_intent(path: Path, expected: dict[str, Any]) -> None:
    actual = _read_document(path, "partial hydration intent")
    if actual is None:
        _write_document_last(path, expected)
    elif actual != expected:
        raise HydrationError("Existing partial hydration state belongs to a different requested asset; inspect it manually.")


def _assert_directory_parent(path: Path, workspace: Path) -> None:
    try:
        path.parent.relative_to(workspace)
    except ValueError as error:
        raise HydrationError("Asset destination must remain below workspace_root.") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise HydrationError("Asset destination parent must be a real directory.")


def ensure_workspace_layout(workspace: Path) -> None:
    """Create only the empty standard workspace parents needed by preflight."""
    for name in _WORKSPACE_DIRECTORIES:
        path = workspace / name
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise HydrationError(f"Workspace path {name!r} must be a real directory.")
        else:
            path.mkdir()


def _model_record(source: dict[str, Any], destination: Path, validation: dict[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "asset": "model", "completion": "complete",
            "repository": source["repository"], "requested_revision": source["requested_revision"],
            "resolved_revision": source["revision"], "destination": str(destination),
            "selected_files": validation["selected_files"], "selected_bytes": validation["selected_bytes"]}


def hydrate_model(config: dict[str, Any], token: str, *, hub: Any = None) -> dict[str, Any]:
    """Reuse generic bootstrap selection/download validation with a secret-only token."""
    source = bootstrap.resolve_model_source(config["repository"], config["revision"], hub=hub,
                                            token=token, allow_cached_token=False)
    destination: Path = config["destination"]
    workspace: Path = config["workspace_root"]
    record_path = _record_path(workspace, "model")
    record = _read_document(record_path, "model completion record")
    expected_destinations = {"destination": destination}
    if record is not None and not _request_matches(record, asset="model", repository=source["repository"],
                                                    requested_revision=source["requested_revision"],
                                                    resolved_revision=source["revision"], destinations=expected_destinations):
        raise HydrationError("Existing model completion record does not match the requested model; use a separate destination.")
    _assert_directory_parent(destination, workspace)
    intent = destination / ".klein-hydration-intent.json"
    if record is None:
        if destination.exists() and destination.is_symlink():
            raise HydrationError("Existing model destination is a symbolic link; inspect it manually.")
        if destination.exists() and any(destination.iterdir()):
            expected_intent = _intent_document("model", source["repository"], source["requested_revision"], source["revision"])
            if _read_document(intent, "model partial hydration intent") != expected_intent:
                raise HydrationError("Existing model destination cannot be proven to be matching partial hydration state; it will not be replaced.")
        elif not destination.exists():
            destination.mkdir()
            _assert_intent(intent, _intent_document("model", source["repository"], source["requested_revision"], source["revision"]))
        else:
            _assert_intent(intent, _intent_document("model", source["repository"], source["requested_revision"], source["revision"]))
    elif not destination.is_dir() or destination.is_symlink():
        raise HydrationError("Model completion record exists but its destination is unavailable; do not replace it automatically.")
    try:
        bootstrap.download_model_source(destination, source)
        validation = bootstrap.validate_model(destination)
    except Exception as error:
        raise HydrationError("Model hydration did not complete validation; partial files were retained only for the matching request.") from error
    completed = _model_record(source, destination, validation)
    if record is not None and record != completed:
        raise HydrationError("Existing model completion record disagrees with the validated local model; do not replace it automatically.")
    _write_document_last(record_path, completed)
    return completed


def _resolve_dataset_source(repository: str, revision: str, token: str, hub: Any = None) -> dict[str, Any]:
    if hub is None:
        import huggingface_hub as hub
    try:
        api = hub.HfApi(token=token)
        api.whoami()
        api.auth_check(repo_id=repository, repo_type="dataset")
        info = api.dataset_info(repository, revision=revision, files_metadata=True)
    except Exception as error:
        raise _error("Hugging Face authentication/dataset access failed. Verify HF_TOKEN read access, repository access, revision, and network connectivity.", error)
    if not _SHA256.fullmatch(getattr(info, "sha", "") or ""):
        raise HydrationError("Hub did not resolve an immutable dataset revision.")
    inventory: dict[str, int] = {}
    for entry in getattr(info, "siblings", ()):
        name = getattr(entry, "rfilename", None)
        size = getattr(entry, "size", None)
        path = PurePosixPath(name) if isinstance(name, str) else None
        if path is None or path.is_absolute() or len(path.parts) != 1 or path.name != name or "\\" in name:
            # Unrelated nested repository content is never downloaded.  The
            # manifest later requires every transport file to be a safe flat
            # name present in this inventory.
            continue
        if type(size) is not int or size <= 0 or name in inventory:
            raise HydrationError("Dataset source has a missing size or duplicate filename.")
        inventory[name] = size
    return {"repository": repository, "requested_revision": revision, "revision": info.sha,
            "inventory": inventory, "hub": hub, "token": token}


def _download_dataset_file(source: dict[str, Any], filename: str, directory: Path) -> None:
    try:
        source["hub"].hf_hub_download(repo_id=source["repository"], repo_type="dataset", filename=filename,
                                       revision=source["revision"], local_dir=str(directory), token=source["token"],
                                       force_download=False)
    except Exception as error:
        raise _error("Dataset download failed. Matching private staging files were retained for a safe retry.", error)


def _dataset_package_intent_path(package: Path) -> Path:
    return package.parent / f".{package.name}.hydration-intent.json"


def _publish_package(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise HydrationError("Dataset package destination already exists and will not be overwritten.")
    try:
        os.rename(staging, destination)
    except OSError as error:
        raise HydrationError("Could not publish validated dataset package staging.") from error


def _dataset_record(source: dict[str, Any], package: Path, destination: Path,
                    manifest: dict[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "asset": "dataset", "completion": "complete",
            "repository": source["repository"], "requested_revision": source["requested_revision"],
            "resolved_revision": source["revision"], "package_destination": str(package),
            "destination": str(destination), "dataset_identity": manifest["dataset_identity"],
            "pair_count": manifest["pair_count"], "shard_count": len(manifest["shards"])}


def _validate_complete_dataset(package: Path, destination: Path, record: dict[str, Any]) -> None:
    try:
        manifest = dataset_transport.validate_package(package)
        if manifest["dataset_identity"] != record.get("dataset_identity") or manifest["pair_count"] != record.get("pair_count"):
            raise HydrationError("Existing dataset package does not match its completion record; do not replace it automatically.")
        dataset_transport.validate_extracted_dataset(destination, manifest)
    except HydrationError:
        raise
    except Exception as error:
        raise HydrationError("Existing completed dataset cannot be validated; do not replace it automatically.") from error


def hydrate_dataset(config: dict[str, Any], token: str, *, hub: Any = None) -> dict[str, Any]:
    """Fetch exactly a manifest and its declared tar shards, then extract safely."""
    source = _resolve_dataset_source(config["repository"], config["revision"], token, hub)
    package: Path = config["package_destination"]
    destination: Path = config["destination"]
    workspace: Path = config["workspace_root"]
    _assert_directory_parent(package, workspace)
    _assert_directory_parent(destination, workspace)
    record_path = _record_path(workspace, "dataset")
    record = _read_document(record_path, "dataset completion record")
    expected_destinations = {"package_destination": package, "destination": destination}
    if record is not None:
        if not _request_matches(record, asset="dataset", repository=source["repository"],
                                requested_revision=source["requested_revision"], resolved_revision=source["revision"],
                                destinations=expected_destinations):
            raise HydrationError("Existing dataset completion record does not match the requested dataset; use separate destinations.")
        _validate_complete_dataset(package, destination, record)
        return record
    intent = _dataset_package_intent_path(package)
    expected_intent = _intent_document("dataset", source["repository"], source["requested_revision"], source["revision"])
    if package.exists() or destination.exists():
        existing_intent = _read_document(intent, "dataset partial hydration intent")
        if existing_intent != expected_intent:
            raise HydrationError("Existing dataset destination cannot be proven to match the requested package; it will not be replaced.")
    else:
        _assert_intent(intent, expected_intent)
    if package.exists():
        try:
            manifest = dataset_transport.validate_package(package)
        except Exception as error:
            raise HydrationError("Existing partial dataset package is invalid; inspect it manually rather than replacing it.") from error
    else:
        staging = package.parent / f".{package.name}.hydrating-{source['revision']}"
        if staging.exists() and (staging.is_symlink() or not staging.is_dir()):
            raise HydrationError("Existing dataset staging path is unsafe; inspect it manually.")
        if not staging.exists():
            staging.mkdir()
        manifest_file = staging / dataset_transport.MANIFEST_NAME
        if not manifest_file.exists():
            _download_dataset_file(source, dataset_transport.MANIFEST_NAME, staging)
        try:
            manifest = dataset_transport.read_manifest(staging)
        except Exception as error:
            raise HydrationError("Downloaded dataset manifest is invalid; private staging was retained for inspection.") from error
        expected_files = {dataset_transport.MANIFEST_NAME, *(shard["path"] for shard in manifest["shards"])}
        for name in expected_files:
            if name not in source["inventory"]:
                raise HydrationError("Dataset source is missing a manifest-declared transport file.")
        for shard in manifest["shards"]:
            if source["inventory"][shard["path"]] != shard["size_bytes"]:
                raise HydrationError("Dataset source shard size does not match its manifest.")
        for name in sorted(expected_files):
            path = staging / name
            if not path.is_file() or path.stat().st_size != source["inventory"][name]:
                _download_dataset_file(source, name, staging)
        cache = staging / ".cache"
        if cache.exists():
            shutil.rmtree(cache)
        try:
            manifest = dataset_transport.validate_package(staging)
        except Exception as error:
            raise HydrationError("Downloaded dataset package failed Phase 4.1 validation; private staging was retained for inspection.") from error
        _publish_package(staging, package)
    if destination.exists():
        try:
            dataset_transport.validate_extracted_dataset(destination, manifest)
        except Exception as error:
            raise HydrationError("Existing dataset destination is invalid and will not be replaced.") from error
    else:
        try:
            dataset_transport.extract_package(package, destination)
        except Exception as error:
            raise HydrationError("Dataset extraction failed; no completed destination was exposed.") from error
    completed = _dataset_record(source, package, destination, manifest)
    _write_document_last(record_path, completed)
    return completed


def hydrate(config: dict[str, Any], *, hub: Any = None, environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Hydrate both assets. A failure never starts training or writes a false record."""
    token = _token(environment)
    ensure_workspace_layout(config["workspace_root"])
    model = hydrate_model(config["model"] | {"workspace_root": config["workspace_root"]}, token, hub=hub)
    dataset = hydrate_dataset(config["dataset"] | {"workspace_root": config["workspace_root"]}, token, hub=hub)
    return {"status": "complete", "model": model, "dataset": dataset}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Explicit credential-free JSON asset configuration.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = hydrate(load_config(args.config))
    except HydrationError as error:
        print(f"Asset hydration failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
