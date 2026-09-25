#!/usr/bin/env python3
"""Explicitly export or import a validated recovery checkpoint; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, BinaryIO


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import klein_checkpoint as checkpoint


FORMAT = "klein-recovery-artifact-export"
SCHEMA_VERSION = 1
EXPORT_ROOT = "exports"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CHUNK = 1024 * 1024


class ArtifactTransportError(RuntimeError):
    """A durable checkpoint transport request cannot be completed safely."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _token(environment: dict[str, str] | None = None) -> str:
    token = (environment or os.environ).get("HF_OUTPUT_TOKEN", "").strip()
    if not token:
        raise ArtifactTransportError("Durable checkpoint transport requires HF_OUTPUT_TOKEN through the runtime environment; interactive login and HF_TOKEN fallback are not used.")
    return token


def _safe_relative(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactTransportError(f"{context} must be a non-empty relative path.")
    path = PurePosixPath(value)
    if (path.is_absolute() or "\\" in value or ":" in value or "\x00" in value
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ArtifactTransportError(f"Unsafe {context}: {value!r}")
    return value


def _digest(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(_CHUNK)
        if not chunk:
            return size, digest.hexdigest()
        digest.update(chunk)
        size += len(chunk)


def _file_entry(root: Path, relative: str) -> dict[str, object]:
    _safe_relative(relative, "checkpoint file path")
    path = root / relative
    if path.is_symlink():
        raise ArtifactTransportError(f"Linked checkpoint file is not transportable: {relative}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ArtifactTransportError(f"Checkpoint file must be an unlinked regular file: {relative}")
    with path.open("rb") as handle:
        size, digest = _digest(handle)
    after = path.stat()
    if (size != before.st_size or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino or after.st_nlink != 1):
        raise ArtifactTransportError(f"Checkpoint changed while its transport inventory was read: {relative}")
    return {"path": relative, "size_bytes": size, "sha256": digest}


def checkpoint_inventory(root: Path, *, private_staging: bool = False) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Call the authoritative validator, then describe its exact regular files.

    ``private_staging`` uses the checkpoint publisher's existing internal
    validation entry point.  It applies the same integrity rules while
    retaining the public rule that a ``.incomplete-*`` directory is never a
    resumable checkpoint root.
    """
    candidate = (checkpoint._validate_checkpoint_contents(root) if private_staging
                 else checkpoint.validate_checkpoint(root))
    manifest = checkpoint._plain(candidate.manifest)
    paths = {"manifest.json", *(entry["path"] for entry in manifest["payloads"])}
    inventory = [_file_entry(candidate.root, path) for path in sorted(paths)]
    return manifest, inventory


def transport_identity(inventory: list[dict[str, object]]) -> str:
    projection = [{key: item[key] for key in ("path", "size_bytes", "sha256")} for item in inventory]
    return hashlib.sha256(_canonical_bytes({"format": FORMAT, "files": projection})).hexdigest()


def _equal_inventory(actual: list[dict[str, object]], expected: list[dict[str, object]]) -> bool:
    return actual == expected


def _export_id(manifest: dict[str, Any], identity: str, label: str | None) -> str:
    hint = label or f"attempt-{manifest['progress']['attempted_optimizer_steps']}"
    hint = re.sub(r"[^A-Za-z0-9_-]+", "-", hint).strip("-") or "checkpoint"
    result = f"{hint}-{identity[:24]}"
    if not _EXPORT_ID.fullmatch(result):
        raise ArtifactTransportError("Export label cannot produce a safe remote namespace.")
    return result


def _namespace(export_id: str) -> str:
    if not _EXPORT_ID.fullmatch(export_id):
        raise ArtifactTransportError("export_id must contain only letters, numbers, '_' or '-'.")
    return f"{EXPORT_ROOT}/{export_id}"


def _record_path(export_id: str) -> str:
    return _namespace(export_id) + "/export-record.json"


def _checkpoint_remote_path(export_id: str, relative: str) -> str:
    return _namespace(export_id) + "/checkpoint/" + _safe_relative(relative, "checkpoint file path")


def _source_checkpoint_provenance(manifest: dict[str, Any], source: Path) -> dict[str, object]:
    progress = manifest["progress"]
    return {"source_name": source.name, "checkpoint_id": manifest["checkpoint_id"],
            "attempted_optimizer_steps": progress["attempted_optimizer_steps"],
            "completed_optimizer_steps": progress["completed_optimizer_steps"],
            "skipped_optimizer_steps": progress["skipped_optimizer_steps"]}


def _record(*, repository: str, requested_revision: str, resolved_revision_before_export: str,
            export_id: str, source: Path, manifest: dict[str, Any], inventory: list[dict[str, object]]) -> dict[str, Any]:
    return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "completion": "complete",
            "repository": repository, "requested_revision": requested_revision,
            "resolved_revision_before_export": resolved_revision_before_export,
            "export_id": export_id, "remote_namespace": _namespace(export_id),
            "checkpoint": _source_checkpoint_provenance(manifest, source),
            "transport_identity": transport_identity(inventory), "files": inventory}


def _read_json_bytes(data: bytes, description: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ArtifactTransportError(f"{description} contains a duplicate JSON key.")
            result[key] = value
        return result
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite")))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, ArtifactTransportError) as error:
        if isinstance(error, ArtifactTransportError):
            raise
        raise ArtifactTransportError(f"{description} is not valid UTF-8 JSON.") from error
    if not isinstance(result, dict):
        raise ArtifactTransportError(f"{description} must be a JSON object.")
    return result


def parse_completion_record(data: bytes, *, repository: str, export_id: str) -> dict[str, Any]:
    record = _read_json_bytes(data, "Remote export completion record")
    expected = {"format", "schema_version", "completion", "repository", "requested_revision",
                "resolved_revision_before_export", "export_id", "remote_namespace", "checkpoint",
                "transport_identity", "files"}
    if set(record) != expected or record["format"] != FORMAT or record["schema_version"] != SCHEMA_VERSION:
        raise ArtifactTransportError("Remote export completion record has an unsupported format/schema.")
    if record["completion"] != "complete" or record["repository"] != repository or record["export_id"] != export_id:
        raise ArtifactTransportError("Remote export completion record does not identify the requested export.")
    if record["remote_namespace"] != _namespace(export_id):
        raise ArtifactTransportError("Remote export completion record has an unsafe namespace.")
    if not isinstance(record["requested_revision"], str) or not _COMMIT.fullmatch(record["resolved_revision_before_export"] or ""):
        raise ArtifactTransportError("Remote export completion record has invalid revision provenance.")
    checkpoint_info = record["checkpoint"]
    if not isinstance(checkpoint_info, dict) or set(checkpoint_info) != {"source_name", "checkpoint_id", "attempted_optimizer_steps", "completed_optimizer_steps", "skipped_optimizer_steps"}:
        raise ArtifactTransportError("Remote export completion record has malformed checkpoint provenance.")
    if not isinstance(checkpoint_info["source_name"], str) or not isinstance(checkpoint_info["checkpoint_id"], str):
        raise ArtifactTransportError("Remote export completion record has malformed checkpoint provenance.")
    if any(type(checkpoint_info[name]) is not int or checkpoint_info[name] < 0 for name in
           ("attempted_optimizer_steps", "completed_optimizer_steps", "skipped_optimizer_steps")):
        raise ArtifactTransportError("Remote export completion record has malformed checkpoint provenance.")
    files = record["files"]
    if not isinstance(files, list) or not files:
        raise ArtifactTransportError("Remote export completion record has no file inventory.")
    parsed: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise ArtifactTransportError("Remote export completion record has malformed file inventory.")
        path = _safe_relative(item["path"], "remote checkpoint file path")
        if path in seen or type(item["size_bytes"]) is not int or item["size_bytes"] < 1:
            raise ArtifactTransportError("Remote export completion record has duplicate or invalid file inventory.")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ArtifactTransportError("Remote export completion record has invalid file hash.")
        seen.add(path)
        parsed.append({"path": path, "size_bytes": item["size_bytes"], "sha256": item["sha256"]})
    if parsed != sorted(parsed, key=lambda item: item["path"]):
        raise ArtifactTransportError("Remote export completion record file inventory must be path-sorted.")
    if record["transport_identity"] != transport_identity(parsed):
        raise ArtifactTransportError("Remote export completion record transport identity does not match its inventory.")
    return record


def _same_completed_export(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    """Compare the logical artifact, not non-authoritative source provenance."""
    return all(existing.get(key) == proposed.get(key) for key in
               ("format", "schema_version", "completion", "repository", "requested_revision",
                "export_id", "remote_namespace", "transport_identity", "files"))


def _hub_source(repository: str, revision: str, token: str, hub: Any = None) -> dict[str, Any]:
    if hub is None:
        import huggingface_hub as hub
    try:
        api = hub.HfApi(token=token)
        api.whoami()
        api.auth_check(repo_id=repository, repo_type="model")
        info = api.model_info(repository, revision=revision, files_metadata=True)
    except Exception as error:
        raise ArtifactTransportError("Hugging Face output-repository access failed. Verify HF_OUTPUT_TOKEN write/read access, repository, revision, and network connectivity.") from error
    resolved = getattr(info, "sha", "") or ""
    if not _COMMIT.fullmatch(resolved):
        raise ArtifactTransportError("Hub did not resolve the output repository revision to an immutable commit.")
    names = set()
    for sibling in getattr(info, "siblings", ()):
        name = getattr(sibling, "rfilename", None)
        if isinstance(name, str):
            names.add(name)
    return {"repository": repository, "requested_revision": revision, "revision": resolved,
            "hub": hub, "api": api, "token": token, "remote_names": names}


def _download_remote(source: dict[str, Any], filename: str, directory: Path) -> Path:
    try:
        result = source["hub"].hf_hub_download(repo_id=source["repository"], repo_type="model", filename=filename,
                                                 revision=source["revision"], local_dir=str(directory),
                                                 token=source["token"], force_download=False)
    except Exception as error:
        raise ArtifactTransportError("Required durable checkpoint export file could not be downloaded.") from error
    path = Path(result)
    if not path.is_file():
        raise ArtifactTransportError("Hub download did not produce a regular file.")
    return path


def _verify_remote_entry(source: dict[str, Any], remote: str, entry: dict[str, object], scratch: Path) -> None:
    downloaded = _download_remote(source, remote, scratch)
    with downloaded.open("rb") as handle:
        size, digest = _digest(handle)
    if size != entry["size_bytes"] or digest != entry["sha256"]:
        raise ArtifactTransportError("Existing remote checkpoint file differs from the requested export; it will not be overwritten.")


def _upload(source: dict[str, Any], local: Path, remote: str) -> str | None:
    try:
        result = source["api"].upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                             repo_id=source["repository"], repo_type="model",
                                             revision=source["requested_revision"], token=source["token"],
                                             parent_commit=source["revision"],
                                             commit_message="Preserve validated Klein recovery checkpoint artifact")
    except Exception as error:
        raise ArtifactTransportError("Durable checkpoint upload failed. No completion record was published for an incomplete export.") from error
    value = getattr(result, "oid", None)
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ArtifactTransportError("Hub upload did not report an immutable commit; completion was not certified.")
    source["revision"] = value
    return value


def export_checkpoint(checkpoint_dir: Path, *, repository: str, revision: str = "main", label: str | None = None,
                      hub: Any = None, environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Copy an already-valid checkpoint to one remote namespace, completion last."""
    token = _token(environment)
    root = Path(checkpoint_dir)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactTransportError("checkpoint_dir must be an existing non-linked directory.")
    manifest, inventory = checkpoint_inventory(root)
    identity = transport_identity(inventory)
    export_id = _export_id(manifest, identity, label)
    # Validate all local checkpoint semantics before contacting the remote
    # repository.  Transport never publishes an artifact that the existing
    # recovery validator has not accepted.
    source = _hub_source(repository, revision, token, hub)
    resolved_revision_before_export = source["revision"]
    record = _record(repository=repository, requested_revision=revision,
                     resolved_revision_before_export=resolved_revision_before_export, export_id=export_id,
                     source=root, manifest=manifest, inventory=inventory)
    record_remote = _record_path(export_id)
    with tempfile.TemporaryDirectory(prefix=".klein-export-verify-", dir=root.parent) as scratch_name:
        scratch = Path(scratch_name)
        if record_remote in source["remote_names"]:
            existing = parse_completion_record(_download_remote(source, record_remote, scratch).read_bytes(),
                                               repository=repository, export_id=export_id)
            if not _same_completed_export(existing, record):
                raise ArtifactTransportError("Existing completed remote export conflicts with this checkpoint; it will not be overwritten.")
            # A completion record makes the namespace immutable.  Every
            # recorded payload must already be present and exact; re-uploading
            # it would silently repair a corrupt completed export.
            for entry in inventory:
                _verify_remote_entry(source, _checkpoint_remote_path(export_id, entry["path"]), entry, scratch)
            return {"status": "complete", "repository": repository, "requested_revision": revision,
                    "resolved_revision_before_export": resolved_revision_before_export,
                    "published_revision": source["revision"], "export_id": export_id,
                    "remote_namespace": _namespace(export_id), "transport_identity": identity,
                    "reused": True}
        for entry in inventory:
            remote = _checkpoint_remote_path(export_id, entry["path"])
            if remote in source["remote_names"]:
                _verify_remote_entry(source, remote, entry, scratch)
            else:
                current = _file_entry(root, entry["path"])
                if current != entry:
                    raise ArtifactTransportError("Checkpoint changed before upload; export was not completed.")
                _upload(source, root / entry["path"], remote)
        # A record is the completion signal.  Recheck local bytes immediately
        # before it is made visible, so a changing checkpoint cannot be certified.
        _, final_inventory = checkpoint_inventory(root)
        if not _equal_inventory(final_inventory, inventory):
            raise ArtifactTransportError("Checkpoint changed during export; no completion record was published.")
        if record_remote not in source["remote_names"]:
            local_record = scratch / "export-record.json"
            local_record.write_bytes(_canonical_bytes(record))
            published_revision = _upload(source, local_record, record_remote)
        else:
            published_revision = source["revision"]
    return {"status": "complete", "repository": repository, "requested_revision": revision,
            "resolved_revision_before_export": resolved_revision_before_export, "published_revision": published_revision,
            "export_id": export_id, "remote_namespace": _namespace(export_id),
            "transport_identity": identity, "reused": False}


def _copy_verified(source: Path, target: Path, entry: dict[str, object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        while True:
            chunk = input_handle.read(_CHUNK)
            if not chunk:
                break
            output_handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ArtifactTransportError("Downloaded checkpoint file does not match the remote completion record.")


def _validate_existing_destination(destination: Path, record: dict[str, Any]) -> None:
    _, inventory = checkpoint_inventory(destination)
    if inventory != record["files"]:
        raise ArtifactTransportError("Existing import destination differs from the requested durable export; it will not be replaced.")


def import_checkpoint(*, repository: str, export_id: str, destination: Path, revision: str = "main",
                      hub: Any = None, environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Download a completed export to private staging and atomically expose it."""
    token = _token(environment)
    source = _hub_source(repository, revision, token, hub)
    export_id = _namespace(export_id).split("/", 1)[1]
    destination = Path(destination).absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ArtifactTransportError("destination must be a non-linked directory path.")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ArtifactTransportError("destination parent must be an existing non-linked directory.")
    record_remote = _record_path(export_id)
    with tempfile.TemporaryDirectory(prefix=".klein-import-record-", dir=destination.parent) as scratch_name:
        scratch = Path(scratch_name)
        # Completion record comes first; it determines every subsequent remote read.
        record = parse_completion_record(_download_remote(source, record_remote, scratch).read_bytes(),
                                         repository=repository, export_id=export_id)
        if destination.exists():
            _validate_existing_destination(destination, record)
            return {"status": "complete", "repository": repository, "requested_revision": revision,
                    "resolved_revision": source["revision"], "export_id": export_id,
                    "transport_identity": record["transport_identity"], "destination": str(destination),
                    "reused": True}
        # Match the checkpoint publisher's private staging convention.  The
        # public validator rejects this name, so a failed import can never be
        # passed to the resume path as a completed checkpoint.
        staging = Path(tempfile.mkdtemp(prefix=f".incomplete-artifact-import-{destination.name}-", dir=destination.parent))
        try:
            for entry in record["files"]:
                remote = _checkpoint_remote_path(export_id, entry["path"])
                downloaded = _download_remote(source, remote, scratch)
                _copy_verified(downloaded, staging / entry["path"], entry)
            _, inventory = checkpoint_inventory(staging, private_staging=True)
            if inventory != record["files"]:
                raise ArtifactTransportError("Imported checkpoint inventory differs from its completion record.")
            # The only definition of a valid recovery checkpoint remains this
            # existing authoritative validator, invoked by checkpoint_inventory.
            try:
                checkpoint._rename_checkpoint_no_replace(staging, destination)
            except OSError as error:
                raise ArtifactTransportError("Could not atomically publish the validated imported checkpoint.") from error
        except Exception:
            # Retain private staging for diagnosis. It is never a valid public
            # recovery root because it has not been atomically published.
            raise
    return {"status": "complete", "repository": repository, "requested_revision": revision,
            "resolved_revision": source["revision"], "export_id": export_id,
            "transport_identity": record["transport_identity"], "destination": str(destination),
            "reused": False}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export one validated local recovery checkpoint.")
    export.add_argument("--checkpoint-dir", type=Path, required=True)
    export.add_argument("--repository", required=True)
    export.add_argument("--revision", default="main")
    export.add_argument("--label", help="Optional safe human-readable prefix for the content-derived export ID.")
    imported = commands.add_parser("import", help="Import one completed remote recovery export.")
    imported.add_argument("--repository", required=True)
    imported.add_argument("--export-id", required=True)
    imported.add_argument("--destination", type=Path, required=True)
    imported.add_argument("--revision", default="main")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "export":
            result = export_checkpoint(args.checkpoint_dir, repository=args.repository, revision=args.revision,
                                       label=args.label)
        else:
            result = import_checkpoint(repository=args.repository, export_id=args.export_id,
                                       destination=args.destination, revision=args.revision)
    except ArtifactTransportError as error:
        print(f"Recovery artifact transport failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
