"""Offline readiness checks for the persistent Klein Base 9B environment.

This utility is deliberately not a trainer, downloader, authenticator, or
launcher.  It validates the installed runtime and safe filesystem capabilities
before an operator starts the existing recovery trainer.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, field
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
STATUSES = {PASS, WARN, FAIL}
DETERMINISTIC_CUBLAS_VALUE = ":4096:8"
CORE_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "bitsandbytes": "bitsandbytes",
}
WORKSPACE_DIRECTORIES = ("models", "datasets", "runs", "evidence", "cache")


@dataclass(frozen=True)
class Check:
    identifier: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"Unsupported check status: {self.status}")


@dataclass
class Result:
    profile: str
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(check.status == FAIL for check in self.checks):
            return FAIL
        if any(check.status == WARN for check in self.checks):
            return WARN
        return PASS

    @property
    def exit_code(self) -> int:
        return 1 if self.status == FAIL else 0

    def add(self, identifier: str, status: str, message: str, **details: Any) -> None:
        self.checks.append(Check(identifier, status, message, details))

    def document(self) -> dict[str, Any]:
        return {"profile": self.profile, "status": self.status, "exit_code": self.exit_code,
                "checks": [asdict(check) for check in self.checks]}


def _error_message(error: Exception) -> str:
    """Keep diagnostics concise and avoid leaking arbitrary external output."""
    message = str(error).replace("\n", " ").strip()
    return message[:300] if message else type(error).__name__


def _existing_directory(path: Path) -> Path | None:
    path = path.absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path if path.is_dir() else None


def _temporary_directory(parent: Path, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def probe_writable_directory(directory: Path) -> Check:
    """Exercise a tiny create/write/unlink cycle in the exact directory."""
    try:
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        handle, name = tempfile.mkstemp(prefix=".klein-preflight-write-", dir=directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(b"klein-preflight\n")
        finally:
            Path(name).unlink(missing_ok=True)
        return Check("filesystem.writable", PASS, "Directory accepts temporary writes.",
                     {"directory": str(directory)})
    except OSError as error:
        return Check("filesystem.writable", FAIL, "Directory does not accept a temporary write.",
                     {"directory": str(directory), "error": _error_message(error)})


def probe_hard_link(directory: Path, source: Path | None = None) -> Check:
    """Exercise the resolver's same-filesystem hard-link operation and clean up."""
    probe_root: Path | None = None
    try:
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        probe_root = _temporary_directory(directory, ".klein-preflight-hardlink-")
        if source is None:
            source = probe_root / "source"
            source.write_bytes(b"klein-preflight\n")
            scope = "temporary source in probe directory"
        else:
            source = source.resolve(strict=True)
            if not source.is_file():
                raise FileNotFoundError(source)
            scope = "selected model safetensors source"
        target = probe_root / "linked"
        os.link(source, target)
        if target.stat().st_size != source.stat().st_size:
            raise OSError("hard-link target size differs from source")
        return Check("filesystem.model_hardlink", PASS,
                     "Same-filesystem hard-link probe succeeded.",
                     {"directory": str(directory), "scope": scope})
    except OSError as error:
        return Check("filesystem.model_hardlink", FAIL,
                     "Same-filesystem hard-link probe failed.",
                     {"directory": str(directory), "error": _error_message(error)})
    finally:
        if probe_root is not None:
            shutil.rmtree(probe_root, ignore_errors=True)


def probe_checkpoint_rename(directory: Path) -> Check:
    """Exercise the recovery publisher's actual atomic no-replace primitive."""
    source: Path | None = None
    destination: Path | None = None
    try:
        from scripts.klein_checkpoint import _rename_checkpoint_no_replace
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        # The publisher's staging and final checkpoint roots are direct siblings
        # in destination.parent, so preserve that placement here as well.
        source = _temporary_directory(directory, ".klein-preflight-rename-")
        destination = source.with_name(source.name + "-published")
        (source / "marker").write_bytes(b"klein-preflight\n")
        _rename_checkpoint_no_replace(source, destination)
        if source.exists() or not (destination / "marker").is_file():
            raise OSError("rename did not publish the expected destination")
        return Check("filesystem.checkpoint_rename", PASS,
                     "Checkpoint no-replace rename probe succeeded.",
                     {"directory": str(directory), "primitive": "publisher no-replace rename"})
    except (OSError, ImportError) as error:
        return Check("filesystem.checkpoint_rename", FAIL,
                     "Checkpoint no-replace rename probe failed.",
                     {"directory": str(directory), "error": _error_message(error)})
    finally:
        if source is not None:
            shutil.rmtree(source, ignore_errors=True)
        if destination is not None:
            shutil.rmtree(destination, ignore_errors=True)


def probe_output_checkpoint_rename(output_path: Path) -> Check:
    """Probe the publisher's destination parent without creating an output run.

    Recovery publishes OUTPUT/checkpoint-N and creates its private staging root
    in OUTPUT.  If OUTPUT does not exist yet, a disposable sibling directory is
    the closest safe stand-in: it is created under OUTPUT.parent, then removed.
    """
    stand_in: Path | None = None
    try:
        if output_path.is_dir():
            directory, stand_in_used = output_path, False
        else:
            stand_in = _temporary_directory(output_path.parent, ".klein-preflight-output-")
            directory, stand_in_used = stand_in, True
        check = probe_checkpoint_rename(directory)
        details = dict(check.details)
        details.update({"publication_parent": str(output_path),
                        "probe_directory": str(directory),
                        "stand_in_output_root": stand_in_used})
        return Check("filesystem.output_checkpoint_rename", check.status, check.message, details)
    except OSError as error:
        return Check("filesystem.output_checkpoint_rename", FAIL,
                     "Checkpoint publication parent could not be probed.",
                     {"publication_parent": str(output_path), "error": _error_message(error)})
    finally:
        if stand_in is not None:
            shutil.rmtree(stand_in, ignore_errors=True)


def _locked_core_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    if not lock_path.is_file():
        return versions
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        for name in CORE_PACKAGES:
            prefix = name + "=="
            if line.lower().startswith(prefix):
                versions[name] = line[len(prefix):].split()[0]
    return versions


def add_repository_checks(result: Result, root: Path) -> None:
    required = ("scripts/train_klein_standalone.py", "scripts/klein_recovery_training.py",
                "scripts/klein_model_resolver.py", "scripts/klein_checkpoint.py",
                "requirements-smoke.txt")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        result.add("repository.runtime", FAIL, "Repository runtime files are missing.", missing=missing,
                   repository=str(root))
    else:
        result.add("repository.runtime", PASS, "Repository runtime files are present.", repository=str(root))

    metadata = root / ".container-build-info.json"
    if not metadata.exists():
        result.add("container.build_metadata", WARN,
                   "Container build metadata is absent; this is expected in an ordinary checkout.", path=str(metadata))
        return
    try:
        document = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("metadata is not an object")
        values = {key: document[key] for key in
                  ("source_commit", "environment_version", "requirements_lock_sha256", "repository")}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("required metadata fields are missing or invalid")
        result.add("container.build_metadata", PASS, "Container build metadata is valid.", **values)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        result.add("container.build_metadata", WARN, "Container build metadata is unreadable or malformed.",
                   path=str(metadata), error=_error_message(error))


def add_runtime_checks(result: Result, root: Path, training: bool) -> None:
    expected_python = (3, 12)
    python_status = PASS if sys.version_info[:2] == expected_python else (FAIL if training else WARN)
    result.add("runtime.python", python_status,
               "Python version matches the locked environment." if python_status == PASS else
               "Python 3.12 is required for the locked training environment.",
               actual=platform.python_version(), expected="3.12")

    linux_amd64 = platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}
    platform_status = PASS if linux_amd64 else (FAIL if training else WARN)
    result.add("runtime.platform", platform_status,
               "Platform is Linux amd64." if platform_status == PASS else
               "Recovery checkpoint publication is qualified only on Linux amd64.",
               system=platform.system(), machine=platform.machine())

    lock_versions = _locked_core_versions(root / "requirements-smoke.txt")
    for distribution, module_name in CORE_PACKAGES.items():
        expected = lock_versions.get(distribution)
        try:
            installed = importlib.metadata.version(distribution)
            # Third-party imports occasionally emit banners/warnings. Keep both
            # terminal and JSON output owned by this preflight.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(module_name)
            status = PASS if expected is None or installed == expected else FAIL
            message = "Core package import and locked version match." if status == PASS else "Core package version differs from requirements-smoke.txt."
            result.add(f"dependency.{distribution}", status, message, installed=installed, expected=expected)
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            result.add(f"dependency.{distribution}", FAIL, "Core package cannot be imported.",
                       expected=expected, error=type(error).__name__)

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        details = {"torch_version": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
                   "cuda_available": cuda_available}
        if cuda_available:
            count = torch.cuda.device_count()
            devices = []
            for index in range(count):
                with torch.cuda.device(index):
                    devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                                    "bf16_supported": bool(torch.cuda.is_bf16_supported())})
            details["devices"] = devices
            if training and (count != 1 or not devices[0]["bf16_supported"]):
                result.add("runtime.cuda", FAIL, "Recovery training requires one visible BF16-capable GPU.", **details)
            else:
                result.add("runtime.cuda", PASS, "CUDA device information is available.", **details)
        else:
            result.add("runtime.cuda", FAIL if training else WARN,
                       "CUDA is unavailable; system inspection can continue, but recovery training cannot run.", **details)
    except (ImportError, AttributeError, RuntimeError) as error:
        result.add("runtime.cuda", FAIL, "PyTorch CUDA runtime could not be inspected.", error=_error_message(error))


def add_workspace_checks(result: Result, workspace: Path, training: bool) -> None:
    if not workspace.exists():
        result.add("workspace.root", WARN, "Workspace directory is absent.", workspace=str(workspace))
        return
    if not workspace.is_dir():
        result.add("workspace.root", FAIL, "Workspace path is not a directory.", workspace=str(workspace))
        return
    writable = probe_writable_directory(workspace)
    result.add("workspace.root", writable.status,
               "Workspace directory exists and accepts temporary writes." if writable.status == PASS else writable.message,
               workspace=str(workspace), **writable.details)
    for name in WORKSPACE_DIRECTORIES:
        directory = workspace / name
        if not directory.exists():
            result.add(f"workspace.{name}", WARN, "Expected workspace directory is absent.", path=str(directory))
        elif not directory.is_dir():
            result.add(f"workspace.{name}", FAIL, "Expected workspace path is not a directory.", path=str(directory))
        else:
            writable = probe_writable_directory(directory)
            result.add(f"workspace.{name}", writable.status,
                       "Expected workspace directory is writable." if writable.status == PASS else writable.message,
                       path=str(directory))

    models = workspace / "models"
    if models.is_dir():
        check = probe_hard_link(models)
        status = check.status if training else (WARN if check.status == FAIL else PASS)
        result.add("filesystem.workspace_model_hardlink", status, check.message,
                   **check.details, probe_context="generic workspace/models probe")
    else:
        result.add("filesystem.workspace_model_hardlink", WARN,
                   "Hard-link capability was not probed because workspace/models is absent.", path=str(models))
    runs = workspace / "runs"
    if runs.is_dir():
        check = probe_checkpoint_rename(runs)
        status = check.status if training else (WARN if check.status == FAIL else PASS)
        result.add("filesystem.workspace_checkpoint_rename", status, check.message,
                   **check.details, probe_context="generic workspace/runs probe")
    else:
        result.add("filesystem.workspace_checkpoint_rename", WARN,
                   "Checkpoint rename capability was not probed because workspace/runs is absent.", path=str(runs))


def add_determinism_check(result: Result, required: bool) -> None:
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual == DETERMINISTIC_CUBLAS_VALUE:
        result.add("runtime.deterministic_recovery_environment", PASS,
                   "CUBLAS workspace configuration matches strict deterministic recovery.", value=actual)
    else:
        result.add("runtime.deterministic_recovery_environment", FAIL if required else WARN,
                   "Strict deterministic recovery requires launch-time CUBLAS_WORKSPACE_CONFIG=:4096:8; this preflight does not set it.",
                   expected=DETERMINISTIC_CUBLAS_VALUE, configured=actual, required=required)


def _add_check(result: Result, check: Check, *, identifier: str | None = None, status: str | None = None) -> None:
    result.add(identifier or check.identifier, status or check.status, check.message, **check.details)


def validate_model_path(path: Path) -> tuple[list[str] | None, Check]:
    try:
        from scripts.klein_model_resolver import resolve_model_files
        from scripts.train_klein_standalone import preflight_model_config
        path = path.resolve(strict=True)
        files = list(resolve_model_files(path))
        if any((path / name).stat().st_size <= 0 for name in files):
            raise ValueError("selected model file is empty")
        index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        transformer = json.loads((path / "transformer/config.json").read_text(encoding="utf-8"))
        preflight_model_config(index, transformer, smoke_model_variant="base-9b", model_path=path)
        return files, Check("training.model", PASS, "Offline Base 9B model selection and metadata are valid.",
                            {"path": str(path), "selected_files": len(files)})
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as error:
        return None, Check("training.model", FAIL, "Model path is not usable by the offline recovery resolver.",
                           {"path": str(path), "error": _error_message(error)})


def validate_dataset_path(path: Path) -> Check:
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = sorted(item for item in path.iterdir() if item.suffix.lower() in extensions)
        pairs = [image for image in images if image.with_suffix(".txt").is_file()]
        if not pairs:
            raise ValueError("no top-level image-caption pairs")
        return Check("training.dataset", PASS, "Dataset has top-level image-caption pairs.",
                     {"path": str(path), "image_count": len(images), "pair_count": len(pairs)})
    except (OSError, ValueError) as error:
        return Check("training.dataset", FAIL, "Dataset path has no usable top-level image-caption pairs.",
                     {"path": str(path), "error": _error_message(error)})


def validate_output_path(path: Path) -> tuple[Path | None, Check]:
    try:
        path = path.absolute()
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(path)
        if not path.parent.is_dir():
            raise FileNotFoundError("output parent must already exist for a filesystem probe")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("recovery publisher rejects linked output ancestors")
        return path, Check("training.output", PASS,
                           "Output path has a non-linked publication parent or an existing parent for a disposable stand-in.",
                           {"path": str(path), "publication_parent": str(path),
                            "output_exists": path.is_dir()})
    except (OSError, ValueError) as error:
        return None, Check("training.output", FAIL, "Output path is not usable for recovery checkpoint publication.",
                           {"path": str(path), "error": _error_message(error)})


def add_training_checks(result: Result, args: argparse.Namespace) -> None:
    missing = [name for name in ("model_path", "dataset_path", "output_path") if getattr(args, name) is None]
    if missing:
        result.add("training.arguments", FAIL, "Training profile requires explicit model, dataset, and output paths.", missing=missing)
        return
    files, model = validate_model_path(args.model_path)
    _add_check(result, model)
    _add_check(result, validate_dataset_path(args.dataset_path))
    output_directory, output = validate_output_path(args.output_path)
    _add_check(result, output)
    if files is not None:
        source_name = next((name for name in files if name.endswith(".safetensors")), None)
        if source_name is None:
            result.add("filesystem.model_selected_hardlink", FAIL, "Resolver returned no safetensors source for hard-link probing.")
        else:
            model_root = args.model_path.resolve(strict=True)
            check = probe_hard_link(model_root.parent, model_root / source_name)
            _add_check(result, check, identifier="filesystem.model_selected_hardlink")
    if output_directory is not None:
        _add_check(result, probe_output_checkpoint_rename(output_directory))


def run_preflight(args: argparse.Namespace) -> Result:
    training = args.profile == "training"
    result = Result(args.profile)
    root = REPOSITORY_ROOT
    add_repository_checks(result, root)
    add_runtime_checks(result, root, training)
    add_workspace_checks(result, args.workspace, training)
    add_determinism_check(result, bool(args.deterministic_recovery))
    if training:
        add_training_checks(result, args)
    return result


def render_human(result: Result) -> str:
    lines = [f"Runpod preflight: {result.status} ({result.profile})"]
    for check in result.checks:
        lines.append(f"[{check.status}] {check.identifier}: {check.message}")
        if check.details:
            details = ", ".join(f"{key}={value}" for key, value in sorted(check.details.items()))
            lines.append(f"  {details}")
    lines.append(f"Exit code: {result.exit_code}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("system", "training"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--deterministic-recovery", action="store_true",
                        help="Require the existing CUBLAS_WORKSPACE_CONFIG=:4096:8 launch contract.")
    parser.add_argument("--json", action="store_true", help="Write the structured result to stdout as JSON.")
    args = parser.parse_args(argv)
    if args.deterministic_recovery and args.profile != "training":
        parser.error("--deterministic-recovery requires --profile training")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_preflight(args)
    if args.json:
        print(json.dumps(result.document(), sort_keys=True))
    else:
        print(render_human(result))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
