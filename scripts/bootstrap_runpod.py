"""Prepare an existing checkout for the two-fresh-run experiment; never train.

The outer entry point needs only Python 3.12's standard library. Preflight is
offline, does not install anything, and never allocates model weights.
"""
import argparse
import contextlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


class BootstrapError(RuntimeError):
    pass


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes unknown arguments, including an accidental token.
        self.exit(2, "Invalid bootstrap arguments; see --help. Tokens must only come from HF_TOKEN or cached login.\n")


@contextlib.contextmanager
def quiet_hub():
    """Do not forward credential-bearing third-party logs or request errors."""
    previous = logging.root.manager.disable
    logging.disable(sys.maxsize)
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous)


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        # Never echo subprocess output: third-party errors may contain credentials.
        raise BootstrapError("External command failed; check Git/Python/pip configuration (output suppressed).")
    return result.stdout.strip()


def verify_git(repo, expected):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected):
        raise BootstrapError("--commit requires the full reviewed 40-character Git commit ID.")
    if Path(run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"])).resolve() != repo.resolve():
        raise BootstrapError("--repo must be the checkout root.")
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    branch = run(["git", "-C", str(repo), "branch", "--show-current"])
    if head != expected.lower():
        raise BootstrapError("Checkout commit mismatch; review/check out the intended commit manually.")
    if run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"]):
        raise BootstrapError("Checkout has local changes or untracked files; preserve/review them before setup.")
    return {"commit": head, "branch": branch or "(detached)"}


def dependency_versions(lock):
    from packaging.requirements import Requirement
    versions, mismatches = {}, []
    for line in lock.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            actual = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        versions[req.name] = actual
        if actual == "missing" or not req.specifier.contains(actual):
            mismatches.append(f"{req.name}: expected {req.specifier}, found {actual}")
    if mismatches:
        raise BootstrapError("Dependency mismatch; use a new isolated environment: " + "; ".join(mismatches))
    return versions


def validate_dataset(root, target_size):
    from scripts.train_klein_standalone import ImageTextDataset
    import torch
    if not root.is_dir() or target_size <= 0 or target_size % 16:
        raise BootstrapError("Dataset must exist; target size must be positive and divisible by 16.")
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if len(images) != 2 or any(not p.with_suffix(".txt").is_file() for p in images):
        raise BootstrapError("Test dataset must contain exactly two top-level images, each with a .txt caption.")
    if {p.stem for p in root.glob("*.txt")} != {p.stem for p in images} or images[0].stem == images[1].stem:
        raise BootstrapError("Dataset captions must correspond uniquely to the two images; no orphan captions.")
    dataset = ImageTextDataset(root, target_size=target_size, fixed_size=True)
    for i in range(2):
        sample = dataset[i]
        if not sample["caption"] or not torch.isfinite(sample["pixel_values"]).all():
            raise BootstrapError("Dataset contains an empty caption or invalid decoded pixels.")
    return {"pairs": 2, "target_size": target_size}


def validate_model(root):
    from scripts.klein_model_resolver import resolve_model_files
    from scripts.train_klein_standalone import preflight_model_config
    files = resolve_model_files(root)
    if any((root / name).stat().st_size == 0 for name in files):
        raise BootstrapError("Selected model contains an empty file.")
    config = json.loads((root / "model_index.json").read_text())
    transformer = json.loads((root / "transformer/config.json").read_text())
    preflight_model_config(config, transformer, smoke_model_variant="base-9b", model_path=root)
    from safetensors import safe_open
    for name in files:
        if name.endswith(".safetensors"):
            # Validate headers/offsets without loading tensors or hashing all 9B bytes.
            with safe_open(root / name, framework="pt", device="cpu") as handle:
                if not handle.keys():
                    raise BootstrapError("Model contains an empty safetensors payload.")
    return {"selected_files": len(files), "selected_bytes": sum((root / name).stat().st_size for name in files)}


def selected_remote_files(siblings):
    """Only standard Diffusers assets; local resolver remains final authority."""
    fixed = {"model_index.json", "scheduler/scheduler_config.json",
             "text_encoder/generation_config.json"}
    fixed.update("tokenizer/" + n for n in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja"))
    selected = {}
    for entry in siblings:
        name = entry.rfilename
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise BootstrapError("Unsafe remote model filename.")
        parts = path.parts
        weight = False
        if len(parts) == 2 and parts[0] in ("transformer", "vae", "text_encoder"):
            stem = "model" if parts[0] == "text_encoder" else "diffusion_pytorch_model"
            weight = bool(re.fullmatch(re.escape(stem) + r"(?:-\d+-of-\d+)?\.safetensors(?:\.index\.json)?", parts[1]))
            weight = weight or parts[1] == "config.json"
        if name in fixed or weight:
            if type(entry.size) is not int or entry.size <= 0 or name in selected:
                raise BootstrapError("Missing size or duplicate remote model entry.")
            selected[name] = entry.size
    if "model_index.json" not in selected:
        raise BootstrapError("Remote model inventory lacks model_index.json.")
    return selected


def download_missing(root, revision, *, hub=None):
    if hub is None:
        import huggingface_hub as hub
    # HF_TOKEN takes precedence; get_token also understands the CLI credential cache.
    token = os.environ.get("HF_TOKEN", "").strip() or hub.get_token()
    if not token:
        raise BootstrapError("Model download requires authentication: set HF_TOKEN through pod secrets, or run hf auth login; accept the model's gated license first.")
    try:
        # Suppress library output and never include its exceptions in user reports.
        with quiet_hub():
            api = hub.HfApi(token=token)
            api.whoami()
            api.auth_check(repo_id=MODEL_ID, repo_type="model")
            info = api.model_info(MODEL_ID, revision=revision, files_metadata=True)
    except Exception:
        raise BootstrapError("Hugging Face authentication/model access failed. Verify HF_TOKEN or hf auth login, gated license approval, token read permissions and network access.") from None
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise BootstrapError("Hub did not resolve an immutable model revision.")
    inventory = selected_remote_files(info.siblings)
    missing = []
    for name, size in inventory.items():
        path = root / name
        if root.resolve() not in path.parent.resolve().parents and path.parent.resolve() != root.resolve():
            raise BootstrapError("Model component directory escapes the requested model root.")
        if path.is_symlink() and not path.exists():
            raise BootstrapError("Model contains a broken file symlink; repair it explicitly before downloading.")
        if path.exists():
            if not path.is_file() or path.stat().st_size != size:
                raise BootstrapError("Existing model file differs in size from the selected revision; inspect it manually or use a separate model directory. No files were overwritten.")
        else:
            missing.append(name)
    root.mkdir(parents=True, exist_ok=True)
    needed = sum(inventory[n] for n in missing)
    # Local-dir downloads keep partial blobs; allow another full missing-file set
    # plus a reserve. This is download headroom, not a training storage estimate.
    if shutil.disk_usage(root).free < 2 * needed + 1024**3:
        raise BootstrapError("Insufficient free model disk space for missing downloads plus staging/reserve.")
    print(json.dumps({"missing_model_files": len(missing), "missing_model_bytes": needed}, sort_keys=True))
    try:
        with quiet_hub():
            hub.utils.disable_progress_bars()
            for name in missing:
                hub.hf_hub_download(repo_id=MODEL_ID, filename=name, revision=info.sha,
                                    local_dir=str(root), token=token, force_download=False)
    except Exception:
        raise BootstrapError("Model download failed. Partial Hub downloads are retained; rerun setup to resume. Check access, network and disk space.") from None
    for name, size in inventory.items():
        if not (root / name).is_file() or (root / name).stat().st_size != size:
            raise BootstrapError("Downloaded model inventory is incomplete or has incorrect sizes.")
    return {"repository": MODEL_ID, "revision": info.sha, "downloaded_files": len(missing)}


def disk_report(paths):
    report = {}
    for name, path in paths.items():
        existing = path
        while not existing.exists():
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        report[name] = {"capacity_bytes": usage.total, "free_bytes": usage.free}
    return report


def gpu_report():
    import torch
    if not torch.cuda.is_available():
        raise BootstrapError("CUDA is unavailable; check GPU attachment, visibility and the NVIDIA driver.")
    if torch.version.cuda != "12.8":
        raise BootstrapError("Expected the locked CUDA 12.8 PyTorch build.")
    devices = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
            devices.append({"index": i, "name": torch.cuda.get_device_name(i),
                            "free_bytes": free, "capacity_bytes": total,
                            "bf16_supported": torch.cuda.is_bf16_supported()})
    if len(devices) != 1 or not devices[0]["bf16_supported"]:
        raise BootstrapError("Recovery requires one visible BF16-capable GPU; set CUDA_VISIBLE_DEVICES explicitly.")
    return devices


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "preflight"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Empty parent reserved for future run-a/run-b outputs")
    parser.add_argument("--venv", type=Path, help="Default: WORKSPACE/venvs/klein-recovery")
    parser.add_argument("--model-revision", default="main", help="Setup resolves this once to an immutable Hub commit")
    parser.add_argument("--target-size", type=int, default=256)
    args = parser.parse_args(argv)
    stage = "platform, checkout and path checks"
    try:
        if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
            raise BootstrapError("Run bootstrap on Linux x86-64 with Python 3.12 (no Python auto-install).")
        for name in ("workspace", "repo", "model", "dataset", "output"):
            setattr(args, name, getattr(args, name).expanduser().resolve())
        args.venv = (args.venv or args.workspace / "venvs/klein-recovery").expanduser().resolve()
        git = verify_git(args.repo, args.commit)
        if Path(__file__).resolve() != args.repo / "scripts/bootstrap_runpod.py":
            raise BootstrapError("Execute the bootstrap from the verified checkout.")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise BootstrapError("Output parent is not empty; choose a new experiment directory. Nothing was overwritten.")
        if any(args.output == p or args.output in p.parents or p in args.output.parents
               for p in (args.repo, args.model, args.dataset, args.venv)):
            raise BootstrapError("Output must be separate from repository, model, dataset and environment.")
        python = args.venv / "bin/python"
        stage = "isolated Python environment setup"
        if not python.exists():
            if args.mode != "setup":
                raise BootstrapError("Isolated environment missing; run setup first.")
            if args.venv.exists():
                raise BootstrapError("Environment path already exists without Python; choose a new path.")
            run([sys.executable, "-m", "venv", str(args.venv)])
            run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r",
                 str(args.repo / "requirements-smoke.txt")])
        if Path(sys.prefix).resolve() != args.venv:
            command = [str(python), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
            return subprocess.call(command)
        if sys.prefix == sys.base_prefix:
            raise BootstrapError("Expected an isolated virtual environment.")
        sys.path.insert(0, str(args.repo))
        stage = "locked dependency verification"
        versions = dependency_versions(args.repo / "requirements-smoke.txt")
        run([sys.executable, "-m", "pip", "check"])
        print(json.dumps({"git": git, "dependencies": versions,
                          "disk": disk_report({k: getattr(args, k) for k in ("workspace", "model", "dataset", "output")})}, sort_keys=True))
        stage = "CUDA device inspection (check visibility and driver compatibility)"
        print(json.dumps({"gpu": gpu_report()}, sort_keys=True))
        stage = "dataset decoding (check two readable images and UTF-8 captions)"
        dataset = validate_dataset(args.dataset, args.target_size)
        if args.mode == "setup":
            stage = "authenticated model download"
            print(json.dumps({"model_source": download_missing(args.model, args.model_revision)}, sort_keys=True))
        stage = "offline Base 9B model validation (check metadata, missing/ambiguous shards and safetensors headers)"
        model = validate_model(args.model)
        # Only an empty output parent is created; no reports or training outputs.
        if args.mode == "setup":
            stage = "empty output directory creation (check filesystem permissions)"
            args.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"status": "environment_ready", "dataset": dataset, "model": model,
                          "qualification": "Not training qualification, provenance proof or a memory-fit guarantee."}, sort_keys=True))
        return 0
    except BootstrapError as exc:
        print(f"Bootstrap error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(f"Bootstrap failed during {stage}; exception details suppressed to protect credentials.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
