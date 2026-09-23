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
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


class BootstrapError(RuntimeError):
    pass


class Progress:
    """Stage/heartbeat messages contain only bootstrap-owned text, never logs."""
    def __init__(self, interval=15):
        self.interval = interval
        self.stream = sys.stdout  # Keep heartbeats visible through quiet_hub().
        self.name = "startup"
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.worker = None
        self.hint = "Review inputs and rerun; existing files are retained."
        self.activity = None

    def emit(self, message):
        print(f"[{self.name} +{time.monotonic() - self.started:.0f}s] {message}",
              file=self.stream, flush=True)

    def start(self, name, hint="Review inputs and rerun; existing files are retained."):
        if self.worker is not None:
            self.close()
            self.emit("completed")
        self.name, self.hint, self.started = name, hint, time.monotonic()
        self.activity = None
        self.stop = threading.Event()
        self.emit("starting")
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            activity = self.activity
            self.emit(activity() if activity else "still running")

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()
        self.worker = None

    def fail(self, message):
        self.close()
        self.emit(f"FAILED: {message} Recovery: {self.hint}")


def safe_pip_event(line, packages):
    """Translate recognized events, never forward arbitrary pip text/URLs."""
    match = re.match(r"^(Collecting|Requirement already satisfied:) ([A-Za-z0-9_.-]+)(?=[= <\[(]|$)", line.strip())
    if match:
        name = match[2].lower().replace("_", "-")
        if name in packages:
            return "pip: " + ("resolving " if match[1] == "Collecting" else "already installed ") + name
    if "ReadTimeoutError" in line or "ConnectTimeoutError" in line:
        return "pip: network socket timeout reported (distinct from quiet output)"
    for prefix, message in (("Downloading ", "pip: downloading artifact"),
                            ("Using cached ", "pip: reusing cached artifact"),
                            ("Installing collected packages:", "pip: installing resolved packages"),
                            ("Successfully installed ", "pip: installation reported complete; verifying next"),
                            ("WARNING: Retrying", "pip: retrying a network request")):
        if line.lstrip().startswith(prefix):
            return message
    return None


def install_dependencies(python, lock, progress, *, timeout=120, retries=3, max_seconds=21600):
    """Stream bounded safe events. Silence is not evidence of a stalled download.

    Socket inactivity is bounded by pip, not by time since its last log line.
    The optional wall-clock budget is separate and explicitly labelled.
    """
    packages = {m[1].lower().replace("_", "-") for m in
                re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==", lock.read_text())}
    command = [str(python), "-u", "-m", "pip", "install", "--disable-pip-version-check",
               "--progress-bar", "off", "--timeout", str(timeout), "--retries", str(retries),
               "-r", str(lock)]
    progress.emit(f"pip socket timeout={timeout}s; retries={retries}; total budget={max_seconds}s (0 disables).")
    events = queue.Queue(maxsize=64)
    read_failed = threading.Event()
    last_output = [time.monotonic()]
    progress.activity = lambda: (f"pip running; no output for {time.monotonic() - last_output[0]:.0f}s; "
                                 "quiet output alone is not evidence of a stall")
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)

    def read_output():
        # Bounded reads and queue; raw lines are discarded immediately.
        try:
            for line in iter(lambda: process.stdout.readline(4096), ""):
                last_output[0] = time.monotonic()
                message = safe_pip_event(line, packages)
                if message:
                    try:
                        events.put_nowait(message)
                    except queue.Full:
                        pass
        except Exception:
            read_failed.set()  # Never expose raw reader exceptions via a thread traceback.

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while process.poll() is None or reader.is_alive() or not events.empty():
            if read_failed.is_set():
                raise BootstrapError("Pip progress reader failed; installation stopped safely. Rerun setup.")
            if process.poll() is None and max_seconds and time.monotonic() - started >= max_seconds:
                raise BootstrapError("Pip exceeded its total runtime budget; this is not proof of a network stall. Increase --pip-max-seconds if the download is merely slow.")
            try:
                progress.emit(events.get(timeout=0.2))
            except queue.Empty:
                pass
        if read_failed.is_set():
            raise BootstrapError("Pip progress reader failed; dependency verification is required before proceeding.")
        if process.returncode:
            raise BootstrapError(f"Pip exited with code {process.returncode}; check network/index access and disk space. Raw pip output was withheld.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()
        progress.activity = None


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
        raise BootstrapError("Dependency mismatch; rerun setup with the same venv and lockfile: " + "; ".join(mismatches))
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


def download_missing(root, revision, *, hub=None, progress=None):
    if progress:
        progress.start("Hugging Face access", "Verify cached login or pod secret access and gated license approval, then rerun setup.")
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
    if progress:
        progress.start("model download", "Rerun setup with the same revision/model path; completed files and partial Hub downloads are retained.")
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
    parser.add_argument("--pip-timeout", type=int, default=120, help="Pip socket inactivity timeout in seconds (default 120); not total download time")
    parser.add_argument("--pip-retries", type=int, default=3, help="Pip connection retries, 0-10 (default 3)")
    parser.add_argument("--pip-max-seconds", type=int, default=21600, help="Total pip runtime budget (default 21600 / six hours; 0 disables); no log-silence timeout")
    parser.add_argument("--progress-interval", type=int, default=15, help="Heartbeat interval in seconds (default 15)")
    args = parser.parse_args(argv)
    if args.pip_timeout <= 0 or not 0 <= args.pip_retries <= 10 or args.pip_max_seconds < 0 or args.progress_interval <= 0:
        parser.error("Invalid timeout, retry or progress interval")
    progress = Progress(args.progress_interval)
    try:
        progress.start("repository validation", "Review platform, commit, local changes and path safeguards; no checkout changes are automatic.")
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
        progress.start("virtual environment setup", "Rerun setup with the same venv if Python exists; no environment or cache is deleted automatically.")
        if not python.exists():
            if args.mode != "setup":
                raise BootstrapError("Isolated environment missing; run setup first.")
            if args.venv.exists():
                raise BootstrapError("Environment path already exists without Python; choose a new path.")
            run([sys.executable, "-m", "venv", str(args.venv)])
        if Path(sys.prefix).resolve() != args.venv:
            command = [str(python), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
            progress.close()  # The child reports its own stages without duplicate heartbeats.
            return subprocess.call(command)
        if sys.prefix == sys.base_prefix:
            raise BootstrapError("Expected an isolated virtual environment.")
        sys.path.insert(0, str(args.repo))
        if args.mode == "setup":
            progress.start("dependency installation", "Rerun setup with the same venv and lockfile to reuse satisfied dependencies/cached artifacts; inspect network and disk space. Do not delete partial downloads.")
            install_dependencies(python, args.repo / "requirements-smoke.txt", progress,
                                 timeout=args.pip_timeout, retries=args.pip_retries, max_seconds=args.pip_max_seconds)
        progress.start("dependency verification", "Rerun setup to reconcile the existing venv with the unchanged lockfile; preflight never installs packages.")
        versions = dependency_versions(args.repo / "requirements-smoke.txt")
        run([sys.executable, "-m", "pip", "check"])
        print(json.dumps({"git": git, "dependencies": versions,
                          "disk": disk_report({k: getattr(args, k) for k in ("workspace", "model", "dataset", "output")})}, sort_keys=True))
        progress.start("GPU and disk checks", "Check GPU visibility, driver compatibility and filesystem capacity.")
        print(json.dumps({"gpu": gpu_report()}, sort_keys=True))
        progress.start("dataset validation", "Check exactly two readable images and matching nonempty UTF-8 captions.")
        dataset = validate_dataset(args.dataset, args.target_size)
        if args.mode == "setup":
            print(json.dumps({"model_source": download_missing(args.model, args.model_revision, progress=progress)}, sort_keys=True))
        progress.start("final model and output checks", "Check model metadata, missing/ambiguous shards, safetensors headers and output permissions; existing outputs are never replaced.")
        model = validate_model(args.model)
        # Only an empty output parent is created; no reports or training outputs.
        if args.mode == "setup":
            args.output.mkdir(parents=True, exist_ok=True)
        progress.close()
        progress.emit("completed")
        print(json.dumps({"status": "environment_ready", "dataset": dataset, "model": model,
                          "qualification": "Not training qualification, provenance proof or a memory-fit guarantee."}, sort_keys=True))
        return 0
    except BootstrapError as exc:
        progress.fail(str(exc))
        return 1
    except KeyboardInterrupt:
        progress.fail("Interrupted; running installation stopped, environment and caches retained.")
        return 130
    except Exception:
        progress.fail("Operation failed; third-party exception details suppressed to protect credentials.")
        return 1
    finally:
        progress.close()


if __name__ == "__main__":
    raise SystemExit(main())
