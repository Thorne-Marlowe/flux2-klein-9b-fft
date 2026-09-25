# Persistent Runpod container (Phase 1)

This document describes the first container image for the Klein Base 9B
recovery project. It packages the repository and the existing locked Python
environment so that a new Pod does not reinstall dependencies from a generic
Runpod image. It is a runtime foundation, not a training launcher or a GPU
qualification.

## Image boundary

`Dockerfile` builds a Linux amd64 image from a Python 3.12 slim Debian base.
The base reference is a build argument and release builds should provide its
content digest (for example, `python:3.12.3-slim-bookworm@sha256:...`). The
image creates `/opt/venv` and installs the unchanged `requirements-smoke.txt`
lock with binary wheels only. PyTorch and CUDA packages therefore come from
the existing lock and its configured package indexes; the image does not add
a second CUDA toolkit source or compile packages.

The repository is copied at build time to `/opt/flux2-klein-9b-fft` and that is
the working directory. The image records the source commit, environment
version, and the lock-file digest in OCI labels and in the small
`.container-build-info.json` file. A release process must pass a real commit,
version, and base-image digest rather than the `unknown`/`unversioned` local
build defaults.

The image intentionally excludes model weights, datasets, checkpoints,
outputs, qualification archives, credentials, SSH keys, local caches, and
`.git`. `.dockerignore` also excludes local virtual environments, test caches,
editor state, and common large tensor/archive formats while retaining source,
tests, documentation, and requirements.

## Entrypoint

`/usr/local/bin/klein-entrypoint` prints a safe identity report containing the
image metadata, repository location, and Python version. It never prints the
environment, reads or persists a token, installs packages, downloads assets,
updates Git, or starts training. Arguments are executed unchanged, so the
container can be used for a command such as:

```bash
docker run --rm klein:local python scripts/train_klein_standalone.py --help
```

With no command, the image remains alive with `sleep infinity` so an operator
can inspect it without an implicit workload:

```bash
docker run --rm --name klein-env klein:local
docker stop klein-env
```

The image has no Jupyter, SSH server, FileBrowser, or other service. Those can
be added later only if an operational need is demonstrated.

## Workspace contract

Persistent Runpod storage should be mounted at `/workspace`. A suggested
layout is:

```text
/workspace/
  models/       # downloaded, resolved model files
  datasets/     # image-caption data
  runs/         # checkpoints, outputs, reports and logs
  evidence/     # qualification and diagnostic artifacts
  cache/        # optional Hugging Face/Pip caches
```

Everything under `/workspace` is external to the image and is the operator's
responsibility to back up. Pod-local layers and any unmounted path do not
survive Pod destruction. The image performs no model or dataset discovery at
startup; the runtime preflight below validates those paths before a training
command is launched.

Runtime secrets remain outside the image. In particular, provide
`HF_TOKEN` (or the approved Runpod/Hugging Face credential mechanism) only to
the command that needs it. Never pass credentials as Docker build arguments or
commit them to `/workspace`.

## Build and local inspection

From the repository root, a release-like build supplies the source commit,
environment version, and the lock digest. The base image should also be
specified with a digest when reproducibility matters:

```bash
LOCK_SHA256=$(sha256sum requirements-smoke.txt | cut -d ' ' -f1)
docker build --platform linux/amd64 \
  --build-arg PYTHON_BASE_IMAGE='python:3.12.3-slim-bookworm@sha256:<verified-digest>' \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --build-arg ENVIRONMENT_VERSION='local-phase1' \
  --build-arg REQUIREMENTS_LOCK_SHA256="$LOCK_SHA256" \
  -t klein:local .
```

The build's dependency layer runs `pip check` and imports the locked core
packages, including a CUDA 12.8 PyTorch wheel. It does not need a GPU. Verify
the resulting metadata and command passthrough with:

```bash
docker run -d --rm --name klein-env klein:local
docker logs klein-env
docker stop klein-env
docker run --rm klein:local python scripts/train_klein_standalone.py --help
docker run --rm klein:local python -m pytest -q tests/test_standalone.py
```

The first command is intentionally idle and is stopped explicitly. Mount a
workspace when inspecting persistence:

```bash
docker run --rm -v "$PWD/workspace:/workspace" klein:local \
  python scripts/train_klein_standalone.py --help
```

Phase 1 does not provide a GHCR workflow, Runpod template, model-sync helper,
or automatic startup download. Those are later phases. The
existing `scripts/bootstrap_runpod.py` remains the generic-Pod fallback: it
can create an environment, authenticate to Hugging Face, fetch missing model
files, and validate a qualification dataset. It is not called by this image
entrypoint and should not be run automatically on every container start.

## Runtime preflight (Phase 2)

`scripts/runpod_preflight.py` is an offline readiness check for a started
container or an ordinary checkout. It never installs packages, authenticates,
downloads assets, starts training, or changes trainer settings. It emits a
human-readable report by default, or one JSON document with `--json` for a
future wrapper or automation.

The `system` profile is a low-cost inspection that does not need models or a
dataset:

```bash
python scripts/runpod_preflight.py --profile system
python scripts/runpod_preflight.py --profile system --workspace /workspace --json
```

It reports the Python/platform/runtime location, optional image build metadata,
core locked-package imports and versions, PyTorch/CUDA build information,
visible GPUs when available, `/workspace` directory and temporary-write
status, and the current deterministic-recovery environment setting. A missing
GPU is a warning in this profile so CPU development machines remain useful for
inspection.

The `training` profile requires explicit asset paths and validates only their
structure; it does not allocate the Base 9B model:

```bash
python scripts/runpod_preflight.py --profile training \
  --workspace /workspace \
  --model-path /workspace/models/FLUX.2-klein-base-9B \
  --dataset-path /workspace/datasets/my-dataset \
  --output-path /workspace/runs/test
```

The model check reuses the offline resolver and Base 9B metadata checks. The
dataset check follows the trainer's top-level image-plus-`.txt` caption pairing
contract without decoding every image. The output check requires an existing,
non-linked output directory or parent suitable for the recovery publisher's
explicit checkpoint destinations. A supplied output leaf may be new when its
parent already exists; the preflight does not create it.

The preflight performs tiny, self-cleaning filesystem probes in the locations
that matter. A training model probe creates a hard link to one selected model
safetensors file inside a temporary sibling view beside the model root, matching
the resolver's no-copy selection operation. A training output probe invokes the
publisher's actual Linux no-replace rename primitive on two disposable sibling
directories directly inside the real output directory. If the requested output leaf is
new, it uses and removes a disposable sibling stand-in output directory under
the existing parent, because recovery will create that output leaf before it
publishes checkpoints. System profile probes use existing `/workspace/models` and
`/workspace/runs` directories when present, but label those as generic rather
than proof of a selected asset path. No model file or checkpoint is modified.

Checks have `PASS`, `WARN`, or `FAIL` status. The overall status is `FAIL` if
any check fails, otherwise `WARN` if any warning remains, otherwise `PASS`.
The process exits `0` for `PASS` or `WARN`, `1` for `FAIL`, and argparse uses
its normal exit `2` for malformed command lines.

For an intended strict deterministic recovery invocation, request the existing
launch contract explicitly:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python scripts/runpod_preflight.py --profile training --deterministic-recovery \
  --model-path /workspace/models/FLUX.2-klein-base-9B \
  --dataset-path /workspace/datasets/my-dataset \
  --output-path /workspace/runs/test
```

Without `--deterministic-recovery`, an absent or different
`CUBLAS_WORKSPACE_CONFIG` is reported as a warning. With that flag it is a
failure. The preflight never sets this variable or toggles PyTorch deterministic
algorithms or backend settings.

Passing preflight is operational readiness evidence for the inspected runtime
and paths. It is not a new Base 9B training, numerical-equivalence, or
deterministic-recovery qualification, and it does not replace the existing
deterministic A100 recovery qualification.

## Deterministic recovery relationship

The image packages the code and dependency lock used by the qualified
deterministic recovery workflow, but containerization does not itself qualify
recovery. Keep `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the existing
`--deterministic_recovery` launch contract explicit at runtime when using that
path. The image must not set deterministic algorithms or backend flags
implicitly; the trainer remains the source of truth.

The first usable-image check is a CPU-only build/inspection: verify the image
metadata, repository commit, locked imports, CLI help, and the existing CPU
tests. A later, small GPU preflight may verify the visible driver and CUDA
runtime without repeating the existing deterministic A100 recovery
qualification unless a material dependency or execution change is introduced.

## Current limitations

The lean Python base has not been built in this environment because Docker is
not available here. The exact base digest, registry publishing, Runpod network
volume behavior, GPU visibility, and runtime asset validation therefore remain
to be checked in the image-build phase. No minimum VRAM or new qualification
claim follows from this document.
