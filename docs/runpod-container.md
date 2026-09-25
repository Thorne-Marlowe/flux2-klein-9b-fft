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

## Disposable asset hydration (Phase 4.2)

`scripts/hydrate_runpod_assets.py` prepares a disposable `/workspace` without
starting training. Copy `configs/runpod-assets.example.json` outside the
repository, replace its placeholder immutable revisions and dataset repository,
then provide `HF_TOKEN` through a Runpod Secret or runtime environment:

```bash
HF_TOKEN='provided-by-runtime-secret' \
python scripts/hydrate_runpod_assets.py --config /workspace/assets.json
```

The token is environment-only. The hydrator never performs an interactive
Hugging Face login, prints a token, writes it to a completion record, or accepts
it on the command line. The model request is resolved to a 40-character Hub
commit and uses the existing Base 9B selection, download, gated-access, and
offline validation logic. It downloads only the Diffusers files selected by the
offline resolver and does not allocate model weights on a GPU.

The dataset source is a Hugging Face dataset repository at an explicitly
resolved revision. The hydrator first downloads `dataset-manifest.json`, then
only the plain tar shards declared by that Phase 4.1 manifest. It validates the
unchanged package and reconstructs the normal top-level image-plus-caption
directory consumed by `--data_dir`. See [dataset transport](dataset-transport.md)
for the tar package contract.

The explicit hydration command creates only missing empty standard workspace
parents (`models`, `datasets`, `runs`, `evidence`, and `cache`); it never clears
or replaces their contents.

Successful assets have credential-free records under
`/workspace/.klein-hydration/model.json` and
`/workspace/.klein-hydration/dataset.json`. They record requested and resolved
revisions, local destinations, and selected model or dataset identity details.
The records are written last. On a matching rerun, assets are revalidated and
reused. A destination or record that cannot be proven to match the requested
asset fails without overwrite. Private partial download staging can be reused
only for the same requested immutable source; it never counts as completion.

After hydration, invoke the existing preflight explicitly, then launch the
existing trainer manually when it reports readiness:

```bash
python scripts/runpod_preflight.py --profile training \
  --workspace /workspace \
  --model-path /workspace/models/FLUX.2-klein-base-9B \
  --dataset-path /workspace/datasets/training-dataset \
  --output-path /workspace/runs/experiment
```

`/workspace` remains disposable. A Network Volume can reduce repeated download
time later, but it is not required. Hydration does not export checkpoints or
results, create Pods, start training, or establish a new GPU or
deterministic-recovery qualification.

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

## GHCR publication (Phase 3)

Phase 3 adds `.github/workflows/publish-container.yml`. It builds the existing
Dockerfile once for Linux amd64, loads that image into the GitHub-hosted runner,
validates the loaded image, and only then authenticates to GitHub Container
Registry (GHCR) and pushes that same local image. It never rebuilds after
validation. The registry digest is the deployment identity for a future Runpod
template:

```text
ghcr.io/thorne-marlowe/flux2-klein-9b-fft@sha256:<digest>
```

Tags are discovery aliases and are not deployment identities. Manual
publication from `master` publishes only `sha-<full-40-character-commit>`.
A strict release tag `vX.Y.Z`, whose target is reachable from `origin/master`,
publishes both `sha-<full-40-character-commit>` and `vX.Y.Z`. The workflow does
not publish `latest`, `master`, `main`, development, or branch aliases. It
refuses an already-existing semantic GHCR tag rather than overwriting it.

### Triggers and package setup

The workflow has two manual modes:

```text
Actions → Build and publish Klein container → Run workflow
publish = false  # build and validate the selected ref only
publish = true   # publish, permitted only from master
```

Manual validation has only `contents: read` permission and never logs in to
GHCR, requests OIDC, creates an attestation, or publishes. Version-tag pushes
matching `v*.*.*` enter the publication path but are rejected unless their
names strictly match `vX.Y.Z` and their commits are reachable from `master`.
There are no ordinary branch-push or pull-request triggers.

The workflow uses GitHub's built-in `GITHUB_TOKEN` after validation, with
`packages: write`, `attestations: write`, and `id-token: write` only on the
publication job. No registry token is passed into the Docker build, build
context, validation containers, tests, or BuildKit cache.

After the first publication, set the GHCR package
`ghcr.io/thorne-marlowe/flux2-klein-9b-fft` to **public** in its GitHub package
settings if future Runpod Pods should pull it anonymously. A private package
would instead require separate runtime registry credentials. The Dockerfile's
OCI source label links the package to this repository.

### Build identity and validation

Publication explicitly supplies the immutable Docker Hub multi-platform index
for `python:3.12.3-slim-bookworm`:

```text
python:3.12.3-slim-bookworm@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63
```

The build remains explicitly Linux amd64. The workflow also calculates the
full checked-out Git commit and SHA-256 of the exact `requirements-smoke.txt`
bytes, then passes both values and a non-placeholder environment version to
the Dockerfile. It validates matching OCI labels and
`.container-build-info.json` fields before publication.

The local candidate validation checks Linux amd64 platform, entrypoint command
passthrough and zero-argument idle behavior, `pip check`, core locked imports,
the CUDA 12.8 PyTorch build, system-profile preflight JSON, both relevant CLI
help paths, the runtime-compatible CPU test suite, and the real Linux checkpoint
no-replace rename test. The preserved-qualification-evidence hash test runs
separately against the read-only checked-out source repository, where its
intentional archive dependency is available; it is not skipped or treated as
an image-runtime test. Expected CPU-runner preflight warnings for missing GPU
or `/workspace` are allowed; any preflight `FAIL` is rejected. It also checks
that source, tests, docs, and requirements are present while `.git`, top-level
runtime asset directories, model/checkpoint/archive formats, and common secret
file types are absent. Qualification documentation and its small inventory
remain source documentation; the archived qualification payload is excluded by
`.dockerignore` and is rejected if it appears.

BuildKit's GitHub Actions cache uses the `runpod-linux-amd64` scope. It is only
a speed optimization and contains no model, dataset, runtime cache, or
credential input. The workflow uses native amd64 GitHub runners and does not
configure QEMU.

After all tags resolve to the published digest, GitHub's native
`actions/attest` action creates and pushes a SLSA build-provenance attestation
for that exact digest. SBOM publication is intentionally deferred. The job
summary and publication-job outputs record the immutable image reference,
commit, requirements-lock digest, pinned Python base, environment version,
published tags, validation result, workflow URL, and attestation URL.

### Evidence boundary and reproducibility limits

A successful publication establishes that the digest-pinned Linux amd64 image
was built from the recorded commit with the recorded base-image and lock-file
digests, passed the listed CPU/container gates, and that exact validated image
was pushed to GHCR and provenance-attested. It does not requalify Base 9B GPU
execution, deterministic recovery, Runpod network-volume semantics, or
production training behavior.

`requirements-smoke.txt` pins package versions but not individual wheel hashes.
The image also installs `libgomp1` from a live Debian repository without a
Debian snapshot or package-version pin. Phase 3 therefore provides traceable,
content-addressed deployment through the final registry digest; it does not
guarantee byte-for-byte identical future rebuilds from the same source commit.
