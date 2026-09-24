# Runpod environment bootstrap

This prepares a Linux x86-64 Python 3.12 environment for two fresh,
uninterrupted determinism runs. It never starts training, runs historical
qualification/tests, changes a branch, or certifies memory fit or numerical
recovery. No particular GPU model or minimum VRAM is assumed.

## Fresh pod

Use persistent storage for the workspace, model and data. Provide Python 3.12,
Git, a compatible NVIDIA driver and network access. Bootstrap does not install
Python or drivers. Clone the project yourself and select the reviewed commit
on `feature/9b-smoke-test`; bootstrap requires an already clean checkout and
the full immutable commit ID, not `HEAD` or a branch name. Use a commit that
contains the bootstrap. No automatic checkout, pull or reset occurs.

```bash
git clone --branch feature/9b-smoke-test \
  https://github.com/Thorne-Marlowe/flux2-klein-9b-fft.git /workspace/flux2-klein-9b-fft
# Review the checkout and obtain the intended full commit ID independently.
git -C /workspace/flux2-klein-9b-fft status --short
git -C /workspace/flux2-klein-9b-fft rev-parse HEAD
```

Upload exactly two images (`.png`, `.jpg`, `.jpeg`, `.webp`) with unique stems
and matching nonempty UTF-8 `.txt` captions to `/workspace/data/two-pairs`.
Images must be top-level; do not include orphan captions. The actual trainer
dataset decodes and preprocesses both on CPU during bootstrap.

Accept access conditions for
[`black-forest-labs/FLUX.2-klein-base-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
with your Hugging Face account. Supply `HF_TOKEN` using the pod's secret
environment facility, or reuse an existing `hf auth login` credential cache.
There is **no token CLI argument**. Do not put tokens in commands, Git files,
logs or experiment reports. A supplied `HF_TOKEN` takes precedence; invalid
explicit credentials fail rather than falling back silently. Without credentials,
setup stops with login instructions. An interactive `hf auth login` can be run
from the newly created environment if setup stopped at authentication.

```bash
export CUDA_VISIBLE_DEVICES=0
python3.12 /workspace/flux2-klein-9b-fft/scripts/bootstrap_runpod.py setup \
  --workspace /workspace \
  --repo /workspace/flux2-klein-9b-fft \
  --commit FULL_REVIEWED_40_CHARACTER_COMMIT \
  --venv /workspace/venvs/klein-recovery \
  --model /workspace/models/FLUX.2-klein-base-9B \
  --dataset /workspace/data/two-pairs \
  --output /workspace/experiments/baseline-01 \
  --target-size 256
```

All paths are configurable. The output is an empty parent reserved for future
`run-a` and `run-b` directories, not a training invocation. Keep it separate
from the checkout/model/data/environment. Setup creates a venv only if absent
and applies `requirements-smoke.txt` to that isolated venv on every `setup`,
including after an interrupted installation. Pip reuses satisfied versions and
cached artifacts; no force-reinstall, cache purge or automatic venv deletion is
performed. Setup reconciles missing/different versions with the same lock,
then verifies every applicable version and `pip check`. Preflight never installs
or repairs packages. If the venv is damaged and has no Python executable, inspect
it manually or choose a new path. The lock
explicitly requires `torch==2.10.0+cu128` and `torchvision==0.25.0+cu128`, so CPU
builds cannot satisfy it. This is a version lock, not an artifact-hash lock.

Stage transitions and elapsed-time heartbeats are flushed immediately (default
every 15 seconds), even when pip produces no output. Only translated pip events
such as resolving a locked package, downloading an artifact or reusing cache
are streamed; raw errors, URLs and arbitrary subprocess output remain hidden.
A quiet heartbeat is **not** a diagnosis of a stalled download. Pip's socket
inactivity timeout detects network silence separately from the time taken to
download a large wheel or install packages.

Optional setup controls:

```text
--pip-timeout 120 --pip-retries 3 --pip-max-seconds 21600 --progress-interval 15
```

These are the defaults: 120-second socket inactivity timeout, three connection
retries (allowed 0–10), and a six-hour total pip runtime budget. Set the total
budget to `0` to disable it, or increase it for a slow connection. The total
budget is not an inactivity detector. There is no automatic whole-install retry
loop. Failures and interrupts report the stage, elapsed time and recovery steps;
rerun the same setup command to reuse the existing environment and caches.
Bootstrap never removes partial downloads, although pip's own partial-download
retention/resumption depends on its version; completed cache entries and installed
packages are reusable. No dependency installations occur during preflight.

Setup validates credentials and gated repository access before requesting model
payloads. `--model-revision` defaults to `main`, resolved once to an immutable
Hub commit and printed without credentials. Record that public revision and
pass it explicitly on later setup calls. Standard Diffusers assets only are
selected; alternate formats/variants and inference exports are not downloaded.
Existing files of the expected size are reused. A conflicting existing size
fails without replacement. Hub partial downloads under the model directory's
`.cache` are retained for retries; do not delete them. Do not run concurrent
bootstraps against the same model or environment directory.

## Existing workspace / lightweight preflight

Repeat the same command with `preflight` instead of `setup`. Preflight is
offline: no login is needed, no installs/downloads occur, and model weights are
not loaded. It verifies versions, CUDA availability/BF16 support, exactly one
visible GPU, the recovery resolver's model selection, Base 9B metadata,
safetensors headers, and the two decoded image-caption pairs. It reports GPU
identity/free VRAM and disk capacity/free space. It does not hash every weight
byte: matching sizes/valid headers are not proof of weights or provenance.
The Base 9B declaration follows the existing trainer's metadata checks.

Repeated setup or preflight is safe while the reserved output parent is empty.
After an experiment creates outputs, select a new empty output parent; existing
outputs are never overwritten or removed. The script writes no credential or
experiment report. Readiness is printed only after all checks succeed.
Third-party failure details are suppressed to avoid leaking credentials.

Disk headroom for downloads is conservatively checked as twice the missing
payload bytes plus 1 GiB; retained partial files already reduce measured free
space. This is not an estimate of optimizer/checkpoint disk requirements, CPU
RAM peaks or GPU fit. Quotas, concurrent writers, persistent-volume hard links,
checkpoint atomic visibility and power-loss durability still need verification
on the chosen filesystem. No training qualification is implied.

Once ready, separately review the commands in [recovery training](recovery-training.md)
and [determinism diagnostics](determinism-diagnostic.md). Use two **fresh,
uninterrupted** runs with identical settings, immutable source/model/data and
distinct output/trace paths. Bootstrap deliberately generates no training command
and does not replay the interrupted recovery qualification experiment.
