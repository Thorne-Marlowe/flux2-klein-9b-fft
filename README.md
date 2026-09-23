# FLUX.2 Klein fine-tuning: Base 9B recovery fork

This fork develops standalone **full-weight FLUX.2 Klein Base 9B training,
diagnostics and checkpoint recovery** alongside the original 4B-oriented code.
A recorded deterministic BF16 experiment reproduced the uninterrupted control's
final model and recoverable state after explicit recovery from checkpoint-2.
This is qualification of the recorded short configuration, not general exact
reproducibility or production-training reliability.

> **Fork notice:** Original 4B examples and AI Toolkit integration are retained
> as upstream material. They are not validated instructions for the fork's 9B
> recovery path. See [Fork Development Status](#fork-development-status) and the
> [qualification record](docs/qualification/deterministic-bf16-a100-2026-09-23.md).

## Standalone Base 9B workflow

- [Environment bootstrap](docs/runpod-bootstrap.md): isolated Python environment,
  locked dependencies, authenticated missing-file downloads and offline preflight.
  Bootstrap never launches training or establishes memory fit.
- [Recovery training](docs/recovery-training.md): explicit opt-in CLI, transactional
  checkpoint publication and fresh-process restoration. The standalone trainer
  uses CLI arguments, not the AI Toolkit YAML files.
- [Determinism diagnostics](docs/determinism-diagnostic.md): bounded traces of
  inputs, RNG, losses, gradients and optimizer observations, plus optional strict
  deterministic execution. Sampled equality is not complete tensor equality.
- [Qualification and comparison](docs/qualification/deterministic-bf16-a100-2026-09-23.md):
  preserved evidence, exact tested settings, manual reproduction commands and the
  CPU-only checkpoint comparison utility.

Use `requirements-smoke.txt` for the documented isolated Python 3.12/CUDA 12.8
standalone environment. Its filename is historical; do not combine it with the
broader upstream `requirements.txt`. Models/data must be provisioned separately
unless explicitly using bootstrap setup. Select Base 9B explicitly; ordinary
training still defaults to 4B when `--model_path` is omitted.

The recovery path uses one visible GPU, BF16, full transformer weights, uncached
fixed-size images, batch/accumulation 1 and workers 0. It supports AdamW or
Adafactor and optional EMA; the qualification used **Adafactor with EMA off**.
The separate one-pair, one-step smoke path also offers AdamW8bit diagnostics;
AdamW8bit is not supported by recovery. Multi-GPU, cached latents, sampling and
trackers are rejected in recovery mode. No LoRA recovery path is implemented.

## Fork Development Status

Implemented work includes Base 9B preflight and component checks, full-transformer
optimizer coverage, frozen-component checks, sequential VAE/text-encoder staging,
one-step smoke tests, gradient/precision/memory diagnostics, optimizer-step evidence,
deterministic data continuation, checkpoint integrity/publication/restoration,
strict-determinism opt-in and read-only exact checkpoint comparison.

**Scoped qualification:** At commit `875a32f`, the supplied single-A100-SXM4-80GB,
deterministic BF16 Base 9B/Adafactor experiment compared four uninterrupted attempts
against two attempts, checkpoint publication, process termination and explicit
fresh-process recovery for attempts 3-4. All 42 final model shards and their
index had matching SHA-256 records; exact semantic optimizer/scheduler/RNG/data
state equality was reported. See the [qualification record](docs/qualification/deterministic-bf16-a100-2026-09-23.md)
for independent checks, evidence limitations and full configuration.

**Still unqualified:** Ordinary nondeterministic exact reproducibility, other
hardware/software/configurations, GPU mid-epoch recovery, EMA recovery on GPU,
longer training reliability and minimum hardware requirements. The historical
nondeterministic trial/control mismatch is not explained by the new result.
Checkpoint manifests retain `qualification='unqualified'`; they do not
self-certify runs. CPU tests verify implementation behavior, not 9B GPU equivalence.

### Development history

- **Phase 1:** Smoke preparation, EMA disabling, Base 9B preflight, optimizer
  coverage and a fixed-size uncached one-pair path.
- **Phase 2A:** Sequential encoding, memory/precision/gradient diagnostics.
- **Phase 2B:** Optimizer-specific step evidence.
- **Recovery development:** Versioned integrity-checked checkpoints, transactional
  publication, exact-state codecs, acknowledged data ordering and continuity tests.
- **Initial qualification:** Publication/restoration succeeded; a nondeterministic
  trial/control mismatch already existed before interruption.
- **Determinism work:** Fresh-run traces, explicit strict backend settings and
  the configuration-specific deterministic BF16 recovery result recorded above.

Git commits preserve the detailed implementation history. The result does not
qualify the AI Toolkit extension, multi-GPU workflow or general fine-tuning.

## Upstream features and 4B reference examples

The original project includes the `klein2` AI Toolkit extension, batch latent
caching, legacy standalone DDP, resolution buckets, EMA and sample-generation
code. These upstream paths are not covered by this fork's qualification.

### Original 4B component description

```text
FLUX.2-klein-base-4B (upstream reference, not the 9B configuration)
  Transformer: Flux2Transformer2DModel (4B)
  Text encoder: Qwen3 4B (frozen)
  VAE: AutoencoderKLFlux2 (BatchNorm + patchification)
  Scheduler: FlowMatchEulerDiscreteScheduler
```

The 9B path validates its own transformer and component metadata; do not infer
9B text-encoder dimensions or memory requirements from the 4B description.

### Original setup and commands (not 9B recovery instructions)

```bash
pip install -r requirements.txt

python scripts/batch_cache_latents.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images --num_gpus 4 --batch_size 8

python scripts/train_klein_standalone.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images --output_dir /path/to/output \
  --batch_size 4 --steps 40000 --lr 3e-5

accelerate launch --num_processes=4 --multi_gpu scripts/train_klein_standalone.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images --output_dir /path/to/output \
  --batch_size 4 --grad_accum 2 --steps 40000 --lr 3e-5
```

Legacy `--resume_from` is separate from the versioned recovery workflow and is
not covered by its qualification; use the documented `--recovery_resume` path
for recovery checkpoints. The previous `train_klein_ddp.py` example referenced
an absent file and has been removed. AI Toolkit requires an external installation;
its [extension](klein2/) and [example YAML](configs/train_fft_klein_base.yaml) are
upstream configuration material, not a verified standalone launch recipe.

## Models Supported

This original table is retained for context. Its checkmarks, LoRA recommendations
and license labels are inherited statements, not this fork's qualification results.
The standalone recovery path accepts Base 9B and rejects known distilled variants.
Consult each model's own terms; this table does not grant rights to model weights.

| Model | Params | License | Recommended |
|-------|--------|---------|-------------|
| FLUX.2-klein-base-4B | 4B | Apache 2.0 | ✅ FFT + LoRA |
| FLUX.2-klein-base-9B | 9B | Non-commercial | ✅ FFT + LoRA |
| FLUX.2-klein-4B (distilled) | 4B | Apache 2.0 | ❌ Not for training |
| FLUX.2-klein-9B (distilled) | 9B | Non-commercial | ❌ Not for training |


## Key Differences from Flux1 (upstream reference)

| Component | Flux1 (dev/schnell) | Flux2 Klein |
|---|---|---|
| Text encoder | CLIP + T5-XXL | Qwen3 |
| VAE | AutoencoderKL (shift_factor) | AutoencoderKLFlux2 (BatchNorm + patchification) |
| Transformer | FluxTransformer2DModel | Flux2Transformer2DModel |
| Position IDs | 3D (H, W, C) | 4D (T, H, W, L) |
| Pipeline | FluxPipeline | Flux2KleinPipeline |

## License information

This repository contains training code and does not redistribute FLUX.2 [klein] Base 9B model weights.

The licensing of this repository's code and the licensing of model weights are separate matters. FLUX.2 [klein] 4B and 4B Base are published by Black Forest Labs under Apache 2.0, while FLUX.2 [klein] 9B and 9B Base are published under the FLUX Non-Commercial License.

This fork targets FLUX.2 [klein] Base 9B. Downloading, using, fine-tuning, or distributing that model or derivatives remains subject to the applicable Black Forest Labs model license. This fork does not modify, replace, or grant additional rights under those terms.

No FLUX.2 [klein] Base 9B model weights are included in this repository.
