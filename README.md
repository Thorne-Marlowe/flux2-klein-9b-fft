# Flux2 Klein Fine-tuning

> ⚠️ **Fork notice:** This fork is developing a dedicated FLUX.2 Klein Base 9B full-weight training and recovery workflow. The original 4B setup examples below are retained from the upstream project and should not be treated as validated instructions for the fork's 9B recovery workflow. See 'Fork Development Status' for current validation results.

First open-source Full Fine-tuning (FFT) implementation for **FLUX.2-klein-base-4B**.

## Features

- **Full Fine-tuning** of Flux2 Klein-base 4B/9B transformer
- **Batch Latent Caching** with multi-GPU support (dramatically faster than sequential)
- **Multi-GPU Training** support via DDP
- **ai-toolkit extension** (`klein2`) for easy integration
- Supports both **tag captions** and **natural language captions**
- Resolution bucketing (512~1024)
- Gradient checkpointing + AdamW8bit / Adafactor optimizer
- EMA smoothing
- Sample generation during training

## Architecture

```
FLUX.2-klein-base-4B
├── Transformer: Flux2Transformer2DModel (4B params)
├── Text Encoder: Qwen3 4B (frozen)
├── VAE: AutoencoderKLFlux2 (BatchNorm + patchify)
└── Scheduler: FlowMatchEulerDiscreteScheduler
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Batch Latent Caching (Multi-GPU)

```bash
# Cache latents on 4 GPUs simultaneously
python scripts/batch_cache_latents.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images \
  --num_gpus 4 \
  --batch_size 8
```

### 3. Training (ai-toolkit)

```bash
# Single GPU
python -m ai-toolkit run configs/train_fft_klein_base.yaml

# Multi-GPU (coming soon)
accelerate launch --num_processes=2 scripts/train_klein_ddp.py configs/train_fft_klein_base.yaml
```

### 4. Training (Standalone - Single GPU)

```bash
python scripts/train_klein_standalone.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images \
  --output_dir /path/to/output \
  --batch_size 4 \
  --steps 40000 \
  --lr 3e-5
```

### 5. Training (Multi-GPU DDP)

```bash
# 4x GPU DDP training
accelerate launch --num_processes=4 --multi_gpu \
  scripts/train_klein_standalone.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images \
  --output_dir /path/to/output \
  --batch_size 4 \
  --grad_accum 2 \
  --steps 40000 \
  --lr 3e-5

# Resume from checkpoint
accelerate launch --num_processes=4 --multi_gpu \
  scripts/train_klein_standalone.py \
  --model_path black-forest-labs/FLUX.2-klein-base-4B \
  --data_dir /path/to/images \
  --output_dir /path/to/output \
  --resume_from /path/to/output/checkpoint-5000/accelerator_state
```

> With DDP, effective batch size = `batch_size x grad_accum x num_gpus`. For example, `--batch_size 4 --grad_accum 2` on 4 GPUs = effective batch 32.

## Models Supported

The table below is preserved from the original README. Its support checkmarks and recommendations are inherited claims, not results from this fork's validation. See [Fork Development Status](#fork-development-status) for the current evidence.

| Model | Params | License | Recommended |
|-------|--------|---------|-------------|
| FLUX.2-klein-base-4B | 4B | Apache 2.0 | ✅ FFT + LoRA |
| FLUX.2-klein-base-9B | 9B | Non-commercial | ✅ FFT + LoRA |
| FLUX.2-klein-4B (distilled) | 4B | Apache 2.0 | ❌ Not for training |
| FLUX.2-klein-9B (distilled) | 9B | Non-commercial | ❌ Not for training |

> ⚠️ **Do NOT fine-tune the distilled models** (FLUX.2-klein-4B/9B). Training breaks the step distillation. Use the `base` variants instead.

## Fork Development Status

This fork extends the original FLUX.2 Klein fine-tuning implementation with a dedicated verification and recovery path for **full-weight fine-tuning of FLUX.2 Klein Base 9B**.

Development currently focuses on validating training execution, checkpoint publication, and numerical continuity across interrupted and resumed runs.

### Implemented

- Base 9B architecture and component compatibility checks.
- One-step smoke-test mode using an uncached image-caption pair.
- Sequential VAE and text-encoder staging to reduce simultaneous GPU memory usage.
- Verification of transformer parameter coverage and frozen components.
- Gradient, optimizer-state, precision, and GPU-memory diagnostics.
- Optimizer-specific step verification for AdamW, Adafactor, and AdamW8bit.
- Recovery checkpoint publication and fresh-process restoration.
- Recovery metadata, configuration serialization, and checkpoint-shard handling.

### Validation status

**Completed:**

- 104 CPU recovery tests passed in the latest test run.
- Real Klein Base 9B training executed on an NVIDIA A100 80GB GPU.
- A four-step BF16 Adafactor experiment completed using two image-caption pairs.
- An interrupted run published a checkpoint after step 2 and resumed in a fresh Python process to complete steps 3–4.
- An uninterrupted control completed the same four-step configuration.
- Both runs published their expected checkpoints without skipped steps.

**Not yet qualified:**

- Exact numerical equivalence between interrupted and uninterrupted training.
- The trial and control model weights already differed at step 2, before the interruption.
- The first source of divergence has not been identified.
- Full training reliability beyond the short qualification experiment has not been established.
- The minimum hardware requirements for this workflow have not been established.

**Current conclusion:** Checkpoint publication and fresh-process restoration have been demonstrated. Exact numerical recovery remains unqualified.

### Current investigation

The next development task is to establish a reproducible baseline using two identical, uninterrupted training runs.

The investigation will identify the first divergence in training inputs, random-number generation, model state, or optimizer updates before repeating the interrupted-versus-uninterrupted recovery comparison.

Checkpoint restoration logic should not be changed solely on the basis of the existing mismatch.

### Development history

- **Phase 1 — Smoke-test preparation:** Added explicit EMA disabling, Base 9B preflight checks, full-transformer optimizer coverage, a fixed-size uncached one-pair path, and initial diagnostics and CPU tests.
- **Phase 2A — Staged encoding and numerical diagnostics:** Added sequential component staging, memory accounting, optimizer reporting, seed and environment metadata, precision diagnostics, and gradient checks.
- **Phase 2B — Optimizer-step evidence:** Added optimizer-specific state and step-counter verification.
- **Recovery development:** Added checkpoint publication, restoration, continuity tests, and fixes for recovery metadata and serialization.
- **Initial recovery qualification:** Demonstrated checkpoint publication and fresh-process restoration, but identified an unresolved numerical mismatch between trial and control runs.

Git commits retain the detailed implementation history.

This section describes the fork's standalone 9B verification and recovery work. It does not establish validation of the original AI Toolkit integration, multi-GPU training, or the complete training workflow.

## Key Differences from Flux1

| | Flux1 (dev/schnell) | Flux2 Klein |
|--|---------------------|-------------|
| Text Encoder | CLIP + T5-XXL | **Qwen3** |
| VAE | AutoencoderKL (shift_factor) | **AutoencoderKLFlux2 (BatchNorm + patchify)** |
| Transformer | FluxTransformer2DModel | **Flux2Transformer2DModel** |
| Position IDs | 3D (H, W, C) | **4D (T, H, W, L)** |
| Pipeline | FluxPipeline | **Flux2KleinPipeline** |

## License

Apache 2.0 (same as FLUX.2-klein-base-4B)
