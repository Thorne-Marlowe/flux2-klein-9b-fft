# Flux2 Klein Fine-tuning

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

This fork extends the standalone training script with a dedicated verification path for full-weight fine-tuning of FLUX.2 Klein Base 9B.

### Implemented

- Base 9B architecture and component compatibility checks.
- One-step smoke-test mode using an uncached image-caption pair.
- Sequential VAE and text-encoder staging to reduce simultaneous GPU memory usage.
- Verification of transformer parameter coverage and frozen components.
- Gradient, optimizer-state, precision, and GPU-memory diagnostics.
- Optimizer-specific step verification for AdamW, Adafactor, and AdamW8bit.

### Validation status

- 38 CPU tests passed as of Phase 2B.
- AdamW and Adafactor exercised on CPU.
- AdamW8bit verification tested using state fixtures; real CUDA behavior remains unverified.
- Real Klein Base 9B forward, backward, and optimizer execution has not yet been validated.

**Important:** This fork currently provides a preparation and diagnostic path for 9B full fine-tuning. Successful end-to-end 9B training has not yet been demonstrated.

### Development changelog

- **Phase 1 — Smoke-test preparation:** Added explicit EMA disabling, Base 9B preflight checks, full-transformer optimizer coverage, a fixed-size uncached one-pair path, and initial diagnostics and CPU tests.
- **Phase 2A — Staged encoding and numerical diagnostics:** Encoded the image and caption before transformer GPU placement, returned frozen components to CPU, and added memory accounting, optimizer configuration/state reporting, seed and environment metadata, a bounded BF16/FP32 precision probe, and checks before and after gradient clipping.
- **Phase 2B — Optimizer-step evidence:** Required initialized optimizer-specific state and step counters advancing exactly once. Added tests rejecting no-op or insufficient evidence while keeping observed BF16 weight changes separate from step verification.

Git commits retain the detailed technical history. These phases cover the standalone verification path; they do not establish validated 9B training or extend the AI Toolkit integration.

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
