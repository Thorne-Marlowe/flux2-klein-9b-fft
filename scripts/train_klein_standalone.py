"""
Standalone Full Fine-tuning for Flux2 Klein-base 4B/9B

No external framework dependencies (ai-toolkit, kohya, etc.).
Supports gradient checkpointing, AdamW8bit, resolution bucketing,
EMA, sample generation, and Multi-GPU DDP training.

Usage:
  # Single GPU
  python scripts/train_klein_standalone.py \
    --model_path black-forest-labs/FLUX.2-klein-base-4B \
    --data_dir /path/to/images \
    --output_dir /path/to/output \
    --batch_size 4 \
    --steps 40000 \
    --lr 3e-5

  # Multi-GPU DDP (4x)
  accelerate launch --num_processes=4 --multi_gpu \
    scripts/train_klein_standalone.py \
    --model_path black-forest-labs/FLUX.2-klein-base-4B \
    --data_dir /path/to/images \
    --output_dir /path/to/output \
    --batch_size 4 \
    --steps 40000 \
    --lr 3e-5
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

BUCKET_SIZES = []
for w in range(256, 1025, 64):
    for h in range(256, 1025, 64):
        if w * h <= 1024 * 1024:
            BUCKET_SIZES.append((w, h))


def find_bucket(w, h):
    aspect = w / h
    return min(BUCKET_SIZES, key=lambda b: abs(b[0] / b[1] - aspect))


class ImageTextDataset(Dataset):
    """Simple dataset: images + .txt captions, with optional pre-cached latents."""

    def __init__(self, data_dir, target_size=1024, use_cached_latents=False):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.use_cached = use_cached_latents
        self.cache_dir = self.data_dir / "_latent_cache"

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.images = sorted(
            p for p in self.data_dir.iterdir() if p.suffix.lower() in exts
        )
        # Filter to only images that have captions
        self.samples = []
        for img_path in self.images:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                self.samples.append((str(img_path), str(txt_path)))

        print(f"Dataset: {len(self.samples)} image-caption pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]

        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        if self.use_cached:
            # Try to load cached latent
            cached = self._find_cached_latent(img_path)
            if cached is not None:
                return {"latent": cached, "caption": caption, "path": img_path}

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        tw, th = find_bucket(w, h)

        scale = max(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        pixel_values = transform(img)

        return {"pixel_values": pixel_values, "caption": caption, "path": img_path}

    def _find_cached_latent(self, img_path):
        if not self.cache_dir.exists():
            return None
        stem = Path(img_path).stem
        for f in self.cache_dir.iterdir():
            if f.name.startswith(stem + "_") and f.suffix == ".safetensors":
                data = load_file(str(f))
                return data["latent"]
        return None


def collate_fn(batch):
    """Collate with same-size grouping (resize to first item)."""
    has_latents = "latent" in batch[0]
    captions = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]

    if has_latents:
        # Latents may differ in size - resize to first
        target_shape = batch[0]["latent"].shape
        latents = []
        for b in batch:
            lat = b["latent"]
            if lat.shape != target_shape:
                lat = F.interpolate(
                    lat.unsqueeze(0),
                    size=target_shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            latents.append(lat)
        return {
            "latents": torch.stack(latents),
            "captions": captions,
            "paths": paths,
        }
    else:
        target_shape = batch[0]["pixel_values"].shape
        pixels = []
        for b in batch:
            pv = b["pixel_values"]
            if pv.shape != target_shape:
                pv = F.interpolate(
                    pv.unsqueeze(0),
                    size=target_shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            pixels.append(pv)
        return {
            "pixel_values": torch.stack(pixels),
            "captions": captions,
            "paths": paths,
        }


# ---------------------------------------------------------------------------
# VAE Encoding (Klein-specific: BatchNorm + patchify)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images_klein(vae, images, device, dtype):
    """Encode images through Klein VAE with BatchNorm normalization."""
    images = images.to(device, dtype=dtype)
    latents = vae.encode(images).latent_dist.sample()

    # Patchify: (B, C, H, W) -> (B, C*4, H/2, W/2)
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(b, c * 4, h // 2, w // 2)

    # BatchNorm normalize
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = (latents - bn_mean) / bn_std

    # Unpatchify back: (B, C*4, H/2, W/2) -> (B, C, H, W)
    b2, c2, h2, w2 = latents.shape
    latents = latents.reshape(b2, c2 // 4, 2, 2, h2, w2)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(b2, c2 // 4, h2 * 2, w2 * 2)

    return latents


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    """Convert timesteps to sigma values for flow matching."""
    sigmas = timesteps / 1000.0
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas.to(dtype=dtype)


def patchify(latents):
    """(B, C, H, W) -> (B, C*4, H/2, W/2)"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify(latents, channels=32):
    """(B, C*4, H/2, W/2) -> (B, C, H, W)"""
    b, c2, h2, w2 = latents.shape
    latents = latents.reshape(b, channels, 2, 2, h2, w2)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, channels, h2 * 2, w2 * 2)


def pack_latents(latents):
    """(B, C, H, W) -> (B, H*W, C) for transformer input."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def unpack_latents(packed, h, w):
    """(B, H*W, C) -> (B, C, H, W)"""
    b, _, c = packed.shape
    return packed.permute(0, 2, 1).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Sample Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(pipeline, prompt, output_path, steps=25, guidance=3.5):
    """Generate a sample image during training."""
    image = pipeline(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=1024,
        width=1024,
    ).images[0]
    image.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(args):
    # Setup loggers
    log_with = []
    if args.log_dir:
        log_with.append("tensorboard")
    if args.wandb:
        log_with.append("wandb")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        log_with=log_with if log_with else None,
        project_dir=args.log_dir,
    )

    device = accelerator.device
    dtype = torch.bfloat16
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes

    if is_main:
        print(f"Loading Flux2 Klein pipeline... ({num_processes} GPU{'s' if num_processes > 1 else ''})")

    # Each process loads the pipeline independently (DDP: full model per GPU)
    pipe = Flux2KleinPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Move frozen models to local device
    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)

    # Optimizer
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            transformer.parameters(),
            lr=args.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    # Dataset
    dataset = ImageTextDataset(
        args.data_dir, target_size=args.target_size, use_cached_latents=args.use_cached_latents
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # LR scheduler with warmup
    warmup_steps = min(args.warmup_steps, args.steps // 10)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps - warmup_steps, eta_min=args.lr * 0.1
    )

    # Prepare with accelerator (wraps transformer in DDP, shards dataloader)
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # EMA (only on main process to save memory on other GPUs)
    ema = None
    if args.use_ema and is_main:
        ema = EMAModel(accelerator.unwrap_model(transformer), decay=args.ema_decay)

    # Training state
    global_step = 0
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        samples_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

    # Resume from checkpoint
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        global_step = int(os.path.basename(args.resume_from).split("-")[-1])
        if is_main:
            print(f"Resumed from step {global_step}")

    if is_main:
        effective_batch = args.batch_size * args.grad_accum * num_processes
        print(f"Training config:")
        print(f"  Model: {args.model_path}")
        print(f"  GPUs: {num_processes}")
        print(f"  Steps: {args.steps}")
        print(f"  Per-GPU batch: {args.batch_size}")
        print(f"  Grad accum: {args.grad_accum}")
        print(f"  Effective batch: {args.batch_size} x {args.grad_accum} x {num_processes} = {effective_batch}")
        print(f"  LR: {args.lr}")
        print(f"  Warmup: {warmup_steps} steps")
        print(f"  Optimizer: {args.optimizer}")
        print(f"  EMA: {args.use_ema} (decay={args.ema_decay})")
        print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"  Dataset: {len(dataset)} samples")
        print(f"  Samples/epoch: {len(dataset)} / {effective_batch} = {len(dataset) // effective_batch} steps")
        print(f"  Wandb: {args.wandb}")

    # Initialize trackers (wandb/tensorboard)
    if is_main and log_with:
        init_kwargs = {}
        if args.wandb:
            init_kwargs["wandb"] = {
                "name": args.wandb_run_name or f"klein_fft_{args.steps}steps",
                "tags": ["klein", "fft", "flux2"],
            }
        accelerator.init_trackers(
            project_name=args.wandb_project or "flux2-klein-finetune",
            config={
                "model": args.model_path,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch": args.batch_size * args.grad_accum * num_processes,
                "lr": args.lr,
                "optimizer": args.optimizer,
                "ema": args.use_ema,
                "dataset_size": len(dataset),
                "num_gpus": num_processes,
            },
            init_kwargs=init_kwargs,
        )

    # Pipeline utilities for packing
    prepare_latent_ids = Flux2KleinPipeline._prepare_latent_ids
    prepare_text_ids = Flux2KleinPipeline._prepare_text_ids
    get_qwen3_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds

    progress = tqdm(
        total=args.steps, initial=global_step, desc="Training",
        disable=not is_main,
    )

    loss_accumulator = 0.0
    log_steps = 0

    while global_step < args.steps:
        transformer.train()
        for batch in dataloader:
            if global_step >= args.steps:
                break

            with accelerator.accumulate(transformer):
                # Encode images (VAE is per-GPU, no communication needed)
                if "latents" in batch:
                    latents = batch["latents"].to(device, dtype=dtype)
                else:
                    latents = encode_images_klein(vae, batch["pixel_values"], device, dtype)

                # Encode text (text encoder is per-GPU)
                with torch.no_grad():
                    prompt_embeds = get_qwen3_embeds(
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        prompt=batch["captions"],
                        device=device,
                        max_sequence_length=256,
                        hidden_states_layers=(9, 18, 27),
                    )

                # Flow matching: sample timestep and noise
                bsz = latents.shape[0]
                u = torch.sigmoid(torch.randn(bsz, device=device))
                timesteps = (u * 1000).long().clamp(0, 999)
                sigmas = get_sigmas(timesteps, n_dim=4, dtype=dtype)

                noise = torch.randn_like(latents)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # Patchify + pack for transformer
                noisy_patched = patchify(noisy_latents)
                latent_ids = prepare_latent_ids(noisy_patched).to(device)
                noisy_packed = pack_latents(noisy_patched)

                txt_ids = prepare_text_ids(prompt_embeds).to(device)

                # Forward pass (DDP handles gradient sync)
                noise_pred = transformer(
                    hidden_states=noisy_packed,
                    timestep=timesteps.float() / 1000.0,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

                # Unpack + unpatchify
                h_half = noisy_latents.shape[2] // 2
                w_half = noisy_latents.shape[3] // 2
                noise_pred = unpack_latents(noise_pred, h_half, w_half)
                noise_pred = unpatchify(noise_pred, channels=latents.shape[1])

                # Flow matching target: noise - latents
                target = noise - latents

                # MSE loss
                loss = F.mse_loss(noise_pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()

                # LR warmup
                if global_step < warmup_steps:
                    warmup_factor = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = args.lr * warmup_factor
                else:
                    lr_scheduler.step()

                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                current_loss = loss.item()
                loss_accumulator += current_loss
                log_steps += 1

                progress.update(1)
                progress.set_postfix(
                    loss=f"{current_loss:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    gpu=num_processes,
                )

                # EMA update (main process only)
                if ema is not None:
                    ema.update(accelerator.unwrap_model(transformer))

                # Save checkpoint (all processes wait via barrier)
                if global_step % args.save_every == 0:
                    if is_main:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(transformer)

                        if ema is not None:
                            ema.apply(unwrapped)

                        unwrapped.save_pretrained(
                            os.path.join(save_path, "transformer"),
                            safe_serialization=True,
                        )

                        if ema is not None:
                            ema.restore(unwrapped)

                        # Also save full accelerator state for resuming
                        accelerator.save_state(
                            os.path.join(save_path, "accelerator_state")
                        )

                        avg_loss = loss_accumulator / max(log_steps, 1)
                        print(f"\nStep {global_step}: saved checkpoint (avg_loss={avg_loss:.4f})")
                        loss_accumulator = 0.0
                        log_steps = 0

                    accelerator.wait_for_everyone()

                # Generate sample (main process only, others wait)
                if (global_step % args.sample_every == 0 and args.sample_prompts):
                    if is_main:
                        unwrapped = accelerator.unwrap_model(transformer)
                        unwrapped.eval()
                        if ema is not None:
                            ema.apply(unwrapped)

                        try:
                            sample_pipe = Flux2KleinPipeline(
                                scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                                    args.model_path, subfolder="scheduler"
                                ),
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                vae=vae,
                                transformer=unwrapped,
                            )
                            samples_dir = os.path.join(args.output_dir, "samples")
                            for pi, prompt in enumerate(args.sample_prompts):
                                out_path = os.path.join(samples_dir, f"step{global_step}_p{pi}.png")
                                with torch.autocast(device_type="cuda", dtype=dtype):
                                    generate_sample(sample_pipe, prompt, out_path)
                                print(f"  Sample {pi}: {out_path}")
                            del sample_pipe
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"  Sample generation failed: {e}")
                            import traceback
                            traceback.print_exc()
                        unwrapped.train()

                        if ema is not None:
                            ema.restore(unwrapped)

                    accelerator.wait_for_everyone()

                # Log
                if global_step % args.log_every == 0 and is_main:
                    avg_loss = loss_accumulator / max(log_steps, 1)
                    print(f"Step {global_step}/{args.steps} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                    if log_with:
                        accelerator.log({
                            "train/loss": avg_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step": global_step,
                        }, step=global_step)

    progress.close()
    accelerator.wait_for_everyone()

    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        if ema is not None:
            ema.apply(unwrapped)
        unwrapped.save_pretrained(
            os.path.join(save_path, "transformer"),
            safe_serialization=True,
        )
        print(f"\nTraining complete! Final model saved to {save_path}")

    if is_main and log_with:
        accelerator.end_training()
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="Flux2 Klein Standalone FFT")
    # Model
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--output_dir", type=str, required=True)
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--use_cached_latents", action="store_true")
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw8bit", choices=["adamw", "adamw8bit", "adafactor"])
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint dir to resume from")
    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # Logging & Saving
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--sample_every", type=int, default=2500)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--sample_prompts", type=str, nargs="*", default=[
        "1girl, white hair, blue eyes, school uniform, looking at viewer",
        "1boy, black hair, red eyes, dark fantasy armor, standing in rain",
    ])
    # Wandb
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="flux2-klein-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
