"""
Standalone Full Fine-tuning for Flux2 Klein-base 4B/9B

No external framework dependencies (ai-toolkit, kohya, etc.).
Supports gradient checkpointing, AdamW8bit, resolution bucketing,
EMA, sample generation, and Multi-GPU DDP training.

Usage:
  # One-step preparation path; model must already be available locally/cached.
  # Writes smoke_diagnostics.json, skips samples and model checkpoints.
  python scripts/train_klein_standalone.py --smoke_test \
    --model_path /path/to/FLUX.2-klein-base-9B \
    --data_dir /path/to/one-pair --output_dir /path/to/smoke \
    --target_size 256 --optimizer adamw

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
import importlib.metadata
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
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

    def __init__(self, data_dir, target_size=1024, use_cached_latents=False,
                 fixed_size=False):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.use_cached = use_cached_latents
        self.fixed_size = fixed_size
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
        with Image.open(img_path) as source:
            img = source.convert("RGB")
        w, h = img.size
        tw, th = (self.target_size, self.target_size) if self.fixed_size else find_bucket(w, h)

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
        from safetensors.torch import load_file

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
def encode_images_klein(vae, images, device, dtype, *, trace=None):
    """Encode images through Klein VAE with BatchNorm normalization."""
    images = images.to(device, dtype=dtype)
    if trace is None:
        latents = vae.encode(images).latent_dist.sample()
    else:
        trace.emit("vae_input", tensors=[("images", images)], rng=True)
        posterior = vae.encode(images).latent_dist
        trace.emit("vae_posterior", tensors=[("mean", posterior.mean), ("std", posterior.std)], rng=True)
        latents = posterior.sample()
        trace.emit("vae_sample", tensors=[("latents", latents)], rng=True)
        del posterior

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


def preflight_dataset(args):
    if args.smoke_test:
        if not args.model_path or args.target_size is None:
            raise ValueError("Smoke test requires explicit --model_path and --target_size")
        if args.use_cached_latents or args.resume_from:
            raise ValueError("Smoke test requires uncached data and no resume checkpoint")
        args.batch_size = args.grad_accum = args.steps = 1
        args.num_workers = 0
        args.use_ema = False
    for name in ("batch_size", "grad_accum", "steps", "target_size",
                 "save_every", "sample_every", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.target_size % 16:
        raise ValueError("target_size must be divisible by 16 for VAE patchification")
    if not Path(args.data_dir).is_dir():
        raise ValueError(f"Dataset directory does not exist: {args.data_dir}")
    dataset = ImageTextDataset(
        args.data_dir, target_size=args.target_size,
        use_cached_latents=args.use_cached_latents, fixed_size=args.smoke_test,
    )
    if len(dataset) < args.batch_size:
        raise ValueError("Dataset cannot produce a full batch; check image-caption pairs")
    if args.smoke_test:
        if len(dataset) != 1:
            raise ValueError("Smoke test requires exactly one image-caption pair")
        sample = dataset[0]  # Decode before allocating model weights.
        if not sample["caption"]:
            raise ValueError("Smoke test caption must not be empty")
        if not torch.isfinite(sample["pixel_values"]).all():
            raise ValueError("Dataset contains non-finite pixels")
    return dataset


def preflight_model_config(config, transformer_config=None, *, smoke_model_variant=None, model_path=None):
    """Smoke-only identity checks; the selector declares identity, not weight provenance."""
    if config.get("_class_name") != "Flux2KleinPipeline":
        raise ValueError("Selected model must be a Flux2KleinPipeline")
    if "is_distilled" in config and config["is_distilled"] is not False:
        raise ValueError("Selected model must declare is_distilled=false when the field is present")
    transformer_entry = config.get("transformer")
    if (not isinstance(transformer_entry, (list, tuple)) or len(transformer_entry) != 2
            or list(transformer_entry) != ["diffusers", "Flux2Transformer2DModel"]):
        raise ValueError("Selected model must use Flux2Transformer2DModel")
    # Recognized conflicting identities veto even an explicit user declaration.
    # Unknown local names are not evidence either for or against Base provenance.
    conflicting_names = {"flux.2-klein-9b", "flux.2-klein-9b-kv", "flux.2-klein-4b",
                         "flux.2-klein-base-4b", "flux.2-dev"}
    identities = [model_path]
    for metadata in (config, transformer_config or {}):
        for key in ("is_distilled", "guidance_distilled", "timestep_distilled"):
            if key in metadata and metadata[key] is not False:
                raise ValueError(f"Conflicting model metadata: {key}={metadata[key]!r}")
        identities.extend(metadata.get(key) for key in (
            "_name_or_path", "name_or_path", "repo_id", "base_model_name_or_path"))
    for identity in identities:
        if isinstance(identity, (str, Path)):
            parts = str(identity).lower().replace("\\", "/").split("/")
            if any(part in conflicting_names or part in {
                f"models--black-forest-labs--{name}" for name in conflicting_names
            } for part in parts):
                raise ValueError(f"Conflicting model identity: {identity}")
    if transformer_config is not None:
        preflight_9b_architecture(transformer_config)
    if "is_distilled" not in config:
        if smoke_model_variant != "base-9b":
            raise ValueError("Missing is_distilled: smoke test requires --smoke_model_variant base-9b "
                             "as a user declaration, not proof of weight provenance")
        if transformer_config is None:
            raise ValueError("Missing is_distilled requires complete Base 9B transformer metadata")
        # Architecture alone cannot distinguish Base from distilled 9B. Require
        # explicit selection AND all expected architecture fields for this fallback.
        for key in ("_class_name", "patch_size", "guidance_embeds", "mlp_ratio", "axes_dims_rope"):
            if key not in transformer_config:
                raise ValueError(f"Missing is_distilled requires complete Base 9B metadata: {key}")


def preflight_9b_architecture(config):
    if "_class_name" in config and config["_class_name"] != "Flux2Transformer2DModel":
        raise ValueError("Selected transformer must be Flux2Transformer2DModel")
    if "is_distilled" in config and config["is_distilled"] is not False:
        raise ValueError("Transformer is_distilled must be false when present")
    # BFL Klein9BParams, expressed using Diffusers config names:
    # https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py
    expected = {"num_layers": 8, "num_single_layers": 24, "num_attention_heads": 32,
                "attention_head_dim": 128, "joint_attention_dim": 12288, "in_channels": 128}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Smoke test requires Klein Base 9B: {key} must be {value}, got {config.get(key)!r}")
    optional = {"out_channels": 128, "patch_size": 1, "guidance_embeds": False,
                "mlp_ratio": 3.0, "axes_dims_rope": [32, 32, 32, 32]}
    for key, value in optional.items():
        actual = config.get(key)
        if key == "out_channels" and actual is None:
            continue  # Diffusers defaults output channels to input channels.
        if key == "axes_dims_rope" and actual is not None:
            actual = list(actual)
        if key in config and actual != value:
            raise ValueError(f"Incompatible Base 9B metadata: {key}={actual!r}")


def preflight_components(pipe, target_size):
    config = pipe.transformer.config
    if config.in_channels != pipe.vae.config.latent_channels * 4:
        raise ValueError("Transformer input channels do not match patched VAE latents")
    if config.joint_attention_dim != pipe.text_encoder.config.hidden_size * 3:
        raise ValueError("Transformer text width does not match the three Qwen hidden states")
    if pipe.text_encoder.config.num_hidden_layers < 27:
        raise ValueError("Text encoder does not provide hidden state layer 27")
    if getattr(config, "guidance_embeds", False):
        raise ValueError("This trainer expects Klein without guidance embeddings")
    if target_size % (pipe.vae_scale_factor * 2):
        raise ValueError("target_size is incompatible with this VAE's patch size")


def preflight_smoke_runtime(accelerator):
    mode = getattr(accelerator.distributed_type, "value", accelerator.distributed_type)
    state = accelerator.state
    if (mode != "NO" or getattr(state, "deepspeed_plugin", None) is not None
            or getattr(state, "fsdp_plugin", None) is not None):
        raise ValueError(f"Smoke test rejects distributed execution modes, including FSDP/DeepSpeed: {mode}")
    if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
        raise ValueError("Smoke test requires a single CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This trainer requires BF16 support")


def parameter_coverage(model, optimizer):
    named = dict(model.named_parameters())
    expected = {id(p) for p in named.values()}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    actual_ids = set(actual)
    groups = {id(p): i for i, group in enumerate(optimizer.param_groups) for p in group["params"]}
    frozen = [name for name, p in named.items() if not p.requires_grad]
    missing = [name for name, p in named.items() if id(p) not in actual_ids]
    if not expected or frozen or missing or actual_ids != expected or len(actual) != len(actual_ids):
        raise ValueError(f"Invalid optimizer coverage: frozen={frozen}, missing={missing}, "
                         f"extra={len(set(actual) - expected)}, duplicates={len(actual) - len(set(actual))}")
    return {"parameter_tensors": len(named), "total_parameters": sum(p.numel() for p in named.values()),
            "trainable_parameters": sum(p.numel() for p in named.values()),
            "optimizer_parameter_tensors": len(actual),
            "optimizer_membership": {name: groups[id(p)] for name, p in named.items()}}


@torch.no_grad()
def gradient_diagnostics(model):
    missing, nonfinite = [], []
    squared_norm = 0.0
    per_parameter = {}
    for name, p in model.named_parameters():
        entry = {"present": p.grad is not None, "finite": None,
                 "nonzero_values": 0, "numel": p.numel()}
        per_parameter[name] = entry
        if p.grad is None:
            missing.append(name)
            continue
        # Bound temporary diagnostic allocations even for the largest weight matrices.
        entry["finite"] = True
        for chunk in p.grad.detach().reshape(-1).split(1_000_000):
            value = chunk.float()
            finite = torch.isfinite(value)
            entry["nonzero_values"] += torch.count_nonzero(finite & (value != 0)).item()
            if not finite.all().item():
                entry["finite"] = False
            else:
                squared_norm += value.double().square().sum().item()
        if not entry["finite"]:
            nonfinite.append(name)
    return {"missing_gradients": missing, "nonfinite_gradients": nonfinite,
            "gradient_l2_norm_before_clip": math.sqrt(squared_norm) if not nonfinite else None,
            "gradient_parameter_tensors": len(per_parameter) - len(missing),
            "nonzero_gradient_parameter_tensors": sum(e["nonzero_values"] > 0 for e in per_parameter.values()),
            "gradient_details": per_parameter}


@torch.no_grad()
def parameter_probes(model):
    # Small, evenly spaced samples from EVERY tensor; never clone the 9B model.
    probes = {}
    for name, p in model.named_parameters():
        count = min(256, p.numel())
        # Integer arithmetic avoids rounded, out-of-bounds indices on huge tensors.
        indices = torch.arange(count, device=p.device) * (p.numel() - 1) // max(count - 1, 1)
        probes[name] = p.detach().reshape(-1)[indices].float().cpu().clone()
    return probes


def parameter_change_diagnostics(before, after):
    deltas = {name: (after[name] - value).abs() for name, value in before.items()}
    return {"parameter_change_scope": "up to 256 evenly spaced values per parameter tensor",
            "sampled_values": sum(d.numel() for d in deltas.values()),
            "changed_sampled_values": sum(torch.count_nonzero(d).item() for d in deltas.values()),
            "changed_parameter_tensors_in_sample": sum(bool(torch.count_nonzero(d)) for d in deltas.values()),
            "max_sampled_parameter_delta": max((d.max().item() for d in deltas.values() if d.numel()), default=0.0),
            "sampled_parameters_finite": all(torch.isfinite(v).all().item() for v in after.values()),
            "parameter_change_details": {
                name: {"sampled_values": d.numel(), "changed_values": torch.count_nonzero(d).item(),
                       "max_abs_delta": d.max().item() if d.numel() else 0.0}
                for name, d in deltas.items()}}


def optimizer_state_dtypes(optimizer):
    """Inspect metadata only, without copying or materializing optimizer state."""
    counts = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            key = str(value.dtype)
            counts[key] = counts.get(key, 0) + 1
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(optimizer.state)
    return counts


def tensor_memory(tensors):
    """Logical tensor bytes, deduplicated by tensor identity (not allocator storage)."""
    groups, seen = {}, set()
    for tensor in tensors:
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        key = f"{tensor.device}/{tensor.dtype}"
        entry = groups.setdefault(key, {"device": str(tensor.device), "dtype": str(tensor.dtype),
                                        "tensors": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["bytes"] += tensor.numel() * tensor.element_size()
    return {"by_device_dtype": groups, "total_bytes": sum(e["bytes"] for e in groups.values()),
            "scope": "logical tensor bytes; distinct views can share storage"}


def state_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from state_tensors(item)


def training_memory(model, optimizer):
    result = {"parameters": tensor_memory(model.parameters()),
              "gradients": tensor_memory(p.grad for p in model.parameters()),
              "optimizer_state": tensor_memory(state_tensors(optimizer.state))}
    result["total_bytes"] = sum(item["total_bytes"] for item in result.values())
    return result


def optimizer_details(model, optimizer):
    groups = [{key: value for key, value in group.items()
               if key != "params" and isinstance(value, (str, int, float, bool, tuple, list, type(None)))}
              for group in optimizer.param_groups]
    states = {}
    for name, p in model.named_parameters():
        state = optimizer.state.get(p, {})
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            step = step.item() if step.numel() == 1 else None
        states[name] = {"keys": sorted(str(key) for key in state), "step": step,
                        "memory": tensor_memory(state_tensors(state))}
    return {"param_groups": groups, "states_by_parameter": states}


def optimizer_step_snapshot(model, optimizer, kind):
    """Copy scalar evidence only; never clone moment tensors or initialize state.

    Schemas: torch.optim.AdamW; transformers.optimization.Adafactor;
    bitsandbytes.optim.AdamW8bit (Optimizer2State.init_state/update_step).
    Unknown/version-incompatible schemas are insufficient, not assumed successful.
    """
    supported = kind in ("adamw", "adafactor", "adamw8bit")
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    entries = {}
    for name, p in model.named_parameters():
        state = optimizer.state.get(p, {})
        group = groups.get(id(p), {})
        shapes = {}
        if kind == "adamw":
            shapes = {"exp_avg": p.shape, "exp_avg_sq": p.shape}
            if group.get("amsgrad", False):
                shapes["max_exp_avg_sq"] = p.shape
        elif kind == "adafactor":
            shapes = ({"exp_avg_sq_row": p.shape[:-1], "exp_avg_sq_col": p.shape[:-2] + p.shape[-1:]}
                      if p.ndim >= 2 else {"exp_avg_sq": p.shape})
            if group.get("beta1") is not None:
                shapes["exp_avg"] = p.shape
        elif kind == "adamw8bit":
            # Both uint8 and small-tensor FP32 fallback states use these names.
            # Source: bitsandbytes/optim/optimizer.py, Optimizer2State.
            shapes = {"state1": p.shape, "state2": p.shape}
        invalid = [key for key, shape in shapes.items()
                   if not isinstance(state.get(key), torch.Tensor) or state[key].shape != shape]
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            step = step.item() if step.numel() == 1 else None
        if (isinstance(step, bool) or not isinstance(step, (int, float))
                or not math.isfinite(step) or step < 0 or int(step) != step):
            step = None
        entries[name] = {"has_gradient": p.grad is not None, "state_initialized": bool(state),
                         "state_keys": sorted(str(key) for key in state),
                         "counter_available": step is not None, "step": step,
                         "invalid_or_missing_state_tensors": invalid,
                         "state_schema_valid": supported and bool(state) and not invalid and id(p) in groups}
    return {"optimizer": kind, "schema_supported": supported, "parameters": entries}


def verify_optimizer_step(before, after, skipped):
    failures = {}
    supported = before["schema_supported"] and after["schema_supported"] and before["optimizer"] == after["optimizer"]
    for name in before["parameters"].keys() | after["parameters"].keys():
        previous, current = before["parameters"].get(name), after["parameters"].get(name)
        reason = None
        if previous is None or current is None:
            reason = "parameter inventory changed"
        elif not previous["has_gradient"]:
            reason = "no pre-step gradient"
        elif not current["state_schema_valid"]:
            reason = "optimizer state absent, incomplete, or unsupported"
        elif previous["state_initialized"] and (not previous["state_schema_valid"] or previous["step"] is None):
            reason = "pre-existing state has insufficient baseline evidence"
        else:
            baseline = previous["step"] if previous["state_initialized"] else 0
            if current["step"] is None:
                reason = "missing or invalid step counter"
            elif current["step"] != baseline + 1:
                reason = "step counter did not advance exactly once"
        if reason:
            failures[name] = reason
    sufficient = supported and bool(before["parameters"]) and not skipped and not failures
    return {"sufficient": sufficient, "schema_supported": supported, "accelerator_skipped": skipped,
            "reason": ("state initialized and counters advanced once for all parameters" if sufficient else
                       "unsupported optimizer schema" if not supported else
                       "Accelerate reported a skipped update" if skipped else "insufficient per-parameter step evidence"),
            "parameter_failures": failures, "before": before, "after": after,
            "scope": "state/counter transition after synchronized return; independent of stored-weight changes; not proof against a deliberately falsified optimizer"}


def dependency_versions():
    versions = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    for name in ("diffusers", "accelerate", "transformers", "bitsandbytes", "safetensors", "torchvision"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def precision_probe(samples, lr, weight_decay):
    """Tiny CPU AdamW/unit-gradient control, NOT a prediction of the selected optimizer."""
    values = [torch.tensor([0., 0.001, 0.01, 0.1, 1., -0.001, -0.01, -0.1, -1.], device="cpu")]
    names = list(samples)
    # Bound size independently of the number of transformer tensors.
    for index in torch.linspace(0, max(len(names) - 1, 0), min(len(names), 16), device="cpu").long().tolist():
        values.append(samples[names[index]][:8])
    initial = torch.cat(values).to(torch.bfloat16)
    results = []
    for rate in dict.fromkeys((3e-5, lr)):
        parameters = [torch.nn.Parameter(initial.clone()), torch.nn.Parameter(initial.float())]
        for p in parameters:
            optimizer = torch.optim.AdamW([p], lr=rate, weight_decay=weight_decay, foreach=False)
            p.grad = torch.ones_like(p)
            optimizer.step()
        reference_delta = parameters[1].detach() - initial.float()
        stored_delta = parameters[0].detach().float() - initial.float()
        rounded_reference = parameters[1].detach().to(torch.bfloat16)
        results.append({"lr": rate, "weight_decay": weight_decay, "values": initial.float().tolist(),
                        "fp32_deltas": reference_delta.tolist(), "bf16_deltas": stored_delta.tolist(),
                        "fp32_updates": int(torch.count_nonzero(reference_delta)),
                        "lost_on_bf16_storage_cast": int(torch.count_nonzero((reference_delta != 0) & (rounded_reference == initial))),
                        "unchanged_in_bf16_optimizer": int(torch.count_nonzero((reference_delta != 0) & (stored_delta == 0)))})
    return {"scope": "CPU AdamW unit-gradient control using magnitude anchors and bounded model samples; not actual training gradients or an 8-bit/Adafactor reference",
            "default_lr": 3e-5, "results": results}


def require_cpu(model, name):
    if any(t.device.type != "cpu" for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError(f"Smoke staging requires {name} on CPU")


@torch.no_grad()
def stage_smoke_encoding(pipe, dataset, device, dtype, smoke):
    """Encode once; move the SAME pipeline modules back to CPU before training."""
    require_cpu(pipe.transformer, "transformer")
    require_cpu(pipe.vae, "VAE")
    require_cpu(pipe.text_encoder, "text encoder")
    batch = collate_fn([dataset[0]])
    checks = {}
    smoke.start_stage("vae_encoding")
    pipe.vae.to(device, dtype=dtype)
    before = frozen_snapshot(pipe.vae)
    latents = encode_images_klein(pipe.vae, batch["pixel_values"], device, dtype)
    checks["vae"] = frozen_diagnostics(pipe.vae, before)
    pipe.vae.to("cpu")
    require_cpu(pipe.vae, "VAE after encoding")
    # Record the encoding peak before releasing unused allocator blocks.
    smoke.start_stage("text_encoding")
    torch.cuda.empty_cache()
    pipe.text_encoder.to(device, dtype=dtype)
    before = frozen_snapshot(pipe.text_encoder)
    embeds = pipe._get_qwen3_prompt_embeds(
        text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, prompt=batch["captions"],
        device=device, max_sequence_length=256, hidden_states_layers=(9, 18, 27))
    checks["text_encoder"] = frozen_diagnostics(pipe.text_encoder, before)
    pipe.text_encoder.to("cpu")
    require_cpu(pipe.text_encoder, "text encoder after encoding")
    require_cpu(pipe.transformer, "transformer after encoding")
    smoke.data["frozen_encoding_checks"] = checks
    smoke.data["pixel_shape"] = list(batch["pixel_values"].shape)
    smoke.data["retained_conditioning_memory"] = tensor_memory([latents, embeds])
    smoke.start_stage("frozen_offloaded")
    torch.cuda.empty_cache()
    if not all(check["unchanged_under_checks"] for check in checks.values()):
        smoke.write()
        raise RuntimeError("Smoke test failed: frozen component changed during encoding")
    return latents, embeds


def frozen_snapshot(model):
    if any(p.requires_grad or p.grad is not None for p in model.parameters()) or model.training:
        raise ValueError("Frozen component must be in eval mode without trainable parameters or gradients")
    return {"samples": parameter_probes(model),
            "versions": {name: (id(p), p._version) for name, p in model.named_parameters()},
            "buffers": {name: b.detach().cpu().clone() for name, b in model.named_buffers()}}


def frozen_diagnostics(model, before):
    samples = parameter_probes(model)
    versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    changed_parameters = [name for name in before["samples"].keys() | samples.keys()
                          if name not in samples or name not in before["samples"]
                          or not torch.equal(samples[name], before["samples"][name])
                          or versions[name] != before["versions"][name]]
    changed_buffers = [name for name in before["buffers"].keys() | buffers.keys()
                       if name not in buffers or name not in before["buffers"]
                       or not torch.equal(buffers[name].detach().cpu(), before["buffers"][name])]
    frozen = all(not p.requires_grad and p.grad is None for p in model.parameters()) and not model.training
    return {"unchanged_under_checks": frozen and not changed_parameters and not changed_buffers,
            "scope": "parameter samples and mutation counters; exact comparison of all buffers; not a full weight comparison",
            "changed_parameters": sorted(changed_parameters), "changed_buffers": sorted(changed_buffers),
            "parameter_tensors_checked": len(samples), "buffer_tensors_checked": len(buffers),
            "frozen_and_eval": frozen}


class SmokeReport:
    """A few allocator queries at stage boundaries; no CUDA tensors are retained."""
    def __init__(self, args):
        self.args = args
        self.device = None
        self.stage = "initialization"
        self.data = {"model": args.model_path, "target_size": args.target_size,
                     "seed": args.seed, "dependency_versions": dependency_versions(),
                     "passed": False, "optimizer_steps": 0, "optimizer_step_completed": False,
                     "status": "running", "cleanup_completed": False,
                     "optimizer_step_returned": False,
                     "stored_weight_changes_observed": None, "stage_memory": {},
                     "memory_scope": "PyTorch CUDA allocator absolute peaks per stage, not allocation deltas"}

    def measure(self):
        if self.device is None:
            return
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
        # Device-wide usage includes other processes and non-PyTorch allocations;
        # it is an instantaneous boundary observation, NOT a measured device peak.
        # Preserve allocator peaks even if the subsequent device-wide query fails.
        self.data["stage_memory"][self.stage] = memory
        for key in ("allocated", "reserved"):
            self.data[f"peak_gpu_{key}_bytes"] = max(
                self.data.get(f"peak_gpu_{key}_bytes", 0), memory[f"peak_{key}_bytes"])
        free, total = torch.cuda.mem_get_info(self.device)
        memory["device_memory_at_boundary"] = {"free_bytes": free, "total_bytes": total,
                                                "used_bytes": total - free}

    def start_stage(self, stage):
        self.measure()
        self.stage = stage
        if self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)

    def write(self):
        # Only failure reporting may tolerate secondary diagnostic errors.
        try:
            self.measure()
        except Exception as error:
            if self.data["status"] != "failed" or self.data["passed"]:
                raise
            self.data["memory_reporting_error"] = f"{type(error).__name__}: {error}"
        path = Path(self.args.output_dir, "smoke_diagnostics.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        # Flush logging before publishing: a broken stdout must not invalidate an
        # already-published success. Atomic replacement is the final operation.
        print(f"Publishing smoke diagnostics: {path} (passed={self.data['passed']})", flush=True)
        temporary.replace(path)


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
    if getattr(args, "deterministic_recovery", False) and (
            not getattr(args, "recovery", False) or getattr(args, "smoke_test", False)):
        raise ValueError("--deterministic_recovery requires recovery training, not legacy or smoke mode")
    if getattr(args, "recovery", False):
        # Direct script execution must resolve the repository's helper package.
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.klein_recovery_training import train_recovery
        return train_recovery(args)
    smoke = SmokeReport(args) if args.smoke_test else None
    try:
        if smoke is not None:
            # Invalidate a previous run's success before preflight or imports.
            smoke.write()
        result = _train(args, smoke)
        if smoke is not None:
            if not smoke.data.get("step_checks_passed", False):
                raise RuntimeError("Smoke test ended without verified optimizer-step checks")
            smoke.start_stage("finalization")
            smoke.data.update(passed=True, status="completed", cleanup_completed=True)
            smoke.write()
        return result
    except Exception as error:
        if smoke is not None:
            smoke.data.update(passed=False, status="failed",
                              error=("out_of_memory" if isinstance(error, torch.OutOfMemoryError)
                                     else smoke.data.get("error", "unexpected_exception")),
                              exception_type=type(error).__name__, error_message=str(error),
                              failed_stage=smoke.stage, traceback=traceback.format_exc())
            # Do not empty the cache or allocate tensors before capturing the failure peak.
            try:
                smoke.write()
            except Exception as report_error:
                # An unwritable output directory must not hide the original failure.
                print(f"Failed to write smoke diagnostics: {report_error}", file=sys.stderr)
        raise


def _train(args, smoke):
    if smoke is not None:
        smoke.start_stage("preflight")
        if args.optimizer is None:
            raise ValueError("Smoke test requires an explicit --optimizer")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    dataset = preflight_dataset(args)
    if smoke is not None:
        for variable in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
            if os.environ.get(variable, "").lower() in ("true", "1", "yes"):
                raise ValueError(f"Smoke test rejects FSDP/DeepSpeed configuration: {variable}")
    if smoke is not None:
        smoke.start_stage("dependency_imports")
    from accelerate import Accelerator
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline

    # Setup loggers
    log_with = []
    if args.log_dir:
        log_with.append("tensorboard")
    if args.wandb:
        log_with.append("wandb")

    if smoke is not None:
        smoke.start_stage("runtime_setup")
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
    if args.smoke_test:
        preflight_smoke_runtime(accelerator)
        smoke.device = device
        smoke.data["gpu_name"] = torch.cuda.get_device_name(device)
        free, total = torch.cuda.mem_get_info(device)
        smoke.data["gpu_at_start"] = {"capacity_bytes": total, "available_bytes": free}
        torch.cuda.reset_peak_memory_stats(device)
        smoke.start_stage("model_loading")

    # Smoke mode is deliberately offline: no model downloads, even if files are missing.
    load_kwargs = {"local_files_only": True} if args.smoke_test else {}
    if args.smoke_test:
        from diffusers import Flux2Transformer2DModel
        model_config = Flux2KleinPipeline.load_config(args.model_path, **load_kwargs)
        transformer_config = Flux2Transformer2DModel.load_config(
            args.model_path, subfolder="transformer", **load_kwargs)
        preflight_model_config(model_config, transformer_config,
                               smoke_model_variant=args.smoke_model_variant, model_path=args.model_path)
        smoke.data["architecture"] = dict(transformer_config)

    if is_main:
        print(f"Loading Flux2 Klein pipeline... ({num_processes} GPU{'s' if num_processes > 1 else ''})")

    # Each process loads the pipeline independently (DDP: full model per GPU)
    pipe = Flux2KleinPipeline.from_pretrained(args.model_path, torch_dtype=dtype, **load_kwargs)
    if args.smoke_test:
        preflight_9b_architecture(pipe.transformer.config)
        preflight_components(pipe, args.target_size)
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    if smoke is not None:
        smoke.start_stage("device_placement_and_optimizer_setup")
    transformer.requires_grad_(True)
    transformer.train()

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if smoke is not None:
        encoded_latents, encoded_embeds = stage_smoke_encoding(pipe, dataset, device, dtype, smoke)
        # A fresh CPU baseline follows the checked encoding/transfer stage. No GPU weight copies.
        frozen_before = {"vae": frozen_snapshot(vae), "text_encoder": frozen_snapshot(text_encoder)}
        smoke.data["frozen_cpu_memory"] = {
            name: {"parameters": tensor_memory(model.parameters()), "buffers": tensor_memory(model.buffers())}
            for name, model in (("vae", vae), ("text_encoder", text_encoder))}
        smoke.start_stage("device_placement_and_optimizer_setup")
    else:
        # Preserve ordinary training's residency and encoding behavior.
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
            **({"weight_decay": args.weight_decay} if args.smoke_test else {}),
        )
    else:
        optimizer = torch.optim.AdamW(
            transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            **({"foreach": False} if args.smoke_test else {}),
        )
    if smoke is not None:
        smoke.data["optimizer"] = {"name": args.optimizer, "class": type(optimizer).__name__,
                                   "configuration": optimizer_details(transformer, optimizer)["param_groups"]}
        smoke.data["memory_before_prepare"] = training_memory(transformer, optimizer)

    coverage = parameter_coverage(transformer, optimizer)
    if is_main:
        counts = {key: value for key, value in coverage.items() if key != "optimizer_membership"}
        print(f"Preflight: model={args.model_path}, config={dict(transformer.config)}, coverage={counts}")

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
    coverage = parameter_coverage(accelerator.unwrap_model(transformer), optimizer)
    if len(dataloader) == 0:
        raise ValueError("Prepared dataloader has no batches")
    if smoke is not None:
        smoke.data["coverage"] = coverage
        require_cpu(pipe.vae, "pipeline VAE before training")
        require_cpu(pipe.text_encoder, "pipeline text encoder before training")
        smoke.data["memory_after_prepare"] = training_memory(accelerator.unwrap_model(transformer), optimizer)

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
                if smoke is not None:
                    latents, prompt_embeds = encoded_latents, encoded_embeds
                elif "latents" in batch:
                    latents = batch["latents"].to(device, dtype=dtype)
                else:
                    latents = encode_images_klein(vae, batch["pixel_values"], device, dtype)

                # Encode text (text encoder is per-GPU)
                if smoke is None:
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
                if smoke is not None:
                    smoke.start_stage("forward")
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
                if args.smoke_test and not torch.isfinite(loss).item():
                    smoke.data.update(loss=str(loss.item()), error="non-finite loss")
                    smoke.write()
                    raise RuntimeError("Smoke test produced a non-finite loss")

                if smoke is not None:
                    smoke.data["loss"] = loss.item()
                    smoke.start_stage("backward")
                accelerator.backward(loss)
                if args.smoke_test:
                    smoke.start_stage("gradient_diagnostics")
                    unwrapped = accelerator.unwrap_model(transformer)
                    diagnostics = smoke.data
                    diagnostics.update(ema_enabled=ema is not None,
                                       **gradient_diagnostics(unwrapped))
                    diagnostics["gradients_before_clip"] = {
                        **{key: value for key, value in diagnostics.items() if key in (
                            "missing_gradients", "nonfinite_gradients", "gradient_parameter_tensors",
                            "nonzero_gradient_parameter_tensors", "gradient_details")},
                        "l2_norm": diagnostics["gradient_l2_norm_before_clip"]}
                    diagnostics["memory_after_backward"] = training_memory(unwrapped, optimizer)
                    if (diagnostics["missing_gradients"] or diagnostics["nonfinite_gradients"]
                            or diagnostics["gradient_l2_norm_before_clip"] == 0):
                        diagnostics.update(passed=False, optimizer_steps=0)
                        smoke.write()
                        raise RuntimeError(f"Smoke test gradient failure: {diagnostics}")
                    before = parameter_probes(unwrapped)
                    diagnostics["learning_rates_used"] = [float(group["lr"]) for group in optimizer.param_groups]
                    diagnostics["max_grad_norm"] = args.max_grad_norm
                    diagnostics["precision_probe"] = precision_probe(before, args.lr, optimizer.param_groups[0].get("weight_decay", 0.0))
                    smoke.start_stage("gradient_clipping")
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                if smoke is not None:
                    after_clip = gradient_diagnostics(unwrapped)
                    after_clip["l2_norm"] = after_clip.pop("gradient_l2_norm_before_clip")
                    diagnostics["gradients_after_clip"] = after_clip
                    if (after_clip["missing_gradients"] or after_clip["nonfinite_gradients"]
                            or after_clip["nonzero_gradient_parameter_tensors"] == 0):
                        diagnostics.update(passed=False, error="invalid gradients after clipping")
                        smoke.write()
                        raise RuntimeError("Smoke test gradient failure after clipping")
                    smoke.start_stage("optimizer_step")
                    step_before = optimizer_step_snapshot(unwrapped, optimizer, args.optimizer)
                optimizer.step()
                if args.smoke_test:
                    torch.cuda.synchronize(device)
                    diagnostics["optimizer_step_returned"] = True
                    diagnostics["optimizer_step_skipped"] = accelerator.optimizer_step_was_skipped
                    evidence = verify_optimizer_step(
                        step_before, optimizer_step_snapshot(unwrapped, optimizer, args.optimizer),
                        accelerator.optimizer_step_was_skipped)
                    diagnostics["optimizer_step_evidence"] = evidence
                    diagnostics["optimizer_steps"] = int(evidence["sufficient"])
                    diagnostics["optimizer_step_completed"] = evidence["sufficient"]
                    smoke.start_stage("post_step_diagnostics")
                    diagnostics["optimizer_state_dtypes"] = optimizer_state_dtypes(optimizer)
                    diagnostics["optimizer_details_after_step"] = optimizer_details(unwrapped, optimizer)
                    diagnostics["memory_after_step"] = training_memory(unwrapped, optimizer)
                    diagnostics.update(parameter_change_diagnostics(before, parameter_probes(unwrapped)))
                    diagnostics["stored_weight_changes_observed"] = diagnostics["changed_sampled_values"] > 0
                    diagnostics["frozen_components"] = {
                        "vae": frozen_diagnostics(vae, frozen_before["vae"]),
                        "text_encoder": frozen_diagnostics(text_encoder, frozen_before["text_encoder"])}
                    diagnostics["step_checks_passed"] = bool(
                        diagnostics["optimizer_steps"] == 1
                        and diagnostics["sampled_parameters_finite"]
                        and all(d["unchanged_under_checks"] for d in diagnostics["frozen_components"].values())
                    )
                    diagnostics["passed_scope"] = "verified optimizer state/counter transition, finite diagnostics and frozen checks; stored-weight changes reported separately, not a training-quality certification"
                    if not diagnostics["step_checks_passed"]:
                        raise RuntimeError("Smoke test failed: insufficient optimizer-step evidence, invalid updates, or frozen component changed; see report")
                    smoke.start_stage("step_finalization")

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
                if not args.smoke_test and global_step % args.save_every == 0:
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
                if (not args.smoke_test and global_step % args.sample_every == 0 and args.sample_prompts):
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

    if smoke is not None:
        smoke.start_stage("cleanup")
    progress.close()
    accelerator.wait_for_everyone()

    # Final save
    if is_main and not args.smoke_test:
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
    if smoke is not None:
        torch.cuda.synchronize(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Flux2 Klein Standalone FFT")
    # Model
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--smoke_model_variant", choices=["base-9b"], default=None,
                        help="Smoke-only user declaration of Base 9B identity when is_distilled is absent; "
                             "not proof of weight provenance and never overrides conflicting metadata")
    parser.add_argument("--output_dir", type=str, required=True)
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=None)
    parser.add_argument("--use_cached_latents", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Offline Base 9B single-GPU, one-step run with staged frozen encoding; requires explicit --model_path, --target_size (e.g. 256), and --optimizer; disables EMA, caching, resume, samples and checkpoints")
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "adamw8bit", "adafactor"], help="Required explicitly for smoke tests; ordinary training defaults to adamw8bit")
    parser.add_argument("--seed", type=int, default=0, help="Smoke/recovery random seed (legacy training unchanged)")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--recovery", action="store_true", help="Opt-in unqualified v2 checkpoint training; single GPU, uncached fixed-size data, batch/accumulation 1, workers 0, no sampling/trackers")
    parser.add_argument("--deterministic_recovery", action="store_true",
                        help="Recovery-only strict deterministic PyTorch algorithms; requires launch-time CUBLAS_WORKSPACE_CONFIG=:4096:8; unsupported operations fail")
    parser.add_argument("--recovery_resume", help="Explicit complete v2 checkpoint root; no latest discovery or v1 migration")
    parser.add_argument("--recovery_model_variant", choices=["base-9b"], help="User Base 9B declaration when is_distilled is absent; not weight provenance")
    parser.add_argument("--recovery_stop_after", type=int, help="Stop and save at this absolute attempt without changing --steps (schedule horizon)")
    parser.add_argument("--determinism_trace", help="Exclusive JSONL sidecar for a fresh recovery run; no backend changes or extra checkpoints")
    parser.add_argument("--determinism_trace_steps", type=int, default=None, help="Observe first 1-4 attempts (default 2); requires --determinism_trace")
    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="Disable EMA shadow weights")
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

    args = parser.parse_args(argv)
    if args.deterministic_recovery and (not args.recovery or args.smoke_test):
        parser.error("--deterministic_recovery requires recovery training, not legacy or smoke mode")
    if args.determinism_trace_steps is not None and not args.determinism_trace:
        parser.error("--determinism_trace_steps requires --determinism_trace")
    if args.determinism_trace:
        if not args.recovery or args.recovery_resume or args.smoke_test:
            parser.error("--determinism_trace requires a fresh --recovery run, without resume or smoke")
        if args.determinism_trace_steps is not None and not 1 <= args.determinism_trace_steps <= 4:
            parser.error("--determinism_trace_steps must be in [1,4]")
    if args.recovery:
        if args.smoke_test or args.model_path is None or args.target_size is None or args.optimizer is None:
            parser.error("--recovery is separate from smoke and requires explicit --model_path, --target_size and --optimizer")
    elif any(value is not None for value in (args.recovery_resume, args.recovery_model_variant, args.recovery_stop_after)):
        parser.error("--recovery_resume, --recovery_model_variant and --recovery_stop_after require --recovery")
    if args.smoke_test:
        if args.model_path is None or args.target_size is None or args.optimizer is None:
            parser.error("--smoke_test requires explicit --model_path (Klein Base 9B), --target_size (e.g. 256), and --optimizer")
    else:
        if args.smoke_model_variant is not None:
            parser.error("--smoke_model_variant is only valid with --smoke_test")
        if args.model_path is None:
            args.model_path = "black-forest-labs/FLUX.2-klein-base-4B"
        if args.target_size is None:
            args.target_size = 1024
        if args.optimizer is None:
            args.optimizer = "adamw8bit"
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
