"""
Batch Latent Caching for Flux2 Klein - Multi-GPU Support

Caches VAE latents to disk in the same format as ai-toolkit,
so they can be directly used for training without re-encoding.

Usage:
  # Single GPU
  python scripts/batch_cache_latents.py --data_dir /path/to/images --num_gpus 1

  # Multi GPU (4x)
  python scripts/batch_cache_latents.py --data_dir /path/to/images --num_gpus 4

  # Custom batch size
  python scripts/batch_cache_latents.py --data_dir /path/to/images --num_gpus 4 --batch_size 16
"""

import argparse
import hashlib
import json
import os
import time
from collections import OrderedDict
from multiprocessing import Process
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm


def get_latent_cache_path(image_path, cache_dir, scale_w, scale_h, crop_x, crop_y, crop_w, crop_h):
    """Generate cache path matching ai-toolkit format."""
    filename = os.path.basename(image_path)
    info = OrderedDict([
        ("filename", filename),
        ("scale_to_width", scale_w),
        ("scale_to_height", scale_h),
        ("crop_x", crop_x),
        ("crop_y", crop_y),
        ("crop_width", crop_w),
        ("crop_height", crop_h),
        ("latent_space_version", "flux1"),
        ("latent_version", 0),
    ])
    hash_input = json.dumps(info, sort_keys=True).encode("utf-8")
    hash_str = __import__("base64").urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode("ascii")
    hash_str = hash_str.replace("=", "")
    stem = Path(image_path).stem
    return os.path.join(cache_dir, f"{stem}_{hash_str}.safetensors")


def load_and_preprocess(image_path, target_size=1024):
    """Load image and resize to target resolution."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Find best bucket (multiples of 16, max target_size)
    aspect = w / h
    candidates = []
    for rw in range(256, target_size + 1, 64):
        for rh in range(256, target_size + 1, 64):
            if rw * rh <= target_size * target_size:
                candidates.append((rw, rh))
    best = min(candidates, key=lambda c: abs(c[0] / c[1] - aspect))
    tw, th = best

    # Resize and center crop
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    img = img.crop((left, top, left + tw, top + th))

    return img, tw, th


def encode_batch(vae, images, device, dtype):
    """Encode a batch of PIL images through Klein VAE."""
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    tensors = torch.stack([transform(img) for img in images]).to(device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(tensors).latent_dist.sample()

        # Klein VAE: patchify + BatchNorm
        # Patchify
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

        # Unpatchify back
        b2, c2, h2, w2 = latents.shape
        latents = latents.reshape(b2, c2 // 4, 2, 2, h2, w2)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(b2, c2 // 4, h2 * 2, w2 * 2)

    return latents


def cache_worker(gpu_id, image_paths, model_path, cache_dir, batch_size, target_size):
    """Worker process for a single GPU."""
    device = f"cuda:{gpu_id}"
    dtype = torch.bfloat16

    print(f"GPU {gpu_id}: Loading VAE...")
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)
    vae = pipe.vae.to(device)
    vae.eval()
    del pipe
    torch.cuda.empty_cache()

    print(f"GPU {gpu_id}: Processing {len(image_paths)} images, batch_size={batch_size}")

    total_time = 0
    cached = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"GPU{gpu_id}"):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        batch_meta = []

        for p in batch_paths:
            try:
                img, tw, th = load_and_preprocess(p, target_size)
                cache_path = get_latent_cache_path(p, cache_dir, tw, th, 0, 0, tw, th)

                if os.path.exists(cache_path):
                    continue

                batch_imgs.append(img)
                batch_meta.append((p, cache_path, tw, th))
            except Exception as e:
                print(f"GPU {gpu_id}: Error loading {p}: {e}")
                continue

        if not batch_imgs:
            continue

        start = time.time()
        latents = encode_batch(vae, batch_imgs, device, dtype)
        elapsed = time.time() - start
        total_time += elapsed

        for j, (path, cache_path, tw, th) in enumerate(batch_meta):
            lat = latents[j].cpu()
            save_file({"latent": lat}, cache_path)
            cached += 1

        if cached % 1000 == 0 and cached > 0:
            avg = total_time / cached
            remaining = (len(image_paths) - cached) * avg
            print(f"GPU {gpu_id}: {cached}/{len(image_paths)}, {avg:.3f}s/img, ETA: {remaining/3600:.1f}h")

        torch.cuda.empty_cache()

    print(f"GPU {gpu_id}: Done! {cached} images in {total_time:.0f}s ({total_time/max(cached,1):.3f}s/img)")


def main():
    parser = argparse.ArgumentParser(description="Batch Latent Caching for Flux2 Klein")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=1024)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "_latent_cache"
    cache_dir.mkdir(exist_ok=True)

    # Find all images
    images = sorted([
        str(p) for p in data_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ])
    print(f"Found {len(images)} images")

    # Filter already cached (rough check by stem)
    cached_stems = set(f.name.split("_")[0] for f in cache_dir.iterdir() if f.suffix == ".safetensors")
    remaining = [p for p in images if Path(p).stem not in cached_stems]
    print(f"Already cached: {len(cached_stems)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All images already cached!")
        return

    if args.num_gpus == 1:
        cache_worker(0, remaining, args.model_path, str(cache_dir), args.batch_size, args.target_size)
    else:
        # Split into chunks per GPU
        chunk_size = len(remaining) // args.num_gpus
        processes = []

        for gpu_id in range(args.num_gpus):
            start = gpu_id * chunk_size
            end = start + chunk_size if gpu_id < args.num_gpus - 1 else len(remaining)
            chunk = remaining[start:end]

            p = Process(target=cache_worker, args=(gpu_id, chunk, args.model_path, str(cache_dir), args.batch_size, args.target_size))
            p.start()
            processes.append(p)
            print(f"Started GPU {gpu_id}: {len(chunk)} images")

        for p in processes:
            p.join()

    print("All caching complete!")


if __name__ == "__main__":
    main()
