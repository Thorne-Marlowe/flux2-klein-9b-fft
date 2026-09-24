"""Read-only verification of a Diffusers transformer recovery artifact.

The default operation loads only the transformer directory.  Supplying
``--model-path`` additionally reconstructs the original Klein pipeline and runs
one small inference; this mode requires the original model files and a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _weight_files(root: Path) -> list[Path]:
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        document = json.loads(index.read_text(encoding="utf-8"))
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Transformer index has no weight_map")
        names = sorted(set(weight_map.values()))
    else:
        names = ["diffusion_pytorch_model.safetensors"]
    files = []
    for name in names:
        path = root / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Missing or unsafe transformer shard: {name}")
        files.append(path)
    return files


def verify_transformer_artifact(artifact, *, expected_tensors=None):
    """Load and validate a transformer using Diffusers' public loader.

    Shards are inspected one at a time with ``safe_open``.  ``expected_tensors``
    is a small-test hook for detecting an intentionally altered artifact; normal
    callers should leave it unset.
    """
    import torch
    from diffusers import Flux2Transformer2DModel
    from safetensors import safe_open

    root = Path(artifact)
    if not (root / "config.json").is_file():
        raise ValueError("Transformer artifact is missing config.json")
    files = _weight_files(root)
    model = Flux2Transformer2DModel.from_pretrained(root, local_files_only=True)
    tensors = model.state_dict()
    seen = set()
    checked = 0
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name not in tensors:
                    raise ValueError(f"Artifact tensor is not accepted by Diffusers: {name}")
                value = shard.get_tensor(name)
                target = tensors[name]
                if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                    raise ValueError(f"Tensor metadata mismatch for {name}")
                if not torch.equal(value, target.cpu()):
                    raise ValueError(f"Loaded tensor value mismatch for {name}")
                if expected_tensors is not None and name in expected_tensors and not torch.equal(value, expected_tensors[name]):
                    raise ValueError(f"Expected tensor value mismatch for {name}")
                seen.add(name)
                checked += 1
    missing = set(tensors) - seen
    if missing:
        raise ValueError(f"Artifact is missing tensors: {sorted(missing)[:3]}")
    return {"artifact": str(root), "tensor_count": checked,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "dtype_counts": {str(dtype): sum(1 for t in tensors.values() if t.dtype == dtype)
                             for dtype in sorted({t.dtype for t in tensors.values()}, key=str)}}


def run_inference(model_path, artifact, *, prompt, output, steps, height, width):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference qualification requires CUDA")
    transformer = Flux2Transformer2DModel.from_pretrained(
        artifact, local_files_only=True, torch_dtype=torch.bfloat16)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path, transformer=transformer, local_files_only=True,
        torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    result = pipe(prompt=prompt, height=height, width=width,
                  num_inference_steps=steps, output_type="pil")
    if not result.images or result.images[0].size != (width, height):
        raise RuntimeError("Inference returned an invalid image")
    image = result.images[0]
    if image.mode not in ("RGB", "RGBA"):
        raise RuntimeError(f"Inference returned unsupported image mode: {image.mode}")
    image.save(output)
    return {"output": str(output), "size": list(image.size), "steps": steps}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a saved Klein transformer artifact")
    parser.add_argument("--artifact", required=True, type=Path,
                        help="Published checkpoint model/ directory")
    parser.add_argument("--model-path", type=Path,
                        help="Original Base 9B directory for optional GPU inference")
    parser.add_argument("--prompt", default="a small red house in sunlight")
    parser.add_argument("--output", type=Path, default=Path("artifact-verification.png"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args(argv)
    report = {"cpu_load": verify_transformer_artifact(args.artifact)}
    if args.model_path is not None:
        report["gpu_inference"] = run_inference(
            args.model_path, args.artifact, prompt=args.prompt, output=args.output,
            steps=args.steps, height=args.height, width=args.width)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
