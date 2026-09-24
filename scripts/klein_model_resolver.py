"""Offline, fail-closed selection for the standard Klein Diffusers layout.

Only standard safetensors (no variant), built-in classes and a fast tokenizer
are supported. The loader receives a private view containing exactly this
selection. Hard links avoid copying 9B weights; metadata is copied. Source
files must remain immutable for the run. This is not a hostile-file sandbox.
No remote code, Hub lookup, revision inference or format fallback is allowed.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

from scripts.klein_checkpoint import CheckpointValidationError, _read_json, _relative_path


def resolve_model_files(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise CheckpointValidationError("Recovery requires an existing offline model directory")
    index = _read_json(root / "model_index.json")
    expected = {"transformer": ["diffusers", "Flux2Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
                "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]}
    if index.get("_class_name") != "Flux2KleinPipeline":
        raise CheckpointValidationError("Recovery requires Flux2KleinPipeline")
    for key, value in expected.items():
        if index.get(key) != value:
            raise CheckpointValidationError(f"Unsupported model component: {key}")
    if index.get("tokenizer") not in (["transformers", "Qwen2Tokenizer"],
                                       ["transformers", "Qwen2TokenizerFast"]):
        raise CheckpointValidationError("Recovery requires the built-in Qwen2 fast tokenizer")
    extras = [k for k, v in index.items() if not k.startswith("_") and
              isinstance(v, list) and k not in expected and k != "tokenizer"]
    if extras:
        raise CheckpointValidationError(f"Unrecorded pipeline dependencies: {extras}")
    selected = {"model_index.json", "scheduler/scheduler_config.json"}
    for component, stem in (("transformer", "diffusion_pytorch_model"),
                            ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")):
        selected.add(f"{component}/config.json")
        folder = root / component
        single = f"{stem}.safetensors"
        idx = single + ".index.json"
        if (folder / single).exists() == (folder / idx).exists():
            raise CheckpointValidationError(f"Missing or ambiguous weights: {component}")
        weights = {single}
        if (folder / idx).exists():
            document = _read_json(folder / idx)
            mapping = document.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise CheckpointValidationError(f"Invalid weight index: {component}")
            weights = set()
            for name in mapping.values():
                name = _relative_path(name)
                if "/" in name or not name.endswith(".safetensors"):
                    raise CheckpointValidationError("Shard reference must be a local safetensors filename")
                weights.add(name)
            selected.add(f"{component}/{idx}")
        present = {p.name for p in folder.iterdir() if p.suffix in (".bin", ".safetensors")}
        if present != weights:
            raise CheckpointValidationError(f"Ambiguous, missing or unindexed weights: {component}")
        selected.update(f"{component}/{name}" for name in weights)
    if (root / "text_encoder/generation_config.json").exists():
        selected.add("text_encoder/generation_config.json")
    # Explicit fast-tokenizer selection: tokenizer.json wins by contract, not
    # loader heuristics. Slow vocab/merges are never exposed or fingerprinted.
    selected.update(("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"))
    for name in ("special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
        if (root / "tokenizer" / name).exists():
            selected.add("tokenizer/" + name)
    allowed_tokenizer = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                         "added_tokens.json", "chat_template.jinja", "vocab.json", "merges.txt"}
    if any(p.name not in allowed_tokenizer for p in (root / "tokenizer").iterdir()):
        raise CheckpointValidationError("Unsupported tokenizer asset; resolver must explicitly record it")
    for relative in sorted(selected):
        path = root / relative
        if not path.is_file():
            raise CheckpointValidationError(f"Missing selected model file: {relative}")
        if path.suffix == ".json":
            config = _read_json(path)
            if not isinstance(config, dict) or config.get("auto_map") or config.get("quantization_config"):
                raise CheckpointValidationError(f"Invalid or custom-code model metadata: {relative}")
            if relative == "tokenizer/tokenizer_config.json":
                for key in ("tokenizer_file", "vocab_file", "merges_file", "added_tokens_file",
                            "special_tokens_map_file", "chat_template_file"):
                    if config.get(key) is not None:
                        raise CheckpointValidationError(f"External tokenizer dependency override: {key}")
    return tuple(sorted(selected))


@contextmanager
def selected_model_view(root, files):
    """Same-filesystem private hard-link view; removed before training starts.

    HF snapshot symlinks are resolved to their regular blob target. Metadata
    contents are unchanged. Hard-link failure is explicit, never a huge copy.
    """
    root = Path(root).resolve()
    if tuple(files) != resolve_model_files(root):
        raise CheckpointValidationError("Model selection changed before loading")
    with tempfile.TemporaryDirectory(prefix=".klein-model-selection-", dir=root.parent) as temp:
        view = Path(temp)
        for relative in files:
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = (root / relative).resolve(strict=True)
            if source.suffix == ".safetensors":
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
        yield view


def load_selected_pipeline(root, files, dtype):
    from diffusers import (Flux2KleinPipeline, Flux2Transformer2DModel,
                           AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler)
    from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
    with selected_model_view(root, files) as view:
        common = {"local_files_only": True, "use_safetensors": True, "torch_dtype": dtype}
        transformer = Flux2Transformer2DModel.from_pretrained(view / "transformer", **common)
        vae = AutoencoderKLFlux2.from_pretrained(view / "vae", **common)
        text = Qwen3ForCausalLM.from_pretrained(view / "text_encoder", trust_remote_code=False, **common)
        tokenizer = Qwen2TokenizerFast.from_pretrained(view / "tokenizer", local_files_only=True,
                                                     trust_remote_code=False)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(view / "scheduler", local_files_only=True)
        index = _read_json(view / "model_index.json")
        # Base declaration is checked by the recovery trainer before allocation.
        return Flux2KleinPipeline(transformer=transformer, vae=vae, text_encoder=text,
                                 tokenizer=tokenizer, scheduler=scheduler,
                                 is_distilled=index.get("is_distilled", False))
