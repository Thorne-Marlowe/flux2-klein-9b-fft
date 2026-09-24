"""Read-only metadata foundation, deliberately NOT a checkpoint manifest.

This foundation has its own format/version and is bound by checkpoint manifest
v2. The v1 reader remains separate: its configuration cannot express clipping,
checkpointing, attention policy or source identity. No v1 reinterpretation or
migration exists. The recovery runtime fingerprints its resolver and integration
sources in addition to the foundation's minimum source inventory.

Hashing is O(total selected source bytes), with 1 MiB read buffers and no hash
cache. Callers supply the complete, explicit list of files selected by their
offline model resolver; this module neither loads a model nor guesses a variant
or Hub revision. Source symlinks (including HF cache blobs) are followed, but
logical paths must remain relative to the selected root. Content is authoritative.
Inputs must remain immutable while collecting metadata; stat checks detect common
concurrent edits but are not a filesystem snapshot or a security boundary.

No trainer imports, RNG seeding, CUDA initialization, optimizer mutation, dataset
iteration, checkpoint writes, publication or retention take place here.
"""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

from scripts.klein_checkpoint import (
    CheckpointCompatibilityError, CheckpointValidationError, _freeze, _same,
    _group_names, _optimizer_kind, _read_json, _relative_path,
)


FOUNDATION_FORMAT = "klein-recovery-metadata"
FOUNDATION_VERSION = 1
REQUIRED_CHECKPOINT_SCHEMA = 2
SOURCE_FILES = ("scripts/train_klein_standalone.py", "scripts/klein_checkpoint.py",
                "scripts/klein_recovery_metadata.py")


def canonical_sha256(document):
    """Canonical UTF-8 JSON: sorted keys, compact separators, preserved Unicode."""
    try:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointValidationError(f"Noncanonical metadata: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _file_record(root, relative):
    relative = _relative_path(relative)
    path = Path(root) / relative
    if not path.is_file():
        raise CheckpointValidationError(f"Missing selected source file: {relative}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino")
    if any(getattr(before, k) != getattr(after, k) for k in fields):
        raise CheckpointValidationError(f"Source changed while hashing: {relative}")
    return {"path": relative, "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def _ordered_files(root, files):
    files = list(files)
    if not files or any(not isinstance(p, str) for p in files):
        raise CheckpointValidationError("Supply nonempty relative source-file paths")
    if len(set(p.casefold() for p in files)) != len(files):
        raise CheckpointValidationError("Duplicate/case-colliding source-file selection")
    return [_file_record(root, name) for name in sorted(files)]


def fingerprint_model(root, resolved_files):
    """Hash explicitly resolved component files; unknown repo/revision stay null.

    Selection must include pipeline metadata, component configs, tokenizer assets
    and weights for transformer/VAE/text encoder. Index references must be present.
    These are minimum selection checks, NOT proof of a complete model selection.
    An omitted index or extra tokenizer asset cannot be discovered from this list.
    A future offline resolver must walk the actual pipeline component references,
    determine the exact loading variant/format without ambiguous alternatives,
    include all configs, selected weight indexes/shards, tokenizer vocabulary,
    merges, special-token/chat-template and other referenced assets, and reject
    missing references. It must supply the exact same immutable selection to the
    loader and this producer, with no network fallback or unrecorded file reads.
    Resolved blob contents, not symlink locations or guessed revisions, identify
    the selection. Resolver implementation is deferred. This fingerprint proves
    neither selection completeness, architecture nor weight provenance.
    """
    records = _ordered_files(root, resolved_files)
    selected = {r["path"] for r in records}
    required = {"model_index.json", "transformer/config.json", "vae/config.json", "text_encoder/config.json"}
    if not required <= selected:
        raise CheckpointValidationError(f"Model selection lacks required metadata: {sorted(required - selected)}")
    for component in ("transformer", "vae", "text_encoder"):
        if not any(p.startswith(component + "/") and p.endswith((".safetensors", ".bin")) for p in selected):
            raise CheckpointValidationError(f"Model selection lacks {component} weights")
    if not any(p.startswith("tokenizer/") for p in selected):
        raise CheckpointValidationError("Model selection lacks tokenizer assets")
    for name in selected:
        if name.endswith(".index.json"):
            index = _read_json(Path(root) / name)
            mapping = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise CheckpointValidationError(f"Invalid model shard index: {name}")
            for shard in mapping.values():
                shard = _relative_path(shard)
                if "/" in shard or (Path(name).parent / shard).as_posix() not in selected:
                    raise CheckpointValidationError(f"Unselected/unsafe shard in {name}: {shard}")
    document = {"repo_id": None, "revision": None, "component_files": records}
    return {"sha256": canonical_sha256(document), "document": document}


def fingerprint_dataset(dataset):
    """Use dataset.samples order exactly; never decode or sort the sample list."""
    root = Path(dataset.data_dir).absolute()
    records = []
    for image, caption in dataset.samples:
        pair = {}
        for key, path in (("image", image), ("caption", caption)):
            try:
                relative = Path(path).absolute().relative_to(root).as_posix()
            except ValueError as error:
                raise CheckpointValidationError(f"Dataset {key} is outside data_dir") from error
            pair[key] = _file_record(root, relative)
        records.append(pair)
    if not records:
        raise CheckpointValidationError("Recovery dataset is empty")
    return {"sha256": canonical_sha256(records), "document": records}


def fingerprint_preprocessing(dataset, *, bucket_sizes):
    """Describe the current standalone __getitem__, not an intended resolution.

    bucket_sizes must be the trainer's actual ordered BUCKET_SIZES. target_size
    has no pixel effect in bucket mode and is recorded as inactive (null).
    """
    if dataset.use_cached:
        raise CheckpointValidationError("Recovery metadata does not support cached latents")
    buckets = [list(b) for b in bucket_sizes]
    if (not buckets or any(len(b) != 2 or any(type(v) is not int or v <= 0 for v in b) for b in buckets)):
        raise CheckpointValidationError("Supply actual ordered positive bucket dimensions")
    if type(dataset.fixed_size) is not bool:
        raise CheckpointValidationError("fixed_size must be an explicit boolean")
    document = {
        "algorithm": "standalone_image_text_v1", "image_mode": "RGB",
        "caption": {"encoding": "utf-8", "strip": True},
        "resolution": {"mode": "fixed_square" if dataset.fixed_size else "aspect_bucket",
                       "target_size": dataset.target_size if dataset.fixed_size else None,
                       "ordered_buckets": None if dataset.fixed_size else buckets,
                       "selection": "first_minimum_absolute_aspect_difference"},
        "resize": {"scale": "max(target_width/width,target_height/height)",
                   "rounding": "int_truncation", "filter": "PIL.LANCZOS"},
        "crop": "center_floor_offsets", "tensor": "torchvision.ToTensor",
        "normalize": {"mean": [0.5], "std": [0.5]},
        "random_augmentation": False,
    }
    if dataset.fixed_size and (type(dataset.target_size) is not int or dataset.target_size <= 0 or dataset.target_size % 16):
        raise CheckpointValidationError("Fixed target_size must be positive and divisible by 16")
    return {"sha256": canonical_sha256(document), "document": document}


def fingerprint_source_code(root, files=SOURCE_FILES):
    """Hash actual code, including dirty/untracked files; Git HEAD is not identity."""
    document = {"files": _ordered_files(root, files)}
    return {"sha256": canonical_sha256(document), "document": document}


def collect_backend_settings(model):
    """Read effective Torch flags and processor selection without initializing CUDA."""
    import torch
    processors = getattr(model, "attn_processors", None)
    if not isinstance(processors, dict) or not processors:
        raise CheckpointValidationError("Model must expose its actual attn_processors for recovery metadata")
    attention = {}
    for name, processor in sorted(processors.items()):
        if not hasattr(processor, "_attention_backend"):
            raise CheckpointValidationError(f"Cannot determine effective attention backend at {name}")
        backend = processor._attention_backend
        if backend is not None and not isinstance(backend, str):
            backend = getattr(backend, "value", backend)
        if backend not in (None, "native"):
            raise CheckpointValidationError(f"Unsupported explicit attention backend at {name}: {backend!r}")
        attention[name] = {"class": type(processor).__module__ + "." + type(processor).__qualname__,
                           "backend": backend}
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sdpa": {name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
                 for name in ("flash", "mem_efficient", "math", "cudnn")},
        "math_sdpa_reduced_precision": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "attention_processors": attention,
    }


def _json_options(optimizer):
    groups = []
    for group in optimizer.param_groups:
        options = {k: v for k, v in group.items() if k != "params"}
        # Native scalar options only; never coerce tensor or callable options.
        canonical_sha256(options)
        groups.append(json.loads(json.dumps(options, allow_nan=False)))
    return groups


def build_recovery_configuration(*, args, dataset, dataloader, model, optimizer,
                                 scheduler, accelerator, resolved_model_files,
                                 bucket_sizes, ema=None, source_root=None):
    """Capture a fresh-run foundation after objects exist, before prepare/steps.

    Isolated opt-in API; no CLI flag or trainer call is introduced. The supplied
    Accelerator defines the selected runtime, while model weights may still be
    on CPU before prepare. Missing dependency versions are reported as null; this
    metadata does not certify an executable/qualified environment.
    """
    import torch
    if args.smoke_test:
        raise CheckpointValidationError("Recovery metadata is separate from --smoke_test")
    if getattr(args, "resume_from", None):
        raise CheckpointValidationError("Build fresh recovery metadata before restoring; resume integration is deferred")
    kind = _optimizer_kind(optimizer)
    if args.optimizer != kind:
        raise CheckpointValidationError("Requested optimizer differs from constructed native optimizer")
    if optimizer.state:
        raise CheckpointValidationError("Build initial recovery configuration before any optimizer step")
    if type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR or scheduler.optimizer is not optimizer:
        raise CheckpointValidationError("Recovery requires the native CosineAnnealingLR bound to this optimizer")
    if dataloader.dataset is not dataset:
        raise CheckpointValidationError("DataLoader must refer to the fingerprinted dataset")
    if getattr(args, "grad_accum", accelerator.gradient_accumulation_steps) != accelerator.gradient_accumulation_steps:
        raise CheckpointValidationError("CLI accumulation differs from constructed Accelerator")
    if getattr(args, "use_cached_latents", dataset.use_cached) != dataset.use_cached:
        raise CheckpointValidationError("CLI cache setting differs from constructed dataset")
    for setting, actual in (("batch_size", dataloader.batch_size), ("num_workers", dataloader.num_workers)):
        if getattr(args, setting) != actual:
            raise CheckpointValidationError(f"CLI {setting} differs from constructed DataLoader")
    active_checkpointing = getattr(model, "is_gradient_checkpointing", None)
    if type(active_checkpointing) is not bool or active_checkpointing != args.gradient_checkpointing:
        raise CheckpointValidationError("Gradient-checkpointing request differs from actual model state")
    if args.use_ema != (ema is not None) or (ema is not None and ema.decay != args.ema_decay):
        raise CheckpointValidationError("EMA request differs from constructed EMA")
    if any(p.dtype != torch.bfloat16 or not p.requires_grad for p in model.parameters()):
        raise CheckpointValidationError("Recovery requires trainable BF16 transformer parameters")
    groups = _group_names(model, optimizer)
    options = _json_options(optimizer)
    for group in options:
        if group.get("initial_lr") != args.lr or group["lr"] != args.lr:
            raise CheckpointValidationError("Fresh optimizer/scheduler LR differs from the trainer's manual-warmup base LR")
    config = {
        "world_size": accelerator.num_processes, "device": str(accelerator.device),
        "distributed_type": getattr(accelerator.distributed_type, "name", str(accelerator.distributed_type)),
        "precision": accelerator.mixed_precision, "batch_size": dataloader.batch_size,
        "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
        "num_workers": dataloader.num_workers, "drop_last": dataloader.drop_last,
        "dataset_size": len(dataset),
        "use_cached_latents": dataset.use_cached, "sampling_enabled": bool(args.sample_prompts),
        "tracking_enabled": bool(getattr(args, "wandb", False) or getattr(args, "log_dir", None)),
        "optimizer": kind, "optimizer_options": options, "parameter_groups": groups,
        "total_optimizer_steps": args.steps, "warmup_steps": min(args.warmup_steps, args.steps // 10),
        "schedule_policy": "standalone_warmup_cosine_v1", "seed": args.seed,
        "seed_policy": "klein_recovery_v1_domain_separated", "ema_enabled": ema is not None,
        "ema_decay": ema.decay if ema is not None else None,
        "max_grad_norm": args.max_grad_norm, "gradient_checkpointing": active_checkpointing,
        "scaler_enabled": getattr(accelerator, "scaler", None) is not None,
    }
    validate_recovery_settings(config)
    if (scheduler.last_epoch != 0 or scheduler.T_max != args.steps - config["warmup_steps"]
            or scheduler.eta_min != args.lr * 0.1):
        raise CheckpointValidationError("Constructed scheduler differs from resolved standalone schedule")
    versions = {}
    for package in ("torch", "torchvision", "transformers", "diffusers", "accelerate", "safetensors", "numpy", "pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["torch"] = torch.__version__
    environment = {"python": platform.python_version(), "platform": sys.platform,
                   "machine": platform.machine(), "dependencies": versions, "cuda_runtime": torch.version.cuda,
                   "backend_settings": collect_backend_settings(model)}
    sources = fingerprint_source_code(source_root or Path(__file__).resolve().parents[1])
    fingerprints = {"model": fingerprint_model(args.model_path, resolved_model_files),
                    "dataset": fingerprint_dataset(dataset),
                    "preprocessing": fingerprint_preprocessing(dataset, bucket_sizes=bucket_sizes),
                    "source_code": sources}
    document = {"format": FOUNDATION_FORMAT, "foundation_version": FOUNDATION_VERSION,
                "required_checkpoint_schema": REQUIRED_CHECKPOINT_SCHEMA, "qualification": "unqualified",
                "configuration": config, "environment": environment, "fingerprints": fingerprints}
    return {"sha256": canonical_sha256(document), "document": document}


def validate_recovery_settings(config):
    """Fail closed on the initial candidate scope, without changing runtime flags."""
    expected = {"world_size": 1, "precision": "bf16", "distributed_type": "NO", "batch_size": 1,
                "gradient_accumulation_steps": 1, "num_workers": 0, "drop_last": True,
                "use_cached_latents": False, "sampling_enabled": False, "scaler_enabled": False,
                "tracking_enabled": False}
    for key, value in expected.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise CheckpointValidationError(f"Unsupported recovery {key}={config.get(key)!r}; required {value!r}")
    if not isinstance(config.get("device"), str) or (config["device"] != "cuda" and not (
            config["device"].startswith("cuda:") and config["device"][5:].isdigit())):
        raise CheckpointValidationError("Recovery requires a selected CUDA device; CPU codecs do not qualify training")
    for key in ("total_optimizer_steps", "warmup_steps", "seed", "dataset_size"):
        minimum = 1 if key in ("total_optimizer_steps", "dataset_size") else 0
        if type(config[key]) is not int or config[key] < minimum:
            raise CheckpointValidationError(f"Recovery {key} must be an integer >= {minimum}")
    if config["seed"] >= 2**64 or config["warmup_steps"] > config["total_optimizer_steps"] // 10:
        raise CheckpointValidationError("Recovery seed/warmup exceeds contract limits")
    norm = config["max_grad_norm"]
    if type(norm) not in (float, int) or not math.isfinite(norm) or norm <= 0:
        raise CheckpointValidationError("Recovery max_grad_norm must be finite and positive")
    if config["optimizer"] not in ("adamw", "adafactor"):
        raise CheckpointValidationError("Recovery supports native AdamW or Adafactor, not AdamW8bit")
    if type(config.get("gradient_checkpointing")) is not bool or type(config.get("ema_enabled")) is not bool:
        raise CheckpointValidationError("Checkpointing and EMA policies must be explicit booleans")
    if config["schedule_policy"] != "standalone_warmup_cosine_v1" or config["seed_policy"] != "klein_recovery_v1_domain_separated":
        raise CheckpointValidationError("Unsupported recovery schedule/seed policy")
    if config["ema_enabled"] and (type(config["ema_decay"]) not in (int, float) or not 0 <= config["ema_decay"] < 1):
        raise CheckpointValidationError("Recovery EMA decay must be finite and in [0,1)")
    if not config["optimizer_options"]:
        raise CheckpointValidationError("Recovery optimizer has no effective parameter groups")
    for group in config["optimizer_options"]:
        for key in ("lr", "initial_lr", "weight_decay"):
            value = group.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
                raise CheckpointValidationError(f"Invalid effective optimizer {key}: {value!r}")
        if config["optimizer"] == "adafactor":
            if any(group.get(k) is not False for k in ("relative_step", "scale_parameter", "warmup_init")):
                raise CheckpointValidationError("Recovery requires standalone fixed-LR Adafactor options")
        elif any(group.get(k) not in (False, None) for k in ("fused", "capturable", "differentiable")):
            raise CheckpointValidationError("Recovery does not support fused/capturable/differentiable AdamW")


def _invalid(context):
    raise CheckpointValidationError(f"Invalid recovery metadata: {context}")


def _keys(value, keys, context):
    if type(value) is not dict or set(value) != set(keys):
        _invalid(f"{context} requires exactly {sorted(keys)}")


def _text(value, context, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{context} must be a nonempty string")


def _digest(value, context):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _invalid(f"{context} must be a lowercase SHA-256 digest")


def _json_tree(value, active=None, depth=0):
    """Reject non-JSON types, cycles and excessive nesting before field access."""
    if depth > 100:
        _invalid("excessive nesting")
    if value is None or isinstance(value, str) or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _invalid("nonfinite number")
        return
    if type(value) not in (dict, list):
        _invalid(f"unsupported JSON value type {type(value).__name__}")
    active = set() if active is None else active
    if id(value) in active:
        _invalid("cyclic document")
    active.add(id(value))
    if type(value) is dict and any(type(k) is not str for k in value):
        _invalid("object keys must be strings")
    for item in value.values() if type(value) is dict else value:
        _json_tree(item, active, depth + 1)
    active.remove(id(value))


def _number(value, context, minimum=0):
    if type(value) not in (int, float) or value < minimum:
        _invalid(f"{context} must be a number >= {minimum}")


def _bool(value, context, nullable=False):
    if type(value) is not bool and not (nullable and value is None):
        _invalid(f"{context} must be boolean")


def _records(records, context):
    if type(records) is not list or not records:
        _invalid(f"{context} must be a nonempty file inventory")
    paths = []
    for record in records:
        _keys(record, {"path", "size_bytes", "sha256"}, context)
        _relative_path(record["path"])
        if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            _invalid(f"{context} size_bytes must be a nonnegative integer")
        _digest(record["sha256"], context)
        paths.append(record["path"])
    return paths


def _file_inventory(records, context):
    paths = _records(records, context)
    if paths != sorted(paths) or len({p.casefold() for p in paths}) != len(paths):
        _invalid(f"{context} must be sorted and free of duplicate paths")
    return paths


def _configuration_structure(config):
    _keys(config, {"world_size", "device", "distributed_type", "precision", "batch_size",
                   "gradient_accumulation_steps", "num_workers", "drop_last", "dataset_size",
                   "use_cached_latents", "sampling_enabled", "tracking_enabled", "optimizer",
                   "optimizer_options", "parameter_groups", "total_optimizer_steps", "warmup_steps",
                   "schedule_policy", "seed", "seed_policy", "ema_enabled", "ema_decay",
                   "max_grad_norm", "gradient_checkpointing", "scaler_enabled"}, "configuration")
    for key in ("device", "distributed_type", "precision", "optimizer", "schedule_policy", "seed_policy"):
        _text(config[key], key)
    groups, options = config["parameter_groups"], config["optimizer_options"]
    if type(groups) is not list or not groups or type(options) is not list or len(options) != len(groups):
        _invalid("optimizer groups/options must be aligned nonempty lists")
    names = []
    for group in groups:
        if type(group) is not list or not group:
            _invalid("parameter group must be a nonempty list")
        for name in group:
            _text(name, "parameter name")
            names.append(name)
    if len(set(names)) != len(names):
        _invalid("duplicate optimizer parameter name")
    for group in options:
        if config["optimizer"] == "adamw":
            fields = {"lr", "initial_lr", "weight_decay", "betas", "eps", "amsgrad", "maximize",
                      "foreach", "capturable", "differentiable", "fused"}
            if type(group) is dict and "decoupled_weight_decay" in group:
                fields.add("decoupled_weight_decay")
            _keys(group, fields, "AdamW options")
            if type(group["betas"]) is not list or len(group["betas"]) != 2:
                _invalid("AdamW betas must have two values")
            for beta in group["betas"]:
                _number(beta, "beta")
                if beta >= 1:
                    _invalid("beta must be below one")
            _number(group["eps"], "eps")
            for key in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"):
                if key in group:
                    _bool(group[key], key)
            for key in ("foreach", "fused"):
                _bool(group[key], key, nullable=True)
        elif config["optimizer"] == "adafactor":
            _keys(group, {"lr", "initial_lr", "weight_decay", "eps", "clip_threshold", "decay_rate",
                          "beta1", "scale_parameter", "relative_step", "warmup_init"}, "Adafactor options")
            if type(group["eps"]) is not list or len(group["eps"]) != 2:
                _invalid("Adafactor eps must have two values")
            for value in group["eps"]:
                _number(value, "eps")
            _number(group["clip_threshold"], "clip_threshold")
            if group["clip_threshold"] == 0 or type(group["decay_rate"]) not in (int, float) or group["decay_rate"] >= 0:
                _invalid("invalid Adafactor clip/decay settings")
            if group["beta1"] is not None:
                _number(group["beta1"], "beta1")
                if group["beta1"] >= 1:
                    _invalid("beta1 must be below one")
        else:
            _invalid("unsupported optimizer")
    if not config["ema_enabled"] and config["ema_decay"] is not None:
        _invalid("disabled EMA must have null decay")
    validate_recovery_settings(config)


def _environment_structure(environment):
    _keys(environment, {"python", "platform", "machine", "dependencies", "cuda_runtime", "backend_settings"}, "environment")
    for key in ("python", "platform", "machine", "cuda_runtime"):
        _text(environment[key], key, nullable=key == "cuda_runtime")
    _keys(environment["dependencies"], {"torch", "torchvision", "transformers", "diffusers", "accelerate",
                                        "safetensors", "numpy", "pillow"}, "dependencies")
    for key, value in environment["dependencies"].items():
        _text(value, key, nullable=True)
    backend = environment["backend_settings"]
    flags = {"deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic",
             "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "fp16_reduced_precision_reduction",
             "bf16_reduced_precision_reduction", "math_sdpa_reduced_precision"}
    _keys(backend, flags | {"float32_matmul_precision", "sdpa", "CUBLAS_WORKSPACE_CONFIG", "attention_processors"}, "backend_settings")
    for key in flags:
        _bool(backend[key], key)
    if backend["float32_matmul_precision"] not in ("highest", "high", "medium"):
        _invalid("unsupported matmul precision")
    # Preserve an explicitly empty environment variable as recorded by the
    # producer; it is distinct from an absent variable (None).
    workspace = backend["CUBLAS_WORKSPACE_CONFIG"]
    if workspace is not None and not isinstance(workspace, str):
        _invalid("CUBLAS_WORKSPACE_CONFIG must be a string or null")
    _keys(backend["sdpa"], {"flash", "mem_efficient", "math", "cudnn"}, "sdpa")
    for key, value in backend["sdpa"].items():
        _bool(value, key)
    processors = backend["attention_processors"]
    if type(processors) is not dict or not processors:
        _invalid("attention_processors must be a nonempty object")
    for name, processor in processors.items():
        _text(name, "processor name")
        _keys(processor, {"class", "backend"}, "attention processor")
        _text(processor["class"], "processor class")
        if processor["backend"] not in (None, "native"):
            _invalid("unsupported attention backend")


def _preprocessing_structure(value):
    _keys(value, {"algorithm", "image_mode", "caption", "resolution", "resize", "crop", "tensor",
                  "normalize", "random_augmentation"}, "preprocessing")
    constants = {"algorithm": "standalone_image_text_v1", "image_mode": "RGB",
                 "caption": {"encoding": "utf-8", "strip": True}, "crop": "center_floor_offsets",
                 "tensor": "torchvision.ToTensor", "normalize": {"mean": [0.5], "std": [0.5]},
                 "resize": {"scale": "max(target_width/width,target_height/height)",
                            "rounding": "int_truncation", "filter": "PIL.LANCZOS"}, "random_augmentation": False}
    for key, expected in constants.items():
        if not _same(_freeze(expected), value[key]):
            _invalid(f"unsupported preprocessing {key}")
    resolution = value["resolution"]
    _keys(resolution, {"mode", "target_size", "ordered_buckets", "selection"}, "resolution")
    if resolution["selection"] != "first_minimum_absolute_aspect_difference":
        _invalid("unsupported bucket selection")
    if resolution["mode"] == "fixed_square":
        target = resolution["target_size"]
        if type(target) is not int or target <= 0 or target % 16 or resolution["ordered_buckets"] is not None:
            _invalid("invalid fixed resolution")
    elif resolution["mode"] == "aspect_bucket":
        buckets = resolution["ordered_buckets"]
        if resolution["target_size"] is not None or type(buckets) is not list or not buckets:
            _invalid("invalid bucket resolution")
        for bucket in buckets:
            if type(bucket) is not list or len(bucket) != 2 or any(type(x) is not int or x <= 0 for x in bucket):
                _invalid("bucket dimensions must be positive integers")
    else:
        _invalid("unsupported resolution mode")


def _validate_foundation(value):
    _json_tree(value)
    _keys(value, {"sha256", "document"}, "foundation envelope")
    _digest(value["sha256"], "foundation digest")
    document = value["document"]
    _keys(document, {"format", "foundation_version", "required_checkpoint_schema", "qualification",
                     "configuration", "environment", "fingerprints"}, "foundation document")
    if (document["format"] != FOUNDATION_FORMAT or type(document["foundation_version"]) is not int
            or document["foundation_version"] != FOUNDATION_VERSION
            or type(document["required_checkpoint_schema"]) is not int
            or document["required_checkpoint_schema"] != REQUIRED_CHECKPOINT_SCHEMA
            or document["qualification"] != "unqualified"):
        _invalid("unsupported foundation format/version/qualification")
    _configuration_structure(document["configuration"])
    _environment_structure(document["environment"])
    fingerprints = document["fingerprints"]
    _keys(fingerprints, {"model", "dataset", "preprocessing", "source_code"}, "fingerprints")
    for name, fingerprint in fingerprints.items():
        _keys(fingerprint, {"sha256", "document"}, f"{name} fingerprint")
        _digest(fingerprint["sha256"], name)
        content = fingerprint["document"]
        if name == "model":
            _keys(content, {"repo_id", "revision", "component_files"}, "model fingerprint")
            _text(content["repo_id"], "repo_id", nullable=True)
            _text(content["revision"], "revision", nullable=True)
            paths = _file_inventory(content["component_files"], "component_files")
            required = {"model_index.json", "transformer/config.json", "vae/config.json", "text_encoder/config.json"}
            if not required <= set(paths) or not any(p.startswith("tokenizer/") for p in paths):
                _invalid("model selection lacks required metadata/tokenizer")
            for component in ("transformer", "vae", "text_encoder"):
                if not any(p.startswith(component + "/") and p.endswith((".safetensors", ".bin")) for p in paths):
                    _invalid(f"model selection lacks {component} weights")
        elif name == "source_code":
            _keys(content, {"files"}, "source fingerprint")
            if not set(SOURCE_FILES) <= set(_file_inventory(content["files"], "source files")):
                _invalid("foundation source identity omits required modules")
        elif name == "dataset":
            if type(content) is not list or len(content) != document["configuration"]["dataset_size"]:
                _invalid("fingerprint sample count differs from configuration")
            for pair in content:
                _keys(pair, {"image", "caption"}, "dataset pair")
                _records([pair["image"], pair["caption"]], "dataset files")
        else:
            _preprocessing_structure(content)
        if canonical_sha256(content) != fingerprint["sha256"]:
            _invalid(f"{name} fingerprint checksum mismatch")
    if canonical_sha256(document) != value["sha256"]:
        _invalid("foundation checksum mismatch")


def validate_metadata_compatibility(saved, current):
    """Validate both complete foundations before strict comparison; no defaults."""
    for value in (saved, current):
        try:
            _validate_foundation(value)
        except CheckpointValidationError:
            raise
        except (KeyError, TypeError, AttributeError, ValueError, OverflowError, RecursionError) as error:
            # Defensive boundary for malformed Python inputs as well as decoded
            # JSON; explicit field validation above supplies normal diagnostics.
            raise CheckpointValidationError(f"Malformed recovery metadata: {error}") from error
    # TorchVersion is a str subclass emitted by the existing producer. Its JSON
    # representation must compare equal after a JSON round trip. Comparing exact
    # canonical JSON retains distinctions such as false/0 and 1/1.0; no defaults
    # or numeric coercions are applied, and input documents are never mutated.
    if json.dumps(saved["document"], sort_keys=True, ensure_ascii=False, allow_nan=False) != json.dumps(
            current["document"], sort_keys=True, ensure_ascii=False, allow_nan=False):
        raise CheckpointCompatibilityError("Recovery metadata differs; inspect files, effective options, source or environment")
