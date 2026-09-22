"""Recovery contract, state codecs and transactional publication.

The opt-in trainer integration lives in klein_recovery_training.py. Manifest v2
adds the complete metadata foundation and cross-validates its codec projection;
v1 remains readable but is not accepted by the integrated resume path. Both
manifest versions explicitly use the unchanged version-1 state payload codecs.
No v1 settings are inferred or migrated. All checkpoints remain unqualified.

Version 1 root (the directory name never supplies training state)::

    manifest.json                 # writer writes this last, before rename
    model/config.json
    model/diffusion_pytorch_model.safetensors
    optimizer.pt, scheduler.pt, trainer.json, rng.pt, data_order.pt
    ema/weights.safetensors       # present iff EMA is enabled

Model/EMA weights may instead be numbered ``*-00001-of-00002.safetensors``
shards plus ``*.safetensors.index.json``. Index weight_map keys must exactly
match the declared model tensors (EMA: parameters only). Checkpoints are
immutable; publication uses a same-filesystem .incomplete-* directory
and atomic rename to a new checkpoint root. Inference exports are separate.

validate_checkpoint checks schema, supported candidate configuration and every
payload's bytes without importing torch or deserializing optimizer/RNG state.
It does NOT prove that opaque payloads contain restorable tensors or that the
manifest identifies authentic model weights. Future restoration must validate
their contents and cross-check their counters against authoritative progress.

Compatibility is separate: validate_configuration_compatibility runs before
allocation; validate_object_compatibility consumes summaries of constructed
objects, not tensors. Codec restoration does not establish qualification.
All v1 checkpoints must say qualification='unqualified'. Initial candidate
scope: one CUDA GPU, BF16, batch/accumulation 1, workers 0, uncached data,
drop_last=True, no sampling, AdamW or Adafactor, optional EMA. Broader settings
are rejected, not implicitly promised recovery support.

Progress is authoritative: global_step is attempted_optimizer_steps, matching
the existing trainer even when Accelerate skips an update. The historical
configuration name total_optimizer_steps means args.steps: the ATTEMPT horizon,
not a promise of that many completed updates. completed + skipped = attempted
= loop_iterations. An exception/partially applied update is not a checkpoint
boundary. EMA advances on every acknowledged attempt, including skips, as in
the existing trainer. No partial accumulation is represented in v1.
Completed means a returned, non-skipped update with supported optimizer-state
step evidence, not necessarily a stored BF16 weight change. An unexplained
no-op or insufficient evidence must fail recovery checkpoint eligibility;
only an explicitly reported Accelerate skip increments skipped_optimizer_steps.
Per-parameter counters alone cannot establish completed-update evidence: they
describe cumulative parameter history, not whether a particular attempt ran.
Future trainer integration must capture authoritative step-return/skip outcomes
at each attempt, together with supported before/after optimizer evidence. Codec
counter checks establish consistency only; they cannot certify that history.
Backend settings and initial optimizer options are explicit;
the changing LR and optimizer/scheduler counters remain in their state payloads.
Object class identifiers use public names (e.g. torch.optim.AdamW), tensor dtype
names omit 'torch.', and parameter_groups preserve both group and member order.

Alignment specification (implemented by the opt-in recovery trainer)
------------------------------------------------------------------
Seeding: on a fresh ordinary recovery-enabled run, before constructing models
or iterators, seed Python with S, NumPy's legacy global RNG with S % 2**32, and
Torch CPU/all CUDA generators with S (0 <= S < 2**64). Keep this separate from
the existing smoke branch. Use independent CPU torch.Generator instances for
data permutations and DataLoader base seeds. Their initial seeds are the first
8 bytes, little-endian unsigned, of SHA-256 of UTF-8 strings
'klein-recovery-v1:data-order:S' and 'klein-recovery-v1:loader:S', with S replaced
by its decimal representation. On resume, restore saved states, not initial
seeds; restore global RNG last, after construction/iterator setup. Existing
ordinary training currently does NOT perform this seeding. These are new-run
recovery semantics, not a claim to reproduce an earlier unseeded run.

Data order: generate one CPU int64 randperm(dataset_size) per epoch. Store that
permutation and the independent order-generator state AFTER its generation.
The trainer acknowledges a batch only after step/skip, LR handling, zero_grad
and EMA finish; prefetched/yielded sampler indices are never authoritative.
At acknowledgement increment attempts and cursor even for a skip. At the last
batch normalize to (epoch + 1, 0), retain the exhausted epoch's permutation and
post-generation RNG state, and mark permutation_epoch = epoch - 1. At an
initial zero-step boundary use permutation_epoch = -1 and an empty permutation.
At a mid-epoch boundary permutation_epoch = epoch; resume uses its suffix,
without replaying transforms for consumed batches. At epoch boundaries generate
exactly one new permutation. Store the loader generator state too; rebuilding
a mid-epoch iterator must not permanently consume an extra base-seed draw.
Workers > 0 remain unsupported. Accelerate iterator/epoch bookkeeping must be
aligned explicitly and qualified with a real prepared loader before integration.

LR: N = total_optimizer_steps, W = min(requested_warmup, N // 10); the manifest
records resolved W. Attempt a (zero-based) uses the previously stored optimizer
LR. AFTER that attempt, if a < W assign base_lr * (a + 1) / W, even on a skip;
otherwise request one wrapped cosine step (Accelerate suppresses it on skip).
Thus the first update uses full base_lr, not base_lr/W. Preserve this ordering.
Cosine uses T_max=N-W, eta_min=base_lr*0.1. During manual warmup its _last_lr can
differ from optimizer.param_groups[*]['lr']; never substitute one for the other.
Construct scheduler before restoring optimizer groups, then load scheduler
state without stepping/replaying warmup. Compare actual group LRs afterwards.

Payload CONTENT specification (validate_checkpoint remains an integrity check;
read_state_payload performs the separate semantic validation described below):
Every state payload has an envelope {schema_version: 1, checkpoint_id, state};
checkpoint_id must equal the manifest; schema_version identifies the state codec
(1 for both v1 and v2 manifests). Unknown envelope/state keys fail,
except native optimizer/scheduler state keys governed by the pinned dependency.

* trainer.json state: progress (exact manifest copy), global_step (= attempts),
  ema_updates (= attempts if enabled, else 0), cosine_updates, group_lrs (ordered
  finite nonnegative numbers), accelerator_step (nonnegative native accumulation
  counter), loss_accumulator (finite), log_steps (nonnegative). Capture logging
  state at the agreed post-save-log-reset boundary. No pending gradient or
  GradScaler state is supported. With A attempts, C completions and W warmup,
  max(0,C-min(A,W)) <= cosine_updates <= min(C,max(0,A-W)); individual skip
  placement determines the exact value, so it cannot be inferred from C alone.
* rng.pt state: python (random.getstate tuple), numpy (algorithm string, keys
  represented as CPU integer tensor, position, has_gauss, cached_gaussian),
  torch_cpu (CPU uint8 RNG tensor), torch_cuda (one CPU uint8 RNG tensor per
  logical CUDA device; exactly one in v1). Validate algorithm, shape, range and
  device count; restoration failure is fatal, never log-and-continue. Avoid
  pickled NumPy ndarray objects: use tensors/primitives for restricted loading.
* data_order.pt state: epoch, next_batch_index (both match progress),
  permutation_epoch, permutation (CPU int64, unique range [0,dataset_size),
  length dataset_size except initial empty state), order_generator_state and
  loader_generator_state (CPU uint8), dataset_fingerprint (manifest match).
  The acknowledged cursor, not an iterator's prefetched cursor, selects suffix.
* optimizer.pt state: state_dict (native state + param_groups), parameter_groups
  (ordered names matching objects.parameter_groups), tensor_metadata (one record
  per nested tensor: parameter name or group index, key path, shape, dtype), and
  step_counters (per initialized parameter, native scalar type/value). Map native
  IDs by ordered groups, not arbitrary dictionary ordering. Require exact group
  coverage, valid state shapes and supported optimizer keys; uninitialized state
  is explicit. Counters cannot exceed completions, but need not all equal C
  because parameters can have absent gradients. Native state and inventories
  must agree. Preserve effective group options, including ordinary Adafactor's
  actual default weight_decay=0, rather than copying the unused CLI setting.
* scheduler.pt state: state_dict (complete native CosineAnnealingLR state),
  cosine_updates (trainer match), group_lrs (trainer and optimizer group match).
  Validate T_max, eta_min, base_lrs against resolved configuration; with the
  pinned Torch initialization, last_epoch = cosine_updates and _step_count =
  cosine_updates + 1. Preserve _last_lr separately, including its warmup staleness.

Adafactor restoration must occur AFTER accelerator.prepare: native optimizer
load_state_dict casts floating state to BF16 parameter dtype. Retain pristine
CPU payload tensors, use native loading for group/ID mapping, then replace each
mapped state tensor from the pristine source using device-only conversion,
preserving saved dtype. Never upcast an already rounded BF16 moment. Validate
every state tensor against its inventory (including FP32 factored/unfactored
moments and RMS) before continuing. No second wrapper load may recast it. This
requires CPU round-trip and prepared-Accelerate tests before claiming recovery.

Fingerprint producer contract (implemented in klein_recovery_metadata.py): SHA-256 of UTF-8 JSON
with sorted object keys, compact separators, ensure_ascii=False, no NaN/Infinity.
model_fingerprint covers an object with repo_id/revision (nullable) and a
path-sorted component_files list of {path, size_bytes, sha256} for the original
pipeline's configs, tokenizer and transformer/frozen-component source weights.
dataset_fingerprint covers the ordered list of {image, caption} records; each
value is {path, size_bytes, sha256}. Paths in these records are relative POSIX
paths, so relocation alone does not change identity. preprocessing_fingerprint
covers the resolved policy object (algorithm version, resize/bucket/crop,
normalization and resolution settings). External source hashes are supplied
again during pre-allocation compatibility checks. The separate metadata builder
does not change this v1 manifest: its additional execution/source fields require
explicit manifest schema 2. V2 binds these fields explicitly; the v1 reader remains separate.

Integrity is a point-in-time check, not a lock or power-loss guarantee. Future
readers/writers must coordinate against replacement between validation and use.
No latest selection, legacy inference or retention here. Individual payload
writes are deliberately non-atomic and refuse existing destinations. The
standalone publisher below supplies private staging and writes a manifest last.
Validate all payloads before mutation. Restoration is not transactional: after
an I/O/device error discard the partially restored objects, never train on them.
Memory boundaries: model/EMA shard budgets bound per-shard staged tensor bytes,
not serialization overhead or total resident memory. Optimizer capture creates
a full independent CPU copy of its state. Live objects, loaded payloads, temporary
native-load casts and verification copies can coexist. Total peak CPU RAM is
neither bounded by the shard budget nor measured/qualified by these CPU tests.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType


FORMAT = "klein-recovery"
SCHEMA_VERSION = 1
LATEST_MANIFEST_VERSION = 2  # State codec envelopes remain version 1.
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_DTYPES = {"bfloat16", "float32", "float64", "float16", "int64", "int32", "int16", "int8", "uint8", "bool"}
_FIXED_PAYLOADS = {"model/config.json", "optimizer.pt", "scheduler.pt", "trainer.json", "rng.pt", "data_order.pt"}


class CheckpointValidationError(ValueError):
    """Checkpoint cannot be used as a validated recovery source."""


class LegacyCheckpointError(CheckpointValidationError):
    """Manifest-less checkpoints/exports require an explicit future migration."""


class CheckpointCompatibilityError(CheckpointValidationError):
    """The requested runtime differs from the checkpoint contract."""


class PublicationOutcome(str, Enum):
    """Publication status; retention remains unimplemented.

    FAILED: new checkpoint was not published; raise/fail the save, keep prior
    committed checkpoints. PUBLISHED: committed; retention status is reported
    separately (deferred in this phase, never represented as successful pruning).
    PUBLISHED_RETENTION_FAILED: new checkpoint remains committed; surface a
    separate retention error and stop pruning, never relabel the save as failed
    or delete the new checkpoint as rollback. Never report PUBLISHED early.
    Future reports must include this outcome, candidate/committed root, failed
    operation, exception type/message and any retention deletions already done.
    Retention failure is a visible warning/error, not a silent successful prune.
    """
    FAILED = "publication_failed"
    PUBLISHED = "published"
    PUBLISHED_RETENTION_FAILED = "published_retention_failed"


@dataclass(frozen=True)
class ValidatedCheckpoint:
    root: Path
    manifest: Mapping
    # Deliberately not a 'can_resume' or 'exact_recovery' success flag.
    qualification: str = "unqualified"


def _fail(message):
    raise CheckpointValidationError(message)


def _keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(f"{context}: expected exactly {sorted(expected)}")


def _integer(value, context, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(f"{context}: expected integer >= {minimum}")


def _text(value, context):
    if not isinstance(value, str) or not value.strip():
        _fail(f"{context}: expected nonempty string")


def _digest(value, context):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        _fail(f"{context}: expected lowercase SHA-256")


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _same(saved, current):
    # Python otherwise considers True == 1 and 1.0 == 1. These are not identical
    # normalized configurations. Accept mutable containers for caller summaries.
    if isinstance(saved, Mapping):
        return isinstance(current, Mapping) and set(saved) == set(current) and all(
            _same(value, current[key]) for key, value in saved.items())
    if isinstance(saved, tuple):
        return isinstance(current, (list, tuple)) and len(saved) == len(current) and all(
            _same(a, b) for a, b in zip(saved, current))
    return type(saved) is type(current) and saved == current


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail(f"{path.name}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    def constant(value):
        _fail(f"{path.name}: non-finite JSON number {value}")
    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            constant(value)
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=floating)
    except CheckpointValidationError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise CheckpointValidationError(f"Cannot read JSON {path.name}: {error}") from error


def _relative_path(value):
    _text(value, "payload path")
    parts = value.split("/")
    if ("\\" in value or ":" in value or "\x00" in value
            or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in parts)
            or PurePosixPath(value).is_absolute()):
        _fail(f"Unsafe payload path: {value!r}")
    return value


def _schema(manifest):
    version = manifest.get("schema_version") if isinstance(manifest, dict) else None
    if version == 2 and type(version) is int:
        return _schema_v2(manifest)
    return _schema_legacy(manifest)


def _schema_legacy(manifest, v2=False):
    _keys(manifest, {"format", "schema_version", "checkpoint_id", "run_id", "parent_checkpoint_id",
                     "created_at", "qualification", "progress", "configuration", "environment",
                     "objects", "payloads"} | ({"metadata"} if v2 else set()), "manifest")
    if manifest["format"] != FORMAT or type(manifest["schema_version"]) is not int or manifest["schema_version"] != (2 if v2 else SCHEMA_VERSION):
        _fail("Unsupported checkpoint format/schema_version; explicit migration required")
    if manifest["qualification"] != "unqualified":
        _fail("No exact-recovery qualification has been established for schema v1")
    for key in ("checkpoint_id", "run_id", "parent_checkpoint_id"):
        value = manifest[key]
        if key == "parent_checkpoint_id" and value is None:
            continue
        if not isinstance(value, str) or not _ID.fullmatch(value):
            _fail(f"Invalid {key}")
    if manifest["parent_checkpoint_id"] == manifest["checkpoint_id"]:
        _fail("Checkpoint cannot be its own parent")
    try:
        timestamp = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("timezone missing")
    except (AttributeError, TypeError, ValueError) as error:
        raise CheckpointValidationError("created_at must be an ISO timestamp with timezone") from error

    config = manifest["configuration"]
    _keys(config, {"model_fingerprint", "dataset_fingerprint", "preprocessing_fingerprint", "dataset_size",
                   "world_size", "batch_size", "gradient_accumulation_steps", "num_workers", "precision",
                   "use_cached_latents", "drop_last", "sampling_enabled", "optimizer", "optimizer_options",
                   "total_optimizer_steps", "warmup_steps", "schedule_policy", "ema_enabled", "ema_decay", "seed"},
          "configuration")
    for key in ("model_fingerprint", "dataset_fingerprint", "preprocessing_fingerprint"):
        _digest(config[key], key)
    for key in ("dataset_size", "total_optimizer_steps"):
        _integer(config[key], key, 1)
    for key in ("warmup_steps", "seed"):
        _integer(config[key], key)
    if config["seed"] >= 2 ** 64:
        _fail("seed must fit an unsigned 64-bit integer")
    if config["warmup_steps"] > config["total_optimizer_steps"] // 10:
        _fail("warmup_steps must be the resolved warmup, at most total_optimizer_steps // 10")
    expected = {"world_size": 1, "batch_size": 1, "gradient_accumulation_steps": 1, "num_workers": 0,
                "precision": "bf16", "use_cached_latents": False, "drop_last": True, "sampling_enabled": False,
                "schedule_policy": "standalone_warmup_cosine_v1"}
    for key, value in expected.items():
        if type(config[key]) is not type(value) or config[key] != value:
            _fail(f"Unsupported recovery configuration: {key}={config[key]!r}")
    if config["optimizer"] not in ("adamw", "adafactor"):
        _fail("Unsupported recovery optimizer (AdamW8bit has not been qualified for this contract)")
    if not isinstance(config["optimizer_options"], dict) or not config["optimizer_options"]:
        _fail("optimizer_options must record effective initial optimizer options")
    for key in ("lr", "weight_decay"):
        value = config["optimizer_options"].get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == "lr" and value == 0):
            _fail(f"Invalid or missing initial optimizer option: {key}")
    if type(config["ema_enabled"]) is not bool:
        _fail("ema_enabled must be boolean")
    if not (v2 and not config["ema_enabled"] and config["ema_decay"] is None) and (type(config["ema_decay"]) not in (int, float) or not math.isfinite(config["ema_decay"])
            or not 0 <= config["ema_decay"] < 1):
        _fail("ema_decay must be finite and in [0, 1)")

    progress = manifest["progress"]
    _keys(progress, {"completed_optimizer_steps", "attempted_optimizer_steps", "skipped_optimizer_steps",
                     "loop_iterations", "epoch", "next_batch_index", "accumulation_step"}, "progress")
    for key, value in progress.items():
        _integer(value, key)
    # v1: one microbatch per loop iteration/update attempt; skips consume data too.
    if (progress["attempted_optimizer_steps"] != progress["completed_optimizer_steps"] + progress["skipped_optimizer_steps"]
            or progress["loop_iterations"] != progress["attempted_optimizer_steps"]
            or progress["accumulation_step"] != 0
            or progress["attempted_optimizer_steps"] > config["total_optimizer_steps"]):
        _fail("Inconsistent optimizer/loop counters or non-boundary checkpoint")
    # End-of-epoch positions are normalized to next epoch, batch zero.
    if (progress["next_batch_index"] >= config["dataset_size"]
            or progress["loop_iterations"] != progress["epoch"] * config["dataset_size"] + progress["next_batch_index"]):
        _fail("Inconsistent consumed-data position")

    if not v2:
        environment = manifest["environment"]
        _keys(environment, {"python", "dependencies", "trainer_commit", "cuda", "platform", "device_type",
                            "device_name", "backend_settings"}, "environment")
        for key in ("python", "trainer_commit", "cuda", "platform", "device_name"):
            _text(environment[key], key)
        if environment["device_type"] != "cuda":
            _fail("Unsupported recovery device_type")
        _keys(environment["dependencies"], {"torch", "torchvision", "transformers", "diffusers", "accelerate",
                                            "safetensors", "numpy", "pillow"}, "dependencies")
        for key, value in environment["dependencies"].items():
            _text(value, key)
        backends = environment["backend_settings"]
        _keys(backends, {"deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
                         "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision"}, "backend_settings")
        for key, value in backends.items():
            if key == "float32_matmul_precision":
                if value not in ("highest", "high", "medium"):
                    _fail("Invalid float32_matmul_precision")
            elif type(value) is not bool:
                _fail(f"Backend flag {key} must be boolean")

    objects = manifest["objects"]
    _keys(objects, {"model_class", "model_tensors", "optimizer_class", "parameter_groups", "scheduler_class"}, "objects")
    for key in ("model_class", "optimizer_class", "scheduler_class"):
        _text(objects[key], key)
    expected_classes = {
        "model_class": "diffusers.Flux2Transformer2DModel",
        "optimizer_class": {"adamw": "torch.optim.AdamW", "adafactor": "transformers.Adafactor"}[config["optimizer"]],
        "scheduler_class": "torch.optim.lr_scheduler.CosineAnnealingLR",
    }
    if any(objects[key] != value for key, value in expected_classes.items()):
        _fail("Unsupported or conflicting training object class")
    tensors = objects["model_tensors"]
    if not isinstance(tensors, list) or not tensors:
        _fail("model_tensors must be a nonempty inventory")
    names, parameters = set(), set()
    for tensor in tensors:
        _keys(tensor, {"name", "shape", "dtype", "kind"}, "model tensor")
        _text(tensor["name"], "tensor name")
        if tensor["name"] in names:
            _fail("Duplicate model tensor")
        names.add(tensor["name"])
        if not isinstance(tensor["shape"], list):
            _fail("Tensor shape must be a list")
        for dim in tensor["shape"]:
            _integer(dim, "tensor dimension")
        if not isinstance(tensor["dtype"], str) or tensor["dtype"] not in _DTYPES or tensor["kind"] not in ("parameter", "buffer"):
            _fail("Invalid tensor dtype/kind")
        if tensor["kind"] == "parameter":
            if tensor["dtype"] != "bfloat16":
                _fail("Candidate BF16 recovery requires BF16 parameter storage")
            parameters.add(tensor["name"])
    groups = objects["parameter_groups"]
    if not isinstance(groups, list) or not groups or any(not isinstance(g, list) or not g for g in groups):
        _fail("parameter_groups must be nonempty ordered lists of parameter names")
    flattened = [name for group in groups for name in group]
    if any(not isinstance(name, str) for name in flattened) or len(set(flattened)) != len(flattened) or set(flattened) != parameters:
        _fail("Optimizer parameter membership must cover all parameters exactly once")
    return names, parameters


def v2_configuration(metadata):
    """Explicit v2 codec projection; never used to reinterpret a v1 checkpoint."""
    from scripts.klein_recovery_metadata import validate_metadata_compatibility
    validate_metadata_compatibility(metadata, metadata)
    d = metadata["document"]
    c = d["configuration"]
    if len(c["optimizer_options"]) != 1:
        _fail("Trainer recovery v2 currently supports one optimizer group")
    fields = ("dataset_size", "world_size", "batch_size", "gradient_accumulation_steps",
              "num_workers", "precision", "use_cached_latents", "drop_last", "sampling_enabled",
              "optimizer", "total_optimizer_steps", "warmup_steps", "schedule_policy",
              "ema_enabled", "ema_decay", "seed")
    result = {k: c[k] for k in fields}
    result["optimizer_options"] = c["optimizer_options"][0]
    for key in ("model", "dataset", "preprocessing"):
        result[key + "_fingerprint"] = d["fingerprints"][key]["sha256"]
    return result


def _schema_v2(manifest):
    """V2 binds the full foundation. State payload codec version remains 1.

    Version 1 remains independently readable; it cannot be resumed by the v2
    trainer because missing execution/source settings cannot be reconstructed.
    The common tensor/progress validation below is reused without migrating v1.
    """
    from scripts.klein_recovery_metadata import validate_metadata_compatibility
    expected = {"format", "schema_version", "checkpoint_id", "run_id", "parent_checkpoint_id",
                "created_at", "qualification", "progress", "configuration", "environment",
                "objects", "payloads", "metadata"}
    _keys(manifest, expected, "v2 manifest")
    metadata = manifest["metadata"]
    validate_metadata_compatibility(metadata, metadata)
    source_paths = {entry["path"] for entry in metadata["document"]["fingerprints"]["source_code"]["document"]["files"]}
    if not {"scripts/klein_model_resolver.py", "scripts/klein_recovery_training.py"} <= source_paths:
        _fail("V2 requires resolver and recovery runtime source fingerprints")
    config = v2_configuration(metadata)
    if not _same(config, manifest["configuration"]) or not _same(
            metadata["document"]["environment"], manifest["environment"]):
        _fail("V2 metadata disagrees with configuration/environment")
    if not isinstance(manifest["objects"], dict):
        _fail("V2 objects must be a dictionary")
    if not _same(metadata["document"]["configuration"]["parameter_groups"],
                 manifest["objects"].get("parameter_groups")):
        _fail("V2 metadata parameter groups disagree")
    # Reuse the structural v1 rules, with explicitly bypassed legacy-only fields.
    return _schema_legacy(manifest, v2=True)


def _weight_layout(paths, directory, stem, tensor_names, root):
    weights = f"{directory}/{stem}.safetensors"
    index = weights + ".index.json"
    subset = {p for p in paths if p.startswith(directory + "/") and p != "model/config.json"}
    if subset == {weights}:
        return
    pattern = re.compile(re.escape(f"{directory}/{stem}") + r"-(\d{5})-of-(\d{5})\.safetensors\Z")
    shards = subset - {index}
    matches = [pattern.fullmatch(p) for p in shards]
    if index not in subset or not shards or any(m is None for m in matches):
        _fail(f"Unexpected or incomplete {directory} weight layout")
    count = len(shards)
    if {int(m[1]) for m in matches} != set(range(1, count + 1)) or any(int(m[2]) != count for m in matches):
        _fail(f"Inconsistent {directory} shard numbering")
    content = _read_json(root / index)
    _keys(content, {"metadata", "weight_map"}, index)
    if not isinstance(content["metadata"], dict) or not isinstance(content["weight_map"], dict):
        _fail(f"Invalid {index}")
    weight_map = content["weight_map"]
    if set(weight_map) != tensor_names or any(not isinstance(p, str) or "/" in p or "\\" in p for p in weight_map.values()):
        _fail(f"Invalid tensor mapping in {index}")
    if {f"{directory}/{p}" for p in weight_map.values()} != shards:
        _fail(f"Index references missing/unexpected shards in {index}")


def _scan(root):
    """Reject links (including Windows junctions), special files and unknown dirs."""
    result = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                _fail(f"Linked checkpoint entry is forbidden: {path.name}")
            relative = path.relative_to(root).as_posix()
            mode = path.stat().st_mode
            if stat.S_ISDIR(mode):
                if relative not in ("model", "ema"):
                    _fail(f"Unexpected checkpoint directory: {relative}")
                pending.append(path)
            elif stat.S_ISREG(mode):
                result.add(relative)
            else:
                _fail(f"Non-regular checkpoint entry: {relative}")
    return result


def validate_checkpoint(root):
    """Validate an explicit canonical root, returning immutable metadata only.

    Missing manifests are legacy/unsupported, never interpreted as exports or
    accelerator_state roots. Staging roots are rejected even with a manifest.
    Opaque payload content interpretation requires read_recovery_states.
    """
    root = Path(root).absolute()
    if any(part.startswith(".incomplete-") for part in root.parts):
        _fail("Incomplete staging directory cannot be resumed")
    return _validate_checkpoint_contents(root)


def _validate_checkpoint_contents(root):
    """Internal shared integrity implementation, not a staging resume API.

    Only the publisher may use this on its own freshly created private staging
    root. Public validation always rejects staging before reaching this helper.
    """
    root = Path(root).absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (root, *root.parents)):
        _fail("Linked checkpoint root/ancestor is forbidden")
    if not root.is_dir():
        _fail("Checkpoint root must be an existing directory")
    if not (root / "manifest.json").exists():
        raise LegacyCheckpointError("Missing manifest.json: legacy checkpoint or inference export; explicit migration required")
    try:
        files = _scan(root)
        manifest = _read_json(root / "manifest.json")
        names, parameters = _schema(manifest)
        payloads = manifest["payloads"]
        if not isinstance(payloads, list) or not payloads:
            _fail("payloads must be a nonempty list")
        paths, folded = set(), set()
        for entry in payloads:
            _keys(entry, {"path", "size_bytes", "sha256"}, "payload")
            path = _relative_path(entry["path"])
            if path.casefold() in folded:
                _fail(f"Duplicate/case-colliding payload: {path}")
            folded.add(path.casefold())
            paths.add(path)
            _integer(entry["size_bytes"], "size_bytes", 1)
            _digest(entry["sha256"], "payload sha256")
        if not _FIXED_PAYLOADS <= paths or "manifest.json" in paths:
            _fail("Missing required payloads or manifest listed as payload")
        allowed_roots = _FIXED_PAYLOADS | {p for p in paths if p.startswith(("model/", "ema/"))}
        if paths != allowed_roots:
            _fail("Unexpected payload entry")
        if files != paths | {"manifest.json"}:
            _fail(f"Payload inventory mismatch: missing={sorted(paths - files)}, unexpected={sorted(files - paths - {'manifest.json'})}")
        for entry in payloads:
            path = root / entry["path"]
            if path.stat().st_size != entry["size_bytes"]:
                _fail(f"Payload size mismatch: {entry['path']}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                _fail(f"Payload checksum mismatch: {entry['path']}")
        _weight_layout(paths, "model", "diffusion_pytorch_model", names, root)
        json_payloads = {name: _read_json(root / name) for name in ("model/config.json", "trainer.json")}
        for name, content in json_payloads.items():
            if not isinstance(content, dict):
                _fail(f"{name} must contain a JSON object")
        if json_payloads["model/config.json"].get("_class_name") != "Flux2Transformer2DModel":
            _fail("Conflicting model/config.json class")
        if manifest["configuration"]["ema_enabled"]:
            _weight_layout(paths, "ema", "weights", parameters, root)
        elif any(p.startswith("ema/") for p in paths) or (root / "ema").exists():
            _fail("Unexpected EMA payloads/directory when EMA is disabled")
        return ValidatedCheckpoint(root.resolve(), _freeze(manifest))
    except OSError as error:
        raise CheckpointValidationError(f"Checkpoint filesystem validation failed: {error}") from error


def validate_configuration_compatibility(checkpoint, *, configuration, environment):
    """Before allocation: strict resolved settings/version comparison, no overrides.

    Output paths/retention/logging frequency are intentionally absent from the
    configuration schema. A match does not establish recovery qualification.
    """
    for section, current in (("configuration", configuration), ("environment", environment)):
        if not _same(checkpoint.manifest[section], current):
            raise CheckpointCompatibilityError(f"{section} differs from checkpoint; migration/qualification required")


def validate_object_compatibility(checkpoint, objects):
    """After construction/preparation: compare classes, tensor schema and group order.

    Caller supplies the same normalized objects schema as the manifest. Do not
    compare initial optimizer tensor values to trained state here. Validation
    of deserialized optimizer/EMA/RNG contents is performed by read_recovery_states.
    """
    if not _same(checkpoint.manifest["objects"], objects):
        raise CheckpointCompatibilityError("Constructed training objects differ from checkpoint")


# Tensor imports are lazy: manifest/integrity checks still need only stdlib.
def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _number(value, context, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (minimum is not None and value < minimum):
        _fail(f"{context}: invalid finite number")


def _tree_copy(value):
    """Independent restricted-load-compatible CPU snapshot, never GPU clones."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.device.type == "meta":
            _fail("Only materialized dense tensors are supported")
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, dict):
        if any(type(k) not in (str, int) for k in value):
            _fail("Unsupported state dictionary key")
        return {k: _tree_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_copy(v) for v in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _fail(f"Unsupported state value: {type(value).__name__}")


def _equal_state(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal_state(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def write_state_payload(path, checkpoint_id, state, **context):
    """Write one validated trainer.json or restricted .pt envelope, exclusively.

    Context is forwarded to validate_state_payload; caller owns staging and
    hashes/publication. This never writes manifest.json or publishes a checkpoint.
    """
    import torch
    path = Path(path)
    envelope = {"schema_version": SCHEMA_VERSION, "checkpoint_id": checkpoint_id, "state": state}
    validate_state_payload(path.name, envelope, checkpoint_id, **context)
    if path.name == "trainer.json":
        with path.open("x", encoding="utf-8") as stream:
            json.dump(envelope, stream, allow_nan=False, sort_keys=True)
    else:
        with path.open("xb") as stream:
            torch.save(envelope, stream)


def read_state_payload(path, checkpoint_id, **context):
    """Read with weights_only=True on CPU; no arbitrary-object pickle fallback."""
    import torch
    path = Path(path)
    try:
        envelope = (_read_json(path) if path.name == "trainer.json" else
                    torch.load(path, map_location="cpu", weights_only=True))
    except Exception as error:
        raise CheckpointValidationError(f"Cannot decode {path.name}: {error}") from error
    return validate_state_payload(path.name, envelope, checkpoint_id, **context)


def validate_state_payload(name, envelope, checkpoint_id, **context):
    _keys(envelope, {"schema_version", "checkpoint_id", "state"}, name)
    if (type(envelope["schema_version"]) is not int or envelope["schema_version"] != SCHEMA_VERSION
            or not isinstance(checkpoint_id, str) or not _ID.fullmatch(checkpoint_id)
            or envelope["checkpoint_id"] != checkpoint_id):
        _fail(f"{name}: checkpoint identity/schema mismatch")
    if "manifest" in context and context["manifest"]["checkpoint_id"] != checkpoint_id:
        _fail(f"{name}: checkpoint identity differs from manifest")
    validators = {"trainer.json": validate_trainer_state, "optimizer.pt": validate_optimizer_state,
                  "scheduler.pt": validate_scheduler_state, "rng.pt": validate_rng_state,
                  "data_order.pt": validate_data_order_state}
    if name not in validators:
        _fail(f"Unknown state payload {name}")
    try:
        validators[name](envelope["state"], **context)
    except CheckpointValidationError:
        raise
    except (TypeError, ValueError, KeyError, RuntimeError, AttributeError, IndexError) as error:
        raise CheckpointValidationError(f"Malformed {name}: {error}") from error
    return envelope["state"]


def validate_trainer_state(state, *, manifest):
    manifest = _plain(manifest)
    _schema(manifest)
    _keys(state, {"progress", "global_step", "ema_updates", "cosine_updates", "group_lrs",
                  "accelerator_step", "loss_accumulator", "log_steps"}, "trainer state")
    if not _same(state["progress"], manifest["progress"]):
        _fail("Trainer progress differs from manifest")
    p, c = manifest["progress"], manifest["configuration"]
    for key in ("global_step", "ema_updates", "cosine_updates", "accelerator_step", "log_steps"):
        _integer(state[key], key)
    a, completed, warmup = p["attempted_optimizer_steps"], p["completed_optimizer_steps"], c["warmup_steps"]
    if (state["global_step"] != a or state["ema_updates"] != (a if c["ema_enabled"] else 0)
            or not max(0, completed - min(a, warmup)) <= state["cosine_updates"] <= min(completed, max(0, a - warmup))
            or state["log_steps"] > a):
        _fail("Inconsistent trainer counters")
    _number(state["loss_accumulator"], "loss_accumulator")
    _validate_lrs(state["group_lrs"], len(manifest["objects"]["parameter_groups"]))


def _validate_lrs(lrs, count):
    if not isinstance(lrs, list) or len(lrs) != count:
        _fail("LR group count mismatch")
    for lr in lrs:
        _number(lr, "group LR", 0)


def validate_scheduler_state(state, *, scheduler, configuration, trainer_state):
    import torch
    if type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        _fail("Pass the native prepared scheduler; only CosineAnnealingLR is supported")
    _keys(state, {"state_dict", "cosine_updates", "group_lrs"}, "scheduler state")
    native = state["state_dict"]
    _keys(native, scheduler.state_dict(), "native scheduler state")
    _integer(state["cosine_updates"], "cosine_updates")
    _validate_lrs(state["group_lrs"], len(scheduler.optimizer.param_groups))
    c = configuration
    updates = state["cosine_updates"]
    _integer(native["T_max"], "T_max", 1)
    _validate_lrs(native["base_lrs"], len(state["group_lrs"]))
    if (updates != trainer_state["cosine_updates"] or state["group_lrs"] != trainer_state["group_lrs"]
            or native["T_max"] != c["total_optimizer_steps"] - c["warmup_steps"]
            or native["eta_min"] != c["optimizer_options"]["lr"] * 0.1
            or native["base_lrs"] != [c["optimizer_options"]["lr"]] * len(state["group_lrs"])
            or type(native["last_epoch"]) is not int or native["last_epoch"] != updates
            or type(native["_step_count"]) is not int or native["_step_count"] != updates + 1):
        _fail("Scheduler configuration/counters/LRs disagree")
    _validate_lrs(native["_last_lr"], len(state["group_lrs"]))
    for key in ("_is_initial", "_get_lr_called_within_step"):
        if key in native and native[key] is not False:
            _fail("Invalid scheduler flag")
    # Pinned Torch 2.10 uses a recurrence based on the current optimizer LR.
    # Here the last manual warmup assignment restores base_lr even on a skip;
    # cosine calls start only afterwards. With no subsequent external LR edits,
    # the recurrence telescopes to this closed form (within rounding tolerance).
    # This assumption is specific to standalone_warmup_cosine_v1, not a general
    # rule for arbitrarily modified CosineAnnealingLR instances. During warmup,
    # native _last_lr remains base_lr while optimizer LR changes independently.
    # Validation does not step the scheduler or assign/reconstruct a restored LR.
    attempts = trainer_state["global_step"]
    _integer(attempts, "global_step")
    warmup = c["warmup_steps"]
    base = c["optimizer_options"]["lr"]
    last_lr = base if updates == 0 else native["eta_min"] + (base - native["eta_min"]) * (
        1 + math.cos(math.pi * updates / native["T_max"])) / 2
    current_lr = base * attempts / warmup if warmup and 0 < attempts <= warmup else last_lr
    if (any(not math.isclose(v, last_lr, rel_tol=1e-12, abs_tol=1e-15) for v in native["_last_lr"])
            or any(not math.isclose(v, current_lr, rel_tol=1e-12, abs_tol=1e-15) for v in state["group_lrs"])):
        _fail("Recorded LR disagrees with warmup/cosine trajectory")


def capture_scheduler_state(scheduler, cosine_updates):
    return {"state_dict": _tree_copy(scheduler.state_dict()), "cosine_updates": cosine_updates,
            "group_lrs": [g["lr"] for g in scheduler.optimizer.param_groups]}


def restore_scheduler_state(scheduler, state, *, configuration, trainer_state):
    """After optimizer restoration; never call step or reconstruct warmup."""
    validate_scheduler_state(state, scheduler=scheduler, configuration=configuration, trainer_state=trainer_state)
    if [g["lr"] for g in scheduler.optimizer.param_groups] != state["group_lrs"]:
        _fail("Restore optimizer LRs before scheduler state")
    scheduler.load_state_dict(_tree_copy(state["state_dict"]))
    if (not _equal_state(scheduler.state_dict(), state["state_dict"])
            or [g["lr"] for g in scheduler.optimizer.param_groups] != state["group_lrs"]):
        _fail("Scheduler restoration verification failed")


def restore_trainer_state(state, *, manifest):
    """Return validated independent counters for future caller assignment."""
    validate_trainer_state(state, manifest=manifest)
    return _tree_copy(state)


def _optimizer_kind(optimizer):
    import torch
    if type(optimizer) is torch.optim.AdamW:
        return "adamw"
    from transformers.optimization import Adafactor
    if type(optimizer) is Adafactor:
        return "adafactor"
    _fail("Only native AdamW/Adafactor supported; unwrap AFTER Accelerate preparation")


def _group_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    try:
        groups = [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    except KeyError:
        _fail("Optimizer contains parameters outside model")
    flat = [n for g in groups for n in g]
    if not groups or any(not g for g in groups) or len(set(flat)) != len(flat) or set(flat) != set(names.values()):
        _fail("Optimizer must cover every model parameter exactly once")
    return groups


def _optimizer_inventory(native, groups):
    import torch
    metadata, counters = [], {}
    def visit(value, owner, path):
        if isinstance(value, torch.Tensor):
            metadata.append({"owner": owner, "path": path, "shape": list(value.shape),
                             "dtype": str(value.dtype).removeprefix("torch.")})
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], owner, path + [key])
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, owner, path + [i])
    for index, (group, names) in enumerate(zip(native["param_groups"], groups)):
        visit({k: v for k, v in group.items() if k != "params"}, index, [])
        for pid, name in zip(group["params"], names):
            state = native["state"].get(pid, {})
            visit(state, name, [])
            step = state.get("step")
            counters[name] = (None if not state else
                              {"type": str(step.dtype).removeprefix("torch.") if isinstance(step, torch.Tensor) else "int",
                               "value": step.item() if isinstance(step, torch.Tensor) else step})
    return metadata, counters


def capture_optimizer_state(model, optimizer, *, completed_steps):
    """CPU snapshot; optimizer payload memory is not sharded in v1."""
    _optimizer_kind(optimizer)
    groups = _group_names(model, optimizer)
    native = _tree_copy(optimizer.state_dict())
    metadata, counters = _optimizer_inventory(native, groups)
    state = {"state_dict": native, "parameter_groups": groups,
             "tensor_metadata": metadata, "step_counters": counters}
    validate_optimizer_state(state, model=model, optimizer=optimizer, completed_steps=completed_steps)
    return state


def _check_tensor(tensor, shape, dtype, context):
    import torch
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
            or tensor.layout != torch.strided or list(tensor.shape) != list(shape) or tensor.dtype != dtype):
        _fail(f"{context}: tensor shape/dtype/device mismatch")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        _fail(f"{context}: nonfinite tensor")


def validate_optimizer_state(state, *, model, optimizer, completed_steps, group_lrs=None):
    import torch
    kind = _optimizer_kind(optimizer)
    _integer(completed_steps, "completed_steps")
    _keys(state, {"state_dict", "parameter_groups", "tensor_metadata", "step_counters"}, "optimizer state")
    groups = _group_names(model, optimizer)
    if state["parameter_groups"] != groups:
        _fail("Optimizer group name/order mismatch")
    native = state["state_dict"]
    _keys(native, {"state", "param_groups"}, "native optimizer")
    if not isinstance(native["state"], dict) or not isinstance(native["param_groups"], list) or len(native["param_groups"]) != len(groups):
        _fail("Invalid native optimizer groups/state")
    parameters = dict(model.named_parameters())
    seen = set()
    for group, names in zip(native["param_groups"], groups):
        # Optimizer.__setstate__ adds differentiable=False to Adafactor.defaults
        # during native loading, but not to its saved parameter groups.
        required = set(optimizer.defaults) - ({"differentiable"} if kind == "adafactor" else set())
        allowed = required | {"params", "initial_lr"}
        if (not isinstance(group, dict) or not required | {"params"} <= group.keys()
                or set(group) - allowed or not isinstance(group["params"], list) or len(group["params"]) != len(names)):
            _fail("Optimizer group options/schema mismatch")
        for key in ("lr", "weight_decay", "initial_lr"):
            if key in group:
                _number(group[key], key, 0)
        if kind == "adamw":
            _number(group["eps"], "eps", 0)
            if not isinstance(group["betas"], tuple) or len(group["betas"]) != 2:
                _fail("Invalid AdamW betas")
            for beta in group["betas"]:
                _number(beta, "beta", 0)
                if beta >= 1:
                    _fail("Invalid AdamW beta")
            for key in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"):
                if key in group and type(group[key]) is not bool:
                    _fail("Invalid AdamW option")
            for key in ("foreach", "fused"):
                if group[key] is not None and type(group[key]) is not bool:
                    _fail("Invalid AdamW option")
            if group["capturable"] or group["differentiable"] or group["fused"] or not group.get("decoupled_weight_decay", True):
                _fail("Unsupported AdamW execution options")
        else:
            if any(group[k] is not False for k in ("relative_step", "scale_parameter", "warmup_init")):
                _fail("Only standalone fixed-LR Adafactor options supported")
            if not isinstance(group["eps"], tuple) or len(group["eps"]) != 2:
                _fail("Invalid Adafactor eps")
            for eps in group["eps"]:
                _number(eps, "eps", 0)
            _number(group["clip_threshold"], "clip_threshold", 0)
            _number(group["decay_rate"], "decay_rate")
            if group["clip_threshold"] == 0 or group["decay_rate"] >= 0:
                _fail("Invalid Adafactor decay/clip options")
            if group["beta1"] is not None:
                _number(group["beta1"], "beta1", 0)
                if group["beta1"] >= 1:
                    _fail("Invalid beta1")
        for pid, name in zip(group["params"], names):
            if type(pid) is not int or pid < 0 or pid in seen:
                _fail("Invalid/duplicate optimizer parameter ID")
            seen.add(pid)
            entry = native["state"].get(pid, {})
            if not isinstance(entry, dict):
                _fail("Invalid parameter state")
            if not entry:
                continue
            p = parameters[name]
            if kind == "adamw":
                expected = {"step": ([], torch.float32), "exp_avg": (p.shape, p.dtype),
                            "exp_avg_sq": (p.shape, p.dtype)}
                if group["amsgrad"]:
                    expected["max_exp_avg_sq"] = (p.shape, p.dtype)
            else:
                dtype = torch.float32 if p.dtype in (torch.bfloat16, torch.float16) else p.dtype
                expected = {"RMS": ([], dtype)}
                if p.ndim >= 2:
                    expected.update(exp_avg_sq_row=(p.shape[:-1], dtype),
                                    exp_avg_sq_col=(p.shape[:-2] + p.shape[-1:], dtype))
                else:
                    expected["exp_avg_sq"] = (p.shape, dtype)
                if group["beta1"] is not None:
                    expected["exp_avg"] = (p.shape, dtype)
            _keys(entry, set(expected) | {"step"}, "parameter state")
            for key, (shape, dtype) in expected.items():
                _check_tensor(entry[key], shape, dtype, f"{name}.{key}")
                if key in ("RMS", "exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col", "max_exp_avg_sq") and (entry[key] < 0).any().item():
                    _fail("Optimizer squared moments/RMS must be nonnegative")
            step = entry["step"].item() if kind == "adamw" else entry["step"]
            if (type(step) not in (int, float) or isinstance(step, bool) or not math.isfinite(step)
                    or int(step) != step or not 1 <= step <= completed_steps
                    or (kind == "adafactor" and type(step) is not int)):
                _fail("Invalid optimizer step counter")
    if any(type(pid) is not int for pid in native["state"]) or not native["state"].keys() <= seen:
        _fail("Unknown optimizer parameter state")
    metadata, counters = _optimizer_inventory(native, groups)
    if not _equal_state(metadata, state["tensor_metadata"]) or not _equal_state(counters, state["step_counters"]):
        _fail("Optimizer inventory/counters mismatch")
    if completed_steps and not any(c is not None for c in counters.values()):
        _fail("Completed steps lack initialized optimizer state")
    if group_lrs is not None and [g["lr"] for g in native["param_groups"]] != group_lrs:
        _fail("Optimizer LRs disagree with trainer")


def restore_optimizer_state(model, optimizer, state, *, completed_steps, group_lrs=None):
    """Call on native optimizer AFTER accelerator.prepare (wrapper unqualified).

    Native load establishes associations/options, then pristine CPU tensors
    replace cast state. Do not subsequently invoke a wrapper load_state_dict.
    """
    import torch
    validate_optimizer_state(state, model=model, optimizer=optimizer,
                             completed_steps=completed_steps, group_lrs=group_lrs)
    native = state["state_dict"]
    optimizer.load_state_dict(native)
    for saved_group, live_group in zip(native["param_groups"], optimizer.param_groups):
        for pid, parameter in zip(saved_group["params"], live_group["params"]):
            original = native["state"].get(pid, {})
            restored = {}
            for key, value in original.items():
                if isinstance(value, torch.Tensor):
                    device = "cpu" if key == "step" else parameter.device
                    # copy=True avoids aliasing saved tensors even on CPU.
                    restored[key] = value.to(device=device, copy=True)
                else:
                    restored[key] = value
            if original:
                optimizer.state[parameter] = restored
    actual = optimizer.state_dict()
    # Native IDs may be renumbered; compare using ordered group membership.
    for saved_group, live_group in zip(native["param_groups"], actual["param_groups"]):
        if not _equal_state({k: v for k, v in saved_group.items() if k != "params"},
                            {k: v for k, v in live_group.items() if k != "params"}):
            _fail("Restored optimizer group options differ")
        for old, new in zip(saved_group["params"], live_group["params"]):
            if not _equal_state(native["state"].get(old, {}), actual["state"].get(new, {})):
                _fail("Restored optimizer values/dtypes/counters differ")


def _rng_tensor(value, device="cpu"):
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.device.type != "cpu" or value.ndim != 1:
        _fail("RNG state must be a CPU uint8 vector")
    try:
        torch.Generator(device=device).set_state(value)
    except (RuntimeError, TypeError) as error:
        raise CheckpointValidationError(f"Invalid RNG generator state: {error}") from error


def capture_rng_state(*, cuda_devices):
    """Explicit device count: production v1 requires one; CPU codec tests use zero."""
    import random
    import numpy as np
    import torch
    _integer(cuda_devices, "cuda_devices")
    if cuda_devices != torch.cuda.device_count():
        _fail("CUDA RNG device count differs from runtime")
    n = np.random.get_state()
    state = {"python": random.getstate(),
             "numpy": {"algorithm": n[0], "keys": torch.tensor(n[1].astype("int64")),
                       "position": n[2], "has_gauss": n[3], "cached_gaussian": n[4]},
             "torch_cpu": torch.get_rng_state(),
             "torch_cuda": [torch.cuda.get_rng_state(i).cpu() for i in range(cuda_devices)]}
    validate_rng_state(state, cuda_devices=cuda_devices)
    return state


def validate_rng_state(state, *, cuda_devices):
    import random
    import numpy as np
    import torch
    _integer(cuda_devices, "cuda_devices")
    _keys(state, {"python", "numpy", "torch_cpu", "torch_cuda"}, "RNG state")
    p = state["python"]
    if (not isinstance(p, tuple) or len(p) != 3 or type(p[0]) is not int or p[0] != 3
            or not isinstance(p[1], tuple) or len(p[1]) != 625
            or any(type(x) is not int or not 0 <= x <= 2**32 - 1 for x in p[1][:-1])
            or type(p[1][-1]) is not int or not 0 <= p[1][-1] <= 624):
        _fail("Invalid Python RNG state")
    if p[2] is not None:
        _number(p[2], "Python cached Gaussian")
    random.Random().setstate(p)
    n = state["numpy"]
    _keys(n, {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}, "NumPy RNG")
    _check_tensor(n["keys"], [624], torch.int64, "NumPy keys")
    if (n["algorithm"] != "MT19937" or ((n["keys"] < 0) | (n["keys"] >= 2**32)).any().item()
            or type(n["position"]) is not int or not 0 <= n["position"] <= 624
            or type(n["has_gauss"]) is not int or n["has_gauss"] not in (0, 1)):
        _fail("Invalid NumPy RNG state")
    _number(n["cached_gaussian"], "NumPy cached Gaussian")
    np.random.RandomState().set_state((n["algorithm"], n["keys"].numpy().astype("uint32"),
                                     n["position"], n["has_gauss"], n["cached_gaussian"]))
    _rng_tensor(state["torch_cpu"])
    if not isinstance(state["torch_cuda"], list) or len(state["torch_cuda"]) != cuda_devices:
        _fail("CUDA RNG state count mismatch")
    if cuda_devices != torch.cuda.device_count():
        _fail("Cannot validate CUDA RNG on a different device count")
    for i, value in enumerate(state["torch_cuda"]):
        _rng_tensor(value, f"cuda:{i}")


def restore_rng_state(state, *, cuda_devices):
    """Strict restoration, last in recovery after models and iterators exist."""
    import random
    import numpy as np
    import torch
    validate_rng_state(state, cuda_devices=cuda_devices)
    n = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((n["algorithm"], n["keys"].numpy().astype("uint32"), n["position"],
                         n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    for i, value in enumerate(state["torch_cuda"]):
        torch.cuda.set_rng_state(value, i)
    if not _equal_state(capture_rng_state(cuda_devices=cuda_devices), state):
        _fail("RNG restoration verification failed")


def capture_data_order_state(*, progress, dataset_fingerprint, permutation, order_generator, loader_generator):
    """Snapshot acknowledged position; caller supplies already generated order."""
    if order_generator is loader_generator or str(order_generator.device) != "cpu" or str(loader_generator.device) != "cpu":
        _fail("Independent data generators must be CPU generators")
    epoch, cursor = progress["epoch"], progress["next_batch_index"]
    return {"epoch": epoch, "next_batch_index": cursor,
            "permutation_epoch": epoch if cursor else epoch - 1,
            "permutation": _tree_copy(permutation), "dataset_fingerprint": dataset_fingerprint,
            "order_generator_state": order_generator.get_state().clone(),
            "loader_generator_state": loader_generator.get_state().clone()}


def validate_data_order_state(state, *, progress, dataset_size, dataset_fingerprint):
    import torch
    _keys(state, {"epoch", "next_batch_index", "permutation_epoch", "permutation",
                  "dataset_fingerprint", "order_generator_state", "loader_generator_state"}, "data order")
    _integer(dataset_size, "dataset_size", 1)
    _digest(dataset_fingerprint, "dataset_fingerprint")
    for key in ("epoch", "next_batch_index"):
        _integer(state[key], key)
        if state[key] != progress[key]:
            _fail("Data cursor differs from trainer progress")
    _integer(state["permutation_epoch"], "permutation_epoch", -1)
    epoch, cursor = state["epoch"], state["next_batch_index"]
    if (cursor >= dataset_size or progress["loop_iterations"] != epoch * dataset_size + cursor
            or progress["attempted_optimizer_steps"] != progress["loop_iterations"]
            or state["permutation_epoch"] != (epoch if cursor else epoch - 1)
            or state["dataset_fingerprint"] != dataset_fingerprint):
        _fail("Inconsistent data position/identity")
    size = 0 if epoch == 0 and cursor == 0 else dataset_size
    _check_tensor(state["permutation"], [size], torch.int64, "permutation")
    if size and not torch.equal(torch.sort(state["permutation"]).values, torch.arange(size)):
        _fail("Permutation is not a unique dataset index range")
    _rng_tensor(state["order_generator_state"])
    _rng_tensor(state["loader_generator_state"])


def restore_data_order_state(state, order_generator, loader_generator, **context):
    """Restore generators and return (epoch, cursor, permutation); no DataLoader."""
    validate_data_order_state(state, **context)
    if order_generator is loader_generator or str(order_generator.device) != "cpu" or str(loader_generator.device) != "cpu":
        _fail("Independent data generators must be CPU generators")
    order_generator.set_state(state["order_generator_state"])
    loader_generator.set_state(state["loader_generator_state"])
    return state["epoch"], state["next_batch_index"], state["permutation"].clone()


def _model_tensors(model):
    """state_dict returns references, not a duplicate model allocation."""
    import torch
    tensors = dict(model.state_dict(keep_vars=True))
    # Recovery includes nonpersistent buffers too; inference state_dict omits
    # them. These are live references and are copied in place on restoration.
    for name, tensor in model.named_buffers(remove_duplicate=False):
        tensors.setdefault(name, tensor)
    if not tensors or any(not isinstance(v, torch.Tensor) for v in tensors.values()):
        _fail("Only tensor model state is supported")
    return tensors


def tensor_inventory(model):
    parameters = set(dict(model.named_parameters()))
    return [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
             "kind": "parameter" if name in parameters else "buffer"}
            for name, t in _model_tensors(model).items()]


def _check_tensor_targets(tensors):
    import torch
    storage = set()
    for name, tensor in tensors.items():
        _text(name, "tensor name")
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type == "meta" or str(tensor.dtype).removeprefix("torch.") not in _DTYPES):
            _fail("Unsupported tensor target")
        identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if tensor.numel() and identity in storage:
            _fail("Aliased/tied tensor storage is not supported by this codec")
        if tensor.numel():
            storage.add(identity)


def _write_tensors(directory, tensors, stem, max_shard_bytes, *, before_write=None):
    """Bounded CPU staging; an individual oversize tensor fails before writing.

    Budget bounds staged tensor bytes per shard, not serialization overhead/RSS.
    No tensor splitting: caller must allow at least the largest single tensor.
    """
    from safetensors.torch import save_file
    _integer(max_shard_bytes, "max_shard_bytes", 1)
    _check_tensor_targets(tensors)
    shards, shard, size = [], [], 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > max_shard_bytes:
            _fail(f"Tensor {name} exceeds shard budget; increase max_shard_bytes explicitly")
        if shard and size + nbytes > max_shard_bytes:
            shards.append(shard)
            shard, size = [], 0
        shard.append(name)
        size += nbytes
    if shard:
        shards.append(shard)
    if not shards or len(shards) > 99999:
        _fail("Invalid shard count")
    directory = Path(directory)
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (directory, *directory.parents)):
        _fail("Linked tensor destination is forbidden")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        _fail("Tensor destination must be empty private staging storage")
    files, weight_map = [], {}
    for i, names in enumerate(shards, 1):
        filename = (f"{stem}.safetensors" if len(shards) == 1 else
                    f"{stem}-{i:05d}-of-{len(shards):05d}.safetensors")
        staged = {name: _tree_copy(tensors[name]) for name in names}
        if before_write is not None:
            before_write(directory / filename)
        save_file(staged, str(directory / filename))
        del staged
        files.append(directory / filename)
        weight_map.update({name: filename for name in names})
    if len(shards) > 1:
        index = directory / f"{stem}.safetensors.index.json"
        if before_write is not None:
            before_write(index)
        with index.open("x", encoding="utf-8") as stream:
            json.dump({"metadata": {}, "weight_map": weight_map}, stream, sort_keys=True)
        files.append(index)
    return files


def _restore_tensors(directory, tensors, stem, allowed_extra=(), *, validate_only=False):
    import torch
    from safetensors import safe_open
    _check_tensor_targets(tensors)
    directory = Path(directory)
    if (not directory.is_dir() or any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                                     for p in (directory, *directory.parents))):
        _fail("Invalid/linked tensor directory")
    files = set()
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file():
            _fail("Invalid tensor file entry")
        files.add(p.name)
    paths = {directory.name + "/" + p for p in files - set(allowed_extra)}
    _weight_layout(paths, directory.name, stem, set(tensors), directory.parent)
    single = f"{stem}.safetensors"
    mapping = ({name: single for name in tensors} if single in files else
               _read_json(directory / f"{single}.index.json")["weight_map"])
    grouped = {}
    for name, filename in mapping.items():
        grouped.setdefault(filename, []).append(name)
    # Entire on-disk tensor schema is checked BEFORE copying any live value.
    # Two passes avoid keeping all shards/tensors in RAM. Hash validation by the
    # caller protects bytes; files must remain immutable across both passes.
    try:
        for apply in ((False,) if validate_only else (False, True)):
            for filename, names in grouped.items():
                with safe_open(str(directory / filename), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != set(names):
                        _fail("Shard tensor name inventory mismatch")
                    for name in names:
                        value = handle.get_tensor(name)
                        target = tensors[name]
                        _check_tensor(value, target.shape, target.dtype, name)
                        if apply:
                            with torch.no_grad():
                                target.copy_(value)
                        del value
    except CheckpointValidationError:
        raise
    except Exception as error:
        raise CheckpointValidationError(f"Tensor restoration failed: {error}") from error


def write_model_state(directory, model, *, max_shard_bytes=256 * 1024**2, before_write=None):
    """One raw model copy on disk, bounded CPU shards, no inference export."""
    # Diffusers config is a FrozenDict/Mapping, not a Transformers PretrainedConfig.
    config = getattr(model, "config", {})
    config = dict(config) if isinstance(config, Mapping) else config.to_dict()
    config = dict(config, _class_name=type(model).__name__)
    encoded_config = json.dumps(config, allow_nan=False, sort_keys=True)
    files = _write_tensors(directory, _model_tensors(model), "diffusion_pytorch_model", max_shard_bytes,
                           before_write=before_write)
    path = Path(directory) / "config.json"
    if before_write is not None:
        before_write(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded_config)
    return files + [path]


def restore_model_state(directory, model, *, expected_inventory=None):
    """In-place copy preserves Parameters; validate manifest separately first."""
    if expected_inventory is not None and _plain(expected_inventory) != tensor_inventory(model):
        _fail("Model tensor inventory differs from manifest")
    config = _read_json(Path(directory) / "config.json")
    if not isinstance(config, dict) or config.get("_class_name") != type(model).__name__:
        _fail("Model config class mismatch")
    _restore_tensors(directory, _model_tensors(model), "diffusion_pytorch_model", ("config.json",))


def _ema_tensors(ema, model, decay):
    if ema.decay != decay or getattr(ema, "backup", {}):
        _fail("EMA decay mismatch or temporary weight replacement is active")
    parameters = dict(model.named_parameters())
    if set(ema.shadow) != set(parameters):
        _fail("EMA must cover all model parameters")
    for name, value in ema.shadow.items():
        if value.shape != parameters[name].shape or value.dtype != parameters[name].dtype:
            _fail("EMA shape/dtype mismatch")
    return ema.shadow


def write_ema_state(directory, ema, model, *, decay, max_shard_bytes=256 * 1024**2, before_write=None):
    """Read existing shadows directly; never call ema.apply or replace weights."""
    return _write_tensors(directory, _ema_tensors(ema, model, decay), "weights", max_shard_bytes,
                           before_write=before_write)


def restore_ema_state(directory, ema, model, *, decay):
    _restore_tensors(directory, _ema_tensors(ema, model, decay), "weights")


def read_recovery_states(checkpoint, *, model, optimizer, scheduler):
    """Production read/validation barrier, no mutation of training objects.

    Requires a structurally validated canonical checkpoint and constructed native
    objects. Rechecks disk integrity and object compatibility; enforces CUDA RNG
    count from the production manifest. Tiny CPU codec tests use the individual
    APIs instead, without relaxing production class/device validation.

    Caller must perform environment/configuration compatibility checks first.
    After this barrier: model, optimizer, scheduler, EMA, data generators, then
    global RNG last. Complete iterator setup before restoring global RNG.
    Returned optimizer tensors occupy CPU RAM; no second model copy is loaded.
    """
    current = validate_checkpoint(checkpoint.root)
    if current.manifest != checkpoint.manifest:
        _fail("Checkpoint changed since integrity validation")
    m = _plain(current.manifest)
    objects = {"model_class": "diffusers." + type(model).__name__,
               "model_tensors": tensor_inventory(model),
               "optimizer_class": {"adamw": "torch.optim.AdamW", "adafactor": "transformers.Adafactor"}[_optimizer_kind(optimizer)],
               "parameter_groups": _group_names(model, optimizer),
               "scheduler_class": "torch.optim.lr_scheduler." + type(scheduler).__name__}
    validate_object_compatibility(current, objects)
    root, identity = current.root, m["checkpoint_id"]
    trainer = read_state_payload(root / "trainer.json", identity, manifest=m)
    opt = read_state_payload(root / "optimizer.pt", identity, model=model, optimizer=optimizer,
                             completed_steps=m["progress"]["completed_optimizer_steps"],
                             group_lrs=trainer["group_lrs"])
    for group in opt["state_dict"]["param_groups"]:
        for key, value in m["configuration"]["optimizer_options"].items():
            actual = group.get("initial_lr") if key == "lr" else group.get(key)
            if not _same(_freeze(value), actual):
                _fail(f"Effective optimizer option differs from manifest: {key}")
    scheduled = read_state_payload(root / "scheduler.pt", identity, scheduler=scheduler,
                                  configuration=m["configuration"], trainer_state=trainer)
    rng = read_state_payload(root / "rng.pt", identity, cuda_devices=m["configuration"]["world_size"])
    data = read_state_payload(root / "data_order.pt", identity, progress=m["progress"],
                             dataset_size=m["configuration"]["dataset_size"],
                             dataset_fingerprint=m["configuration"]["dataset_fingerprint"])
    _restore_tensors(root / "model", _model_tensors(model), "diffusion_pytorch_model",
                     ("config.json",), validate_only=True)
    if m["configuration"]["ema_enabled"]:
        _restore_tensors(root / "ema", dict(model.named_parameters()), "weights", validate_only=True)
    return {"trainer": trainer, "optimizer": opt, "scheduler": scheduled, "rng": rng, "data_order": data}


@dataclass(frozen=True)
class PublicationResult:
    outcome: PublicationOutcome
    destination: Path
    staging_root: Path | None
    disk_estimate: Mapping | None
    failed_stage: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    retention: str = "deferred"
    power_loss_durable: bool = False


class CheckpointPublicationError(RuntimeError):
    """Inspect .result.outcome: an error AFTER rename still means PUBLISHED."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"{result.outcome.value} at {result.failed_stage}: "
                         f"{result.exception_type}: {result.exception_message}; "
                         f"destination={result.destination}, staging={result.staging_root}")


def estimate_checkpoint_bytes(manifest, states, *, max_shard_bytes=256 * 1024**2):
    """Conservative additional free-space estimate, not an allocation guarantee.

    Counts model/EMA from tensor metadata (no model load); counts full backing
    storage of optimizer/RNG/data tensors, deliberately overcounting aliases.
    Allows 4x JSON metadata size, 4 KiB per state tensor, 1 MiB per possible shard,
    a second payload-sized staging/filesystem reserve, and 64 MiB fixed slack.
    Rename does not actually duplicate staging bytes. Existing checkpoints are
    already charged against disk_usage.free. Quotas, concurrent writers, delayed
    allocation and filesystem compression are not predicted or reserved.
    """
    import torch
    _integer(max_shard_bytes, "max_shard_bytes", 1)
    sizes = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8,
             "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1}
    model_bytes, ema_bytes, tensor_count = 0, 0, 0
    for item in manifest["objects"]["model_tensors"]:
        size = math.prod(item["shape"]) * sizes[item["dtype"]]
        if size > max_shard_bytes:
            _fail(f"Tensor {item['name']} exceeds shard budget")
        model_bytes += size
        if manifest["configuration"]["ema_enabled"] and item["kind"] == "parameter":
            ema_bytes += size
        tensor_count += 1
    storage = {}
    def metadata(value, bucket):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            storage[bucket] = storage.get(bucket, 0) + value.untyped_storage().nbytes()
            tensor_count += 1
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, Mapping):
            return {str(k): metadata(v, bucket) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [metadata(v, bucket) for v in value]
        return value
    meta = {key: metadata(value, key) for key, value in states.items()}
    encoded_bytes = len(json.dumps({"manifest": _plain(manifest), "states": meta}, allow_nan=False).encode("utf-8"))
    raw = model_bytes + ema_bytes + sum(storage.values())
    # At worst each model/EMA tensor occupies its own shard.
    possible_shards = len(manifest["objects"]["model_tensors"]) * (2 if ema_bytes else 1)
    overhead = 4 * encoded_bytes + 4096 * tensor_count + 1024**2 * possible_shards
    return {"model_bytes": model_bytes, "ema_bytes": ema_bytes,
            "optimizer_bytes": storage.get("optimizer", 0),
            "other_state_bytes": sum(v for k, v in storage.items() if k != "optimizer"),
            "metadata_filesystem_allowance_bytes": overhead,
            "staging_reserve_bytes": raw, "fixed_slack_bytes": 64 * 1024**2,
            "required_free_bytes": 2 * raw + overhead + 64 * 1024**2}


def _publication_contexts(manifest, states, model, optimizer, scheduler, ema):
    _schema(manifest)
    if manifest["payloads"] != []:
        _fail("Publication metadata must have an empty payload inventory; publisher computes it")
    _keys(states, {"trainer", "optimizer", "scheduler", "rng", "data_order"}, "publication states")
    objects = {"model_class": "diffusers." + type(model).__name__,
               "model_tensors": tensor_inventory(model),
               "optimizer_class": {"adamw": "torch.optim.AdamW", "adafactor": "transformers.Adafactor"}[_optimizer_kind(optimizer)],
               "parameter_groups": _group_names(model, optimizer),
               "scheduler_class": "torch.optim.lr_scheduler." + type(scheduler).__name__}
    if not _same(_freeze(manifest["objects"]), objects):
        _fail("Publication objects differ from manifest")
    c, p, trainer = manifest["configuration"], manifest["progress"], states["trainer"]
    if c["ema_enabled"] != (ema is not None):
        _fail("Publication EMA presence differs from manifest")
    if ema is not None:
        _ema_tensors(ema, model, c["ema_decay"])
    contexts = {
        "trainer": {"manifest": manifest},
        "optimizer": {"model": model, "optimizer": optimizer,
                      "completed_steps": p["completed_optimizer_steps"], "group_lrs": trainer["group_lrs"]},
        "scheduler": {"scheduler": scheduler, "configuration": c, "trainer_state": trainer},
        "rng": {"cuda_devices": c["world_size"]},
        "data_order": {"progress": p, "dataset_size": c["dataset_size"], "dataset_fingerprint": c["dataset_fingerprint"]},
    }
    for name, context in contexts.items():
        filename = "trainer.json" if name == "trainer" else name + ".pt"
        validate_state_payload(filename, {"schema_version": SCHEMA_VERSION,
                               "checkpoint_id": manifest["checkpoint_id"], "state": states[name]},
                               manifest["checkpoint_id"], **context)
    for group in states["optimizer"]["state_dict"]["param_groups"]:
        for key, value in c["optimizer_options"].items():
            actual = group.get("initial_lr") if key == "lr" else group.get(key)
            if not _same(_freeze(value), actual):
                _fail(f"Publication optimizer option differs from manifest: {key}")
    return contexts


def _rename_checkpoint_no_replace(source, destination):
    """Fail closed without a native atomic no-replace operation.

    POSIX os.rename can replace an empty destination directory: checking exists
    first is insufficient. Linux needs renameat2(RENAME_NOREPLACE). Windows
    os.rename already refuses an existing destination. No copy/move fallback.
    Assumes a trusted, local filesystem and stable parent directory.
    """
    import os
    import sys
    if os.name == "nt":
        os.rename(source, destination)
    elif sys.platform == "linux":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError("libc renameat2 unavailable; atomic no-replace publication unsupported")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    else:
        raise OSError("Atomic no-replace publication unsupported on this platform")


def publish_checkpoint(destination, *, manifest, states, model, optimizer, scheduler,
                       ema=None, max_shard_bytes=256 * 1024**2, fault_injector=None):
    """Publish one explicit checkpoint at a caller-acknowledged boundary.

    Inputs are Phase 4B.2 snapshots, revalidated before I/O; manifest.payloads
    must be empty. Caller must stop mutations of all inputs/model/EMA until this
    function finishes, and supply verified environment/configuration metadata.
    Parent must already exist on a trusted local filesystem. A unique sibling
    .incomplete-* root is created with tempfile.mkdtemp private permissions.

    Every failure retains staging for diagnosis; no deletion/pruning is done.
    Before rename, CheckpointPublicationError.result.outcome is FAILED. After a
    successful rename it is PUBLISHED, even if a diagnostic hook then fails.
    Success returns PublicationResult(PUBLISHED), retention='deferred'.
    fault_injector(stage) is an optional deterministic test hook, never a writer.
    Stages include inputs, disk_preflight, staging_create, write:<relative path>,
    manifest_write, integrity_validation, semantic_validation, rename, after_rename.

    ATOMIC VISIBILITY ONLY, NOT POWER-LOSS DURABILITY. Streams close before
    rename, but this phase does not fsync. Linux durable publication would need
    fsync of every payload/manifest, model/EMA subdirectories and staging root,
    then rename and fsync of the parent directory. See:
    https://man7.org/linux/man-pages/man2/fsync.2.html
    https://man7.org/linux/man-pages/man2/rename.2.html
    No guarantee for network filesystems, process termination at syscall return,
    hostile concurrent path replacement, disk/controller power loss or quotas.
    CPU peak RAM remains unbounded/unmeasured; state rereads can coexist with
    the original full CPU optimizer snapshot. No additional model export exists.
    """
    import os
    import shutil
    import tempfile
    destination = Path(destination).absolute()
    staging, estimate, published, stage = None, None, False, "inputs"
    def event(name):
        nonlocal stage
        stage = name
        if fault_injector is not None:
            fault_injector(name)
    try:
        event("inputs")
        if any(p.startswith(".incomplete-") for p in destination.parts):
            _fail("Destination cannot be a staging path")
        if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
               for p in (destination, *destination.parents)):
            _fail("Linked publication destination/ancestor forbidden")
        if not destination.parent.is_dir():
            _fail("Publication parent must already exist")
        if os.path.lexists(destination):
            raise FileExistsError(f"Checkpoint destination already exists: {destination}")
        metadata = _plain(manifest)
        contexts = _publication_contexts(metadata, states, model, optimizer, scheduler, ema)
        event("disk_preflight")
        estimate = estimate_checkpoint_bytes(metadata, states, max_shard_bytes=max_shard_bytes)
        estimate["available_free_bytes"] = shutil.disk_usage(destination.parent).free
        if estimate["available_free_bytes"] < estimate["required_free_bytes"]:
            raise OSError("Insufficient free disk space for conservative checkpoint estimate")
        event("staging_create")
        staging = Path(tempfile.mkdtemp(prefix=f".incomplete-{destination.name}-", dir=destination.parent))
        def before_write(path):
            event("write:" + path.relative_to(staging).as_posix())
        write_model_state(staging / "model", model, max_shard_bytes=max_shard_bytes, before_write=before_write)
        for name, context in contexts.items():
            filename = "trainer.json" if name == "trainer" else name + ".pt"
            before_write(staging / filename)
            write_state_payload(staging / filename, metadata["checkpoint_id"], states[name], **context)
        if ema is not None:
            write_ema_state(staging / "ema", ema, model, decay=metadata["configuration"]["ema_decay"],
                            max_shard_bytes=max_shard_bytes, before_write=before_write)
        event("inventory")
        entries = []
        for relative in sorted(_scan(staging)):
            path = staging / relative
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    digest.update(chunk)
            entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        metadata["payloads"] = entries
        event("manifest_write")
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, allow_nan=False, sort_keys=True)
        event("integrity_validation")
        _validate_checkpoint_contents(staging)
        event("semantic_validation")
        for name, context in contexts.items():
            filename = "trainer.json" if name == "trainer" else name + ".pt"
            read_state_payload(staging / filename, metadata["checkpoint_id"], **context)
        _restore_tensors(staging / "model", _model_tensors(model), "diffusion_pytorch_model",
                         ("config.json",), validate_only=True)
        if ema is not None:
            _restore_tensors(staging / "ema", _ema_tensors(ema, model, metadata["configuration"]["ema_decay"]),
                             "weights", validate_only=True)
        result = PublicationResult(PublicationOutcome.PUBLISHED, destination, None, _freeze(estimate))
        event("rename")
        _rename_checkpoint_no_replace(staging, destination)
        published = True
        # Optional diagnostics must never relabel a committed checkpoint FAILED.
        event("after_rename")
        return result
    except Exception as error:
        result = PublicationResult(PublicationOutcome.PUBLISHED if published else PublicationOutcome.FAILED,
                                   destination, None if published else staging,
                                   _freeze(estimate) if estimate is not None else None,
                                   stage, type(error).__name__, str(error))
        raise CheckpointPublicationError(result) from error
