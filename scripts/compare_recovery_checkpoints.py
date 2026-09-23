"""Read-only, CPU-only exact comparison of two explicit v2 checkpoint roots.

This compares evidence; it does not run training, restore state, prove process
termination, or automatically qualify a configuration. No arbitrary pickle
fallback, checkpoint rewriting, model construction or CUDA calls are used.
"""
import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from scripts import klein_checkpoint as ck


PROVENANCE_FIELDS = ("checkpoint_id", "run_id", "created_at", "parent_checkpoint_id")
STATE_KEYS = {
    "trainer.json": {"progress", "global_step", "ema_updates", "cosine_updates", "group_lrs",
                     "accelerator_step", "loss_accumulator", "log_steps"},
    "optimizer.pt": {"state_dict", "parameter_groups", "tensor_metadata", "step_counters"},
    "scheduler.pt": {"state_dict", "cosine_updates", "group_lrs"},
    "rng.pt": {"python", "numpy", "torch_cpu", "torch_cuda"},
    "data_order.pt": {"epoch", "next_batch_index", "permutation_epoch", "permutation",
                      "dataset_fingerprint", "order_generator_state", "loader_generator_state"},
}


def _keys(value, expected, path):
    if not isinstance(value, dict) or value.keys() != set(expected):
        raise ck.CheckpointValidationError(f"Invalid keys at {path}")


def _tree(value, path):
    """Reject unsupported/nonfinite contents, including identical malformed pairs."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
            raise ck.CheckpointValidationError(f"Unsupported tensor at {path}")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ck.CheckpointValidationError(f"Nonfinite tensor at {path}")
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ck.CheckpointValidationError(f"Unsupported mapping key at {path}")
            _tree(item, f"{path}/{key}")
    elif type(value) in (list, tuple):
        for i, item in enumerate(value):
            _tree(item, f"{path}/{i}")
    elif value is None or type(value) in (str, int, bool):
        pass
    elif type(value) is float and math.isfinite(value):
        pass
    else:
        raise ck.CheckpointValidationError(f"Unsupported value at {path}")


def first_difference(left, right, path):
    """Exact typed equality (no tolerances), with a stable first mismatch path."""
    if type(left) is not type(right):
        return path + "/type"
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype:
            return path + "/dtype"
        if left.shape != right.shape:
            return path + "/shape"
        return None if torch.equal(left, right) else path + "/values"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + "/keys"
        for key in sorted(left, key=lambda k: (type(k).__name__, str(k))):
            difference = first_difference(left[key], right[key], f"{path}/{key}")
            if difference:
                return difference
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return path + "/length"
        for i, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}/{i}")
            if difference:
                return difference
    elif left != right:
        return path
    return None


def _load_state(root, name, manifest, trainer=None):
    value = (ck._read_json(root / name) if name.endswith(".json") else
             torch.load(root / name, map_location="cpu", weights_only=True))
    _keys(value, {"schema_version", "checkpoint_id", "state"}, name)
    if type(value["schema_version"]) is not int or value["schema_version"] != ck.SCHEMA_VERSION:
        raise ck.CheckpointValidationError(f"Unsupported state envelope version: {name}")
    if value["checkpoint_id"] != manifest["checkpoint_id"]:
        raise ck.CheckpointValidationError(f"Envelope identity disagrees with its own manifest: {name}")
    state = value["state"]
    _keys(state, STATE_KEYS[name], name + "/state")
    _tree(state, name + "/state")
    if name == "trainer.json":
        ck.validate_trainer_state(state, manifest=manifest)
    elif name == "data_order.pt":
        ck.validate_data_order_state(state, progress=manifest["progress"],
            dataset_size=manifest["configuration"]["dataset_size"],
            dataset_fingerprint=manifest["configuration"]["dataset_fingerprint"])
    elif name == "scheduler.pt":
        if (not isinstance(state["state_dict"], dict) or
                state["cosine_updates"] != trainer["cosine_updates"] or
                state["group_lrs"] != trainer["group_lrs"] or
                state["state_dict"].get("last_epoch") != trainer["cosine_updates"]):
            raise ck.CheckpointValidationError("Scheduler counters/LRs disagree with trainer")
    elif name == "optimizer.pt":
        groups = manifest["objects"]["parameter_groups"]
        native = state["state_dict"]
        _keys(native, {"state", "param_groups"}, "optimizer native state")
        if state["parameter_groups"] != groups or len(native["param_groups"]) != len(groups):
            raise ck.CheckpointValidationError("Optimizer group association mismatch")
        ids = []
        for i, (group, names) in enumerate(zip(native["param_groups"], groups)):
            if len(group["params"]) != len(names) or group["lr"] != trainer["group_lrs"][i]:
                raise ck.CheckpointValidationError("Optimizer parameter count/LR mismatch")
            ids.extend(group["params"])
        if any(type(pid) is not int for pid in ids) or len(set(ids)) != len(ids) or not set(native["state"]) <= set(ids):
            raise ck.CheckpointValidationError("Invalid optimizer parameter IDs")
        metadata, counters = ck._optimizer_inventory(native, groups)
        if first_difference(metadata, state["tensor_metadata"], "inventory") or first_difference(counters, state["step_counters"], "counters"):
            raise ck.CheckpointValidationError("Optimizer tensor inventory/counters disagree with saved state")
    elif name == "rng.pt":
        _keys(state["numpy"], {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}, "NumPy RNG")
        if not isinstance(state["python"], tuple) or len(state["python"]) != 3:
            raise ck.CheckpointValidationError("Invalid Python RNG structure")
        if not isinstance(state["torch_cuda"], list) or len(state["torch_cuda"]) != manifest["configuration"]["world_size"]:
            raise ck.CheckpointValidationError("CUDA RNG inventory mismatch")
        for tensor in [state["torch_cpu"], *state["torch_cuda"]]:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
                raise ck.CheckpointValidationError("Invalid RNG byte tensor")
        # Do not call validate_rng_state: it requires the original CUDA runtime.
    return state


def compare_checkpoints(control, resumed):
    report = {"format": "klein-checkpoint-comparison-v1", "status": "invalid",
              "scope": "Complete model/EMA file SHA-256 and exact typed recoverable-state comparison; not process-restart proof or automatic qualification.",
              "provenance_differences": [], "compared_payloads": []}
    stage = "checkpoint integrity"
    try:
        checkpoints = []
        for label, root in (("control", control), ("resumed", resumed)):
            stage = label + "/integrity"
            checkpoints.append(ck.validate_checkpoint(root))
        stage = "manifest compatibility"
        manifests = [ck._plain(c.manifest) for c in checkpoints]
        if any(m["schema_version"] != 2 for m in manifests):
            raise ck.CheckpointValidationError("Comparison supports manifest v2 only; no v1 migration")
        a, b = manifests
        report["strict_determinism_recorded"] = all(
            m["environment"]["backend_settings"]["deterministic_algorithms"] and
            not m["environment"]["backend_settings"]["deterministic_warn_only"] and
            not m["environment"]["backend_settings"]["cudnn_benchmark"] and
            m["environment"]["backend_settings"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            for m in manifests)
        report["provenance_differences"] = ["manifest/" + key for key in PROVENANCE_FIELDS if a[key] != b[key]]
        def different(path):
            report.update(status="different", first_difference=path)
            return report
        # Ignore only named top-level provenance. Payload hashes are verified
        # independently above, then handled by file/semantic policy below.
        ignored = set(PROVENANCE_FIELDS) | {"payloads"}
        diff = first_difference({k: v for k, v in a.items() if k not in ignored},
                                {k: v for k, v in b.items() if k not in ignored}, "manifest")
        if diff:
            return different(diff)
        inventories = [{p["path"]: p for p in m["payloads"]} for m in manifests]
        if inventories[0].keys() != inventories[1].keys():
            return different("payload_inventory")
        for name in sorted(inventories[0]):
            if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
                if inventories[0][name]["sha256"] != inventories[1][name]["sha256"]:
                    return different(name + "/sha256")
                report["compared_payloads"].append({"path": name, "comparison": "complete_file_sha256"})
        stage = "model/config.json"
        configs = [ck._read_json(c.root / "model/config.json") for c in checkpoints]
        for config in configs:
            if "_name_or_path" in config and not isinstance(config["_name_or_path"], str):
                raise ck.CheckpointValidationError("Invalid model/config.json _name_or_path")
        if configs[0].get("_name_or_path") != configs[1].get("_name_or_path"):
            report["provenance_differences"].append("model/config.json/_name_or_path")
        diff = first_difference({k: v for k, v in configs[0].items() if k != "_name_or_path"},
                                {k: v for k, v in configs[1].items() if k != "_name_or_path"}, "model/config.json")
        if diff:
            return different(diff)
        report["compared_payloads"].append({"path": "model/config.json", "comparison": "exact_except_top_level_name_or_path"})
        stage = "trainer.json"
        trainers = [_load_state(c.root, "trainer.json", m) for c, m in zip(checkpoints, manifests)]
        for name in STATE_KEYS:
            stage = name
            states = trainers if name == "trainer.json" else [
                _load_state(c.root, name, m, trainer) for c, m, trainer in zip(checkpoints, manifests, trainers)]
            diff = first_difference(states[0], states[1], name + "/state")
            if diff:
                return different(diff)
            if a["checkpoint_id"] != b["checkpoint_id"]:
                report["provenance_differences"].append(name + "/checkpoint_id")
            report["compared_payloads"].append({"path": name, "comparison": "exact_typed_state"})
            del states  # Retain at most one pair of binary state payloads.
        report["status"] = "exact_match"
    except Exception as exc:
        report.update(status="invalid", error_type=type(exc).__name__,
                      validation_stage=stage,
                      error="Checkpoint integrity, envelope or comparison validation failed; no match established.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("resumed", type=Path)
    args = parser.parse_args(argv)
    result = compare_checkpoints(args.control, args.resumed)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return {"exact_match": 0, "different": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
