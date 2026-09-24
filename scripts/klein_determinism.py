"""Opt-in first-attempt observations, not a numerical recovery certificate.

No RNG draws, backend changes, full-model snapshots or checkpoint writes.
CPU full hashes stream 256 KiB chunks. GPU observations transfer at most 64 KiB
per tensor, otherwise 64 fixed elements; transfers are batched within 1 MiB.
Record metadata scales with tensor count, not weight size. Host reads of CUDA
data necessarily wait for that data; no explicit device synchronization is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

CHUNK_BYTES = 256 * 1024
COMPLETE_GPU_BYTES = 64 * 1024
TRANSFER_BYTES = 1024 * 1024
PROBE_ELEMENTS = 64
MAX_TENSORS = 16384


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _flat_values(tensor, indices):
    if tensor.is_contiguous():
        return tensor.view(-1)[indices]
    # Never reshape a full noncontiguous model tensor into a copied allocation.
    return tensor[torch.unravel_index(indices, tensor.shape)]


def _cpu_hash(tensor):
    digest = hashlib.sha256()
    if tensor.is_contiguous():
        raw = tensor.reshape(-1).view(torch.uint8)
        for start in range(0, raw.numel(), CHUNK_BYTES):
            digest.update(raw[start:start + CHUNK_BYTES].numpy().tobytes())
    else:
        count = max(1, CHUNK_BYTES // (8 * (tensor.ndim + 2) + tensor.element_size()))
        for start in range(0, tensor.numel(), count):
            indices = torch.arange(start, min(start + count, tensor.numel()))
            raw = _flat_values(tensor, indices).contiguous().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def tensor_records(named, *, full_cpu=False, probe_only=False):
    """Stable sorted records. Hash equality for a probe covers ONLY its indices.

    CPU model hashing can be complete; probe_only caps model/gradient/optimizer
    records at 64 elements even for smaller tensors. No float reductions.
    """
    named = sorted(named, key=lambda pair: pair[0])
    if len(named) > MAX_TENSORS or len({name for name, _ in named}) != len(named):
        raise ValueError("Diagnostic tensor inventory exceeds limit or has duplicate names")
    records, pending, pending_bytes, device = [], [], 0, None

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        raw = torch.cat([item[1] for item in pending]).cpu().numpy().tobytes()
        offset = 0
        for record, block in pending:
            data = raw[offset:offset + block.numel()]
            record["sha256"] = _digest(data)
            if record["coverage"] == "sampled" or record["numel"] <= PROBE_ELEMENTS:
                record["observed_bytes_hex"] = data.hex()
            offset += block.numel()
        pending, pending_bytes = [], 0

    for name, value in named:
        if value is None:
            records.append({"name": name, "present": False})
            continue
        tensor = value.detach()
        if tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported diagnostic tensor: {name}")
        record = {"name": name, "present": True, "shape": list(tensor.shape),
                  "dtype": str(tensor.dtype), "device": str(tensor.device), "numel": tensor.numel()}
        records.append(record)
        if tensor.device.type == "cpu" and full_cpu:
            record.update(coverage="complete", observed_elements=tensor.numel(), sha256=_cpu_hash(tensor))
            continue
        complete = (tensor.numel() <= PROBE_ELEMENTS if probe_only else
                    tensor.numel() * tensor.element_size() <= COMPLETE_GPU_BYTES)
        if complete:
            selected = tensor.contiguous().reshape(-1)
            record.update(coverage="complete", observed_elements=tensor.numel())
        else:
            count = min(PROBE_ELEMENTS, tensor.numel())
            indices = torch.arange(count, device=tensor.device) * (tensor.numel() - 1) // max(count - 1, 1)
            selected = _flat_values(tensor, indices)
            record.update(coverage="sampled", observed_elements=count,
                          indices=[i * (tensor.numel() - 1) // max(count - 1, 1) for i in range(count)])
        block = selected.contiguous().view(torch.uint8)
        if device != tensor.device or pending_bytes + block.numel() > TRANSFER_BYTES:
            flush()
            device = tensor.device
        pending.append((record, block))
        pending_bytes += block.numel()
    flush()
    return records


def rng_record(data=None):
    import numpy as np
    n = np.random.get_state()
    result = {"python": _digest(_json(random.getstate()).encode()),
              "numpy": _digest(_json([n[0], n[1].tolist(), n[2], n[3], n[4]]).encode()),
              "torch_cpu": _cpu_hash(torch.get_rng_state()), "cuda_initialized": torch.cuda.is_initialized()}
    # Do not initialize CUDA just to observe it.
    result["torch_cuda"] = ([_cpu_hash(s) for s in torch.cuda.get_rng_state_all()]
                            if result["cuda_initialized"] else [])
    if data is not None:
        result.update(order_generator=_cpu_hash(data.order.get_state()),
                      loader_generator=_cpu_hash(data.loader_rng.get_state()),
                      permutation=_cpu_hash(data.permutation), epoch=data.epoch, cursor=data.cursor)
    return result


def model_tensors(model, gradients=False):
    if gradients:
        return [(name, parameter.grad) for name, parameter in model.named_parameters()]
    return list(model.named_parameters()) + list(model.named_buffers())


def optimizer_observation(model, optimizer):
    """Inspect native state directly; never call the full CPU snapshot codec."""
    tensors, scalars, groups = [], {}, []
    names = {id(p): name for name, p in model.named_parameters()}
    for group in optimizer.param_groups:
        groups.append({key: value for key, value in group.items() if key != "params"})
        for p in group["params"]:
            name = names[id(p)]
            state = optimizer.state.get(p, {})  # .get must not initialize state.
            scalars[name] = {"initialized": bool(state)}
            for key, value in sorted(state.items()):
                if isinstance(value, torch.Tensor):
                    tensors.append((name + "/" + key, value))
                else:
                    scalars[name][key] = value
    return {"groups": groups, "state": scalars, "tensors": tensor_records(tensors, probe_only=True)}


class DeterminismTrace:
    """Exclusive JSONL sidecar; incomplete traces are not successful comparisons."""
    def __init__(self, path, steps=2):
        if type(steps) is not int or not 1 <= steps <= 4:
            raise ValueError("Trace steps must be in [1,4]")
        self.path, self.steps, self.attempt = Path(path), steps, None
        self.sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_json({"format": "klein-determinism-v1", "steps": steps,
                               "probe_elements": PROBE_ELEMENTS, "complete_tensor_bytes": COMPLETE_GPU_BYTES,
                               "comparison_scope": "first observed difference; sampled equality is not full equality"}) + "\n")

    def emit(self, stage, *, tensors=(), details=None, full_cpu=False, probe_only=False, rng=False, data=None):
        record = {"sequence": self.sequence, "attempt": self.attempt, "stage": stage,
                  "details": details, "tensors": tensor_records(tensors, full_cpu=full_cpu, probe_only=probe_only)}
        if rng:
            record["rng"] = rng_record(data)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(record) + "\n")
        self.sequence += 1

    def finish(self):
        self.emit("trace_complete")


def _difference(a, b, path="record"):
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return path + ".keys"
        for key in sorted(a):
            found = _difference(a[key], b[key], path + "." + key)
            if found:
                return found
    elif isinstance(a, list):
        if len(a) != len(b):
            return path + ".length"
        for index, (left, right) in enumerate(zip(a, b)):
            found = _difference(left, right, f"{path}[{index}]")
            if found:
                return found
    elif a != b:
        return path
    return None


def compare_traces(left, right):
    """Streaming comparison; no checkpoint tensors are loaded."""
    from itertools import zip_longest
    complete = [False, False]
    with Path(left).open(encoding="utf-8") as a, Path(right).open(encoding="utf-8") as b:
        for index, pair in enumerate(zip_longest(a, b)):
            if None in pair:
                return {"status": "incomplete", "record": index}
            try:
                def invalid_constant(value):
                    raise ValueError(f"Nonfinite JSON literal: {value}")
                rows = [json.loads(line, parse_constant=invalid_constant) for line in pair]
                if any(not isinstance(row, dict) for row in rows):
                    raise ValueError("Record is not an object")
                if index == 0 and any(row.get("format") != "klein-determinism-v1" for row in rows):
                    raise ValueError("Unknown trace format")
                if index > 0 and any(row.get("sequence") != index - 1 for row in rows):
                    raise ValueError("Invalid trace sequence")
                if any(complete):
                    raise ValueError("Records after completion")
            except (ValueError, TypeError) as error:
                return {"status": "invalid", "record": index, "error": str(error)}
            mismatch = _difference(*rows)
            if mismatch:
                return {"status": "first_observed_difference", "record": index,
                        "stage": rows[0].get("stage", "header"), "attempt": rows[0].get("attempt"),
                        "field": mismatch, "scope": "observations only; not proof of first numerical divergence"}
            complete = [row.get("stage") == "trace_complete" for row in rows]
    return {"status": "no_observed_difference" if all(complete) else "incomplete",
            "scope": "sampled tensors are not proven identical in full"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two opt-in Klein determinism sidecars")
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    result = compare_traces(args.left, args.right)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "no_observed_difference" else 1)
