# Fresh-run determinism diagnostic

This opt-in sidecar helps locate the **first observed difference** between two
fresh uninterrupted recovery runs. It does not fix nondeterminism, qualify exact
recovery, or change deterministic-algorithm/backend settings.

The subsequent [scoped deterministic BF16 recovery result](qualification/deterministic-bf16-a100-2026-09-23.md)
is preserved separately. That experiment does not qualify ordinary nondeterministic
exact reproducibility. Use `compare_recovery_checkpoints.py` for complete final
checkpoint comparison; this sampled trace comparator intentionally reports
configuration differences such as a trial's `stop_after` value and is not a
replacement for the checkpoint comparator.

Append to the existing recovery command (same training settings in both runs):

```text
--determinism_trace /LOCAL/run-a/trace.jsonl --determinism_trace_steps 2
```

For the second run use another output directory and trace path. Trace length is
1–4 attempts, default 2; this limit does not stop training or change the schedule.
Resume, smoke and legacy training reject the diagnostic flags. Existing trace
files are never overwritten. Do not place the trace inside a checkpoint root.
No additional model checkpoints are created; the normal save cadence is unchanged.

Compare on CPU:

```bash
python -B scripts/klein_determinism.py /LOCAL/run-a/trace.jsonl /LOCAL/run-b/trace.jsonl
```

Exit 0 means no observed difference in completed traces; exit 1 means a
difference, incomplete trace or invalid trace. Matching samples cannot establish
complete tensor equality. Inspect `status`, `stage`, `attempt` (zero-based) and
`field`. This is an observation ordering, not proof of the first causal divergence.
The tool does not compare checkpoint timestamps, run IDs or serialized file bytes.

The trace records seed/RNG state hashes, loaded tensor hashes, effective recovery
metadata, diagnostic source identity, GPU identity, data cursor/generator hashes,
sample paths and caption hashes, preprocessed pixels, VAE posterior/sample,
normalized latents, embeddings, timestep draws/noise, forward inputs/output/loss,
pre/post-clipping gradients, optimizer evidence/state, updated weights, EMA and the
acknowledged boundary. Normal saves within the observed attempts have RNG records
before/after. Restoration, checkpoint schemas and the publisher are unchanged.

Loaded CPU parameters/buffers are hashed once in full with 256 KiB chunks. This
scans all loaded components and may take time; it does not clone or save them.
Input/output observations hash complete tensors up to 64 KiB; larger tensors use
64 evenly spaced elements. Prepared weights, gradients and optimizer tensors
always use at most 64 elements each. All records have explicit `coverage`, shapes,
dtypes and indices when sampled.
Complete CPU pixel hashes also stream in chunks. Raw probe bytes are hexadecimal
in their recorded dtype; full large tensor contents are not written. GPU transfers
are batched into at most 1 MiB per batch, with small temporary gather/concatenation
buffers. Tensor inventory metadata scales with count (maximum 16,384 records per
observation), not parameter bytes. There is no full optimizer CPU snapshot for
diagnostics. Host observation necessarily waits for the transferred data, but
there are no added explicit CUDA synchronize calls or full-gradient reductions.

Diagnostics do not draw or replay random samples. Disabled diagnostics perform
no capture. CPU tests check enabled/disabled RNG, losses, gradients, optimizer
state and weights, but cannot certify CUDA neutrality or unchanged execution
timing. An exception leaves an incomplete sidecar without `trace_complete`.
No telemetry service, GPU experiment or automatic mismatch repair is involved.

Use the same code/dependencies and normal checkpoint cadence for both runs.
Source identity changes with instrumentation: generate fresh baselines; do not
relax recovery compatibility checks to reuse older checkpoints.

## Opt-in strict deterministic recovery

For the next two-fresh-run investigation, set the cuBLAS workspace environment
**before starting each Python process**, and append `--deterministic_recovery`
to both otherwise identical recovery commands, alongside the trace flags:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Append to each existing fresh --recovery command:
# --deterministic_recovery --determinism_trace /LOCAL/run-a/trace.jsonl --determinism_trace_steps 2
# Use a separate run-b output directory and trace path for the second run.
```

The flag enables `torch.use_deterministic_algorithms(True, warn_only=False)`
and disables cuDNN benchmarking before Accelerator/model initialization.
Missing or different workspace values and already-initialized CUDA contexts
are rejected; no late environment assignment or warning-only fallback occurs.
Unsupported deterministic operations raise explicitly, potentially before a
gradient observation. A failed run's trace is incomplete, not evidence of a
successful comparison. Record the error and last observation before proceeding.

The seeded trace records the opt-in and effective strict settings; the existing
configuration observation records the full backend settings (including TF32,
SDPA, attention processors and cuBLAS workspace). Precision, optimizer, RNG
draws and training schedule are unchanged. Without this flag the diagnostic
continues to observe the existing backend settings without changing them.
The flag is recovery-only; the trace still requires fresh runs. Existing strict
checkpoint compatibility checks remain unchanged. Settings are process-wide,
so use fresh processes rather than reusing an interpreter between experiments.

These settings do not guarantee exact reproducibility across devices, library
versions or all operations. Differing sampled pre-clipping gradients at attempt
0 do not establish BF16, Adafactor or restoration as the cause. This experiment
investigates the first observed difference; it does not establish a fix.
