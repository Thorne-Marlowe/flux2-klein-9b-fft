# Fresh-run determinism diagnostic

This opt-in sidecar helps locate the **first observed difference** between two
fresh uninterrupted recovery runs. It does not fix nondeterminism, qualify exact
recovery, or change deterministic-algorithm/backend settings.

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
