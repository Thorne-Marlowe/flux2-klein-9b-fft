# Deterministic Base 9B recovery qualification — 2026-09-23

The supplied experiment reports exact final-state recovery for the configuration
below: an uninterrupted four-attempt control versus explicit recovery from the
completed-attempt-2 checkpoint in a new Python process, continuing through
attempt 4. This is a scoped empirical result, not a general reproducibility or
production-training guarantee. Ordinary nondeterministic exact reproducibility
remains unqualified. The earlier nondeterministic mismatch is not explained by
this result; BF16, Adafactor and restoration have not been identified as its cause.

## Preserved evidence and verification levels

- [Original evidence archive](evidence/2026-09-23/recovery-qualification-evidence.tar.gz), preserved byte-for-byte (1,949,665 bytes).
- SHA-256: `c889f28adf4a917ff5240eebc4af38cfe3cdbe66236538a13b810f1aa35da35d`.
- [Machine-readable inventory](evidence/2026-09-23/inventory.json) records member hashes, environment, effective settings and model/data/source fingerprints.
- Tested commit: `875a32f61726d332d886a9729a86b3ee2d363919`; recorded Git status is clean.

The archived result reports all **42 final model safetensors shards and their
index SHA-256 identical**, plus exact semantic equality of `optimizer.pt`,
`scheduler.pt`, `rng.pt` and `data_order.pt`. Raw checkpoint files differ in
provenance and serialization; byte equality of entire checkpoint archives is
not the acceptance condition.

Repository review independently checked matching shard/index hash inventories,
their consistency with both manifests, the actual attached manifest/trainer/
model-config hashes, current schema acceptance and identical trainer state.
All five recorded recovery-source file hashes match Git blobs at the tested
commit. The common first-two-attempt trace records match apart from the intended
`stop_after` value. Final progress is four completed attempts, zero skips,
`epoch=2`, `next_batch_index=0`, with identical loss accumulator and final LR.

The archive does **not** contain checkpoint tensor payloads, the original
semantic-comparison program/output, checkpoint-2's manifest, the resumed-phase
trace, a full launch transcript, or the image/model inputs. The semantic state
match and fresh-process restart are retained as supplied experimental evidence,
not independently re-executed claims. The new comparison utility below cannot
recompute those missing binary comparisons from this evidence archive alone.

## Recorded scope

| Setting | Recorded value |
|---|---|
| Model / precision | FLUX.2 Klein Base 9B, full transformer weights, BF16 |
| GPU | One NVIDIA A100-SXM4-80GB (compute capability 8.0) |
| Runtime | Linux x86-64, Python 3.12.3, PyTorch 2.10.0+cu128, CUDA 12.8 |
| Libraries | Accelerate 1.12.0, Diffusers 0.37.0, Transformers 5.3.0, torchvision 0.25.0+cu128, safetensors 0.7.0 |
| Data | Two fingerprinted image-caption pairs, uncached fixed size 256 |
| Batch / accumulation / workers | 1 / 1 / 0 |
| Seed / attempts / checkpoint interval | 42 / 4 / 2 |
| Optimizer | Adafactor; LR 3e-5, effective weight decay 0, relative_step=False, scale_parameter=False, warmup_init=False |
| Schedule | Zero warmup; existing cosine schedule; final LR 3e-6 |
| Other settings | Gradient checkpointing enabled; max gradient norm 1; EMA disabled; no sampling or trackers |
| Determinism | Algorithms enabled, warn_only=False, cuDNN benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8 |

The inventory preserves the remaining backend flags and optimizer options; no
TF32, SDPA or reduced-precision settings should be inferred or silently changed.
The saved model config's `_diffusers_version=0.37.0.dev0` is model metadata, not
the installed version, which the runtime metadata records as 0.37.0.

### What the data boundary establishes

`AcknowledgedData.acknowledge()` increments the cursor once per consumed attempt
and normalizes it to zero while incrementing the epoch after `len(dataset)`
items. The phase-1 trace records `epoch=1, next_batch_index=0` after attempt 2.
With two pairs and batch size 1 this is technically the trainer's normalized
epoch boundary. The result is named **explicit recovery from completed-attempt-2**
to avoid implying that GPU recovery at arbitrary mid-epoch boundaries was tested.

### Why the standalone environment probe says False

The original `environment.txt` is preserved, including
`Deterministic algorithms: False`. PyTorch's deterministic-algorithm flag is
process-local; exporting `CUBLAS_WORKSPACE_CONFIG` does not enable it. A separate
Python environment probe that does not call the recovery entry point therefore
can report False while the training processes report True. The recovery entry
point calls `torch.use_deterministic_algorithms(True, warn_only=False)` before
Accelerator/model initialization. Both seeded traces and final manifests record
that effective True setting, cuDNN benchmarking disabled and the required
workspace value. The original probe command/timing is absent, so this explains
the process-local distinction without inventing its collection history.

## Reproduce the protocol (explicit, manual GPU execution)

These are reconstructed commands from recorded effective settings, not an
archived command transcript. No tool in this change launches them. Use the
tested commit for a historical reproduction; the documentation/comparison-only
update does not modify the five fingerprinted training/recovery sources.
Provision dependencies separately, keep source/model/data immutable, and use
fresh distinct output directories on one visible GPU. Preserve all three launch
commands, process exits and logs in future evidence. Keep the horizon at four
attempts on resume.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
common=(--recovery --deterministic_recovery --recovery_model_variant base-9b
  --model_path /LOCAL/FLUX.2-klein-base-9B --data_dir /LOCAL/two-pairs
  --target_size 256 --optimizer adafactor --no_ema
  --batch_size 1 --grad_accum 1 --num_workers 0 --seed 42
  --steps 4 --warmup_steps 0 --lr 3e-5 --max_grad_norm 1
  --gradient_checkpointing --save_every 2 --log_every 1 --sample_prompts)

# Control: one fresh process, attempts 1–4.
python scripts/train_klein_standalone.py "${common[@]}" \
  --output_dir /LOCAL/control \
  --determinism_trace /LOCAL/control-trace.jsonl --determinism_trace_steps 4

# Trial phase 1: process exits after publishing checkpoint-2.
python scripts/train_klein_standalone.py "${common[@]}" \
  --output_dir /LOCAL/trial --recovery_stop_after 2 \
  --determinism_trace /LOCAL/trial-phase1-trace.jsonl --determinism_trace_steps 4

# Trial phase 2: NEW Python process; trace flags deliberately omitted on resume.
python scripts/train_klein_standalone.py "${common[@]}" \
  --output_dir /LOCAL/trial --recovery_resume /LOCAL/trial/checkpoint-2

# CPU-only comparison, AFTER both runs finish; requires full checkpoint roots.
python -B scripts/compare_recovery_checkpoints.py \
  /LOCAL/control/checkpoint-4 /LOCAL/trial/checkpoint-4
```

## Comparison contract

The utility is read-only and uses the existing v2 integrity validator to check
every payload's size/hash and the canonical inventory before interpreting it.
It compares model/EMA shards and indexes by verified whole-file SHA-256, without
loading a model. Binary state uses `torch.load(weights_only=True,
map_location='cpu')` with no arbitrary-pickle fallback. Typed recursive state
comparison checks every tensor's dtype, shape and values with no tolerance;
it is not a sampled probe. Floats/tensors containing nonfinite values fail.
Numerically equal signed zeros compare equal in semantic state; model-file
comparison remains byte-hash equality. A different shard layout is a difference,
not automatically reassembled into another serialization.

Only these differences are excluded from cross-run equality:

- Manifest top-level `checkpoint_id`, `run_id`, `created_at`, `parent_checkpoint_id`.
- State-envelope `checkpoint_id`, **after** checking it against that file's own manifest; envelope versions must match the codec.
- Model config's top-level `_name_or_path`; all other config keys are compared.
- Serialization sizes/hashes of semantically compared files may differ, but each is independently verified against its own manifest. No payload is skipped.

Exclusions that actually differ are named in the JSON report. No recursive
ID/path/hash removal is allowed. Metadata/configuration/environment/progress and
parameter-group identity must match. Trainer, optimizer inventories/LRs,
scheduler counters and acknowledged data state receive structural cross-checks.
CUDA RNG byte tensors are compared on CPU; the utility does not validate them
against a live GPU generator or replace the native-object validation done during
restoration. It does not construct a model, scheduler or optimizer.

Exit codes: `0` = `exact_match`, `1` = `different` with a first field/file path,
`2` = `invalid`/missing/corrupted/unsupported input. `strict_determinism_recorded`
reports effective recorded settings, not automatic qualification. Full state
equality cannot prove a process restart. Passing CPU comparator tests cannot
replace the GPU experiment. Matching evidence is not proof of weight provenance.

Hashing scans both complete checkpoints. State files are loaded one pair at a
time on CPU; optimizer states are not sharded and their peak RAM is not bounded
by the model-shard budget. No full-model tensors are allocated. Compare immutable
checkpoint directories; concurrent writers are unsupported. Output is JSON on
stdout; the tool never overwrites or rewrites a checkpoint/report.

## Boundaries and tests

The schema's literal `qualification='unqualified'` and existing trainer messages
remain unchanged: checkpoints do not self-certify new runs. This external record
qualifies only the supplied tested configuration. Other hardware/software,
optimizers, EMA, warmup, skipped updates, mid-epoch GPU resume, nondeterministic
exact reproducibility, minimum hardware and longer production runs remain
unqualified. No retention, staging, performance or architecture work is included.

CPU tests cover tiny real checkpoint publication, provenance-only differences,
exact state/model mismatches, envelope identity, corruption/missing files,
configuration changes, unsafe pickle rejection, CPU-only operation and archive
preservation. Existing recovery tests continue to cover native optimizer and
fresh-process continuity independently of this historical qualification.

```bash
python -B -m unittest discover -s tests -p test_checkpoint_comparison.py -v
python -B -m unittest discover -s tests -q
```
