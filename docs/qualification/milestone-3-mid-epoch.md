# Milestone 3 qualification plan: genuine mid-epoch recovery

This is a preparation document, not a qualification result. The real Base 9B
experiment must run on one A100 with the exact configuration below before any
claim is made.

The dataset contains exactly **three** image-caption pairs, sorted by filename.
With batch size 1 and workers 0, each acknowledged attempt consumes one pair.
The interrupted run stops after attempt 1 and publishes `checkpoint-1`: its
progress is `epoch=0`, `next_batch_index=1`, so the checkpoint is unambiguously
inside the first three-sample epoch. A seven-attempt continuation completes the
remaining two samples, all three samples of epoch 1, and one sample of epoch 2.
The expected final progress is `attempts=7`, `completed=7`, `skipped=0`,
`epoch=2`, `next_batch_index=1`.

The future Runpod session should first establish the environment and model/data
identity, then run the separate Milestone 2B inference-only check against an
existing qualified checkpoint. Preserve that result before beginning this
qualification. Milestone 2B and this recovery qualification remain separate
claims.

Set the paths and launch environment:

```bash
export MODEL=/LOCAL/FLUX.2-klein-base-9B
export DATA=/LOCAL/milestone-3-data-3-pairs
export RUNS=/LOCAL/milestone-3-runs
export EVIDENCE=/LOCAL/milestone-3-evidence
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$RUNS" "$EVIDENCE"
```

The control is a fresh run with a seven-attempt schedule and saves only its
final checkpoint:

```bash
python -B scripts/train_klein_standalone.py --recovery --deterministic_recovery \
  --recovery_model_variant base-9b --model_path "$MODEL" --data_dir "$DATA" \
  --output_dir "$RUNS/control" --target_size 256 --optimizer adafactor \
  --no_ema --batch_size 1 --grad_accum 1 --num_workers 0 --steps 7 \
  --warmup_steps 0 --lr 3e-5 --weight_decay 0 --max_grad_norm 1 \
  --save_every 7 --log_every 1 --seed 42 --sample_prompts \
  2>&1 | tee "$EVIDENCE/control.log"
```

The interrupted run uses `save_every=1` only to publish the deliberate
checkpoint-1, then stops at that absolute attempt:

```bash
python -B scripts/train_klein_standalone.py --recovery --deterministic_recovery \
  --recovery_model_variant base-9b --model_path "$MODEL" --data_dir "$DATA" \
  --output_dir "$RUNS/trial" --target_size 256 --optimizer adafactor \
  --no_ema --batch_size 1 --grad_accum 1 --num_workers 0 --steps 7 \
  --warmup_steps 0 --lr 3e-5 --weight_decay 0 --max_grad_norm 1 \
  --save_every 1 --log_every 1 --seed 42 --sample_prompts \
  --recovery_stop_after 1 2>&1 | tee "$EVIDENCE/interrupted.log"
```

Start a fresh process and identify the checkpoint explicitly. The resumed run
uses the same seven-attempt horizon and saves its final checkpoint:

```bash
python -B scripts/train_klein_standalone.py --recovery --deterministic_recovery \
  --recovery_model_variant base-9b --model_path "$MODEL" --data_dir "$DATA" \
  --output_dir "$RUNS/resumed" --target_size 256 --optimizer adafactor \
  --no_ema --batch_size 1 --grad_accum 1 --num_workers 0 --steps 7 \
  --warmup_steps 0 --lr 3e-5 --weight_decay 0 --max_grad_norm 1 \
  --save_every 7 --log_every 1 --seed 42 --sample_prompts \
  --recovery_resume "$RUNS/trial/checkpoint-1" \
  2>&1 | tee "$EVIDENCE/resumed.log"
```

Compare the final checkpoints with the existing CPU-only comparator:

```bash
python -B scripts/compare_recovery_checkpoints.py \
  "$RUNS/control/checkpoint-7" "$RUNS/resumed/checkpoint-7" \
  | tee "$EVIDENCE/comparison.json"
```

Acceptance requires exact model shard/index agreement and exact recoverable
optimizer, scheduler, RNG, data-order, trainer counters and cursor state under
the comparator's documented provenance exclusions. Logs must show the last
acknowledged attempt before interruption, the resumed next sample, finite loss
and gradients, and seven actual completed updates. Similar losses or images do
not establish success.

Preserve `git rev-parse HEAD`, clean/modified status, dependency versions,
`nvidia-smi -q`, model revision/path identity, the sorted three-pair inventory,
all three logs, checkpoint manifests and SHA-256 inventories. Do not duplicate
the large checkpoint payloads solely as evidence; retain the original
checkpoints separately. The helper below validates the dataset shape and emits
the same command plan without loading or downloading a model:

```bash
python -B scripts/prepare_recovery_qualification.py \
  --model-path "$MODEL" --data-dir "$DATA" --evidence-dir "$EVIDENCE" \
  --output-root "$RUNS" > "$EVIDENCE/plan.json"
```

This result, if successful, qualifies only deterministic BF16 Adafactor
recovery on the recorded A100/software/configuration and this three-pair,
seven-attempt experiment. It does not qualify other GPUs, ordinary
nondeterministic recovery, other optimizers, EMA, batching, worker counts,
precision modes, arbitrary dataset lengths or interruption during an in-flight
operation.
