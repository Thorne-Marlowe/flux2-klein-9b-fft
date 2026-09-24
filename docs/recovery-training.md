# Opt-in recovery training (configuration-specific qualification)

The [Milestone 3 qualification record](qualification/milestone-3-mid-epoch.md)
records the completed single-A100/Base 9B/Adafactor seven-attempt mid-epoch and
cross-epoch recovery result. The earlier [four-attempt record](qualification/deterministic-bf16-a100-2026-09-23.md)
is preserved as historical evidence. Ordinary nondeterministic exact
reproducibility remains unqualified.
Checkpoint manifests and existing trainer messages retain the conservative
`unqualified` label; a checkpoint does not self-certify a new experiment.

`--recovery` uses manifest v2 and the existing version-1 payload codecs. Legacy
training and `--smoke_test` retain their separate behavior. V1 checkpoints remain
readable by the integrity validator, but the recovery trainer rejects them:
execution/source metadata cannot be inferred. There is no migration, latest
discovery, retention, sampling recovery or tracker recovery.

The candidate configuration is one visible CUDA GPU, BF16, full weights, batch
and accumulation 1, workers 0, uncached fixed-size images, AdamW or Adafactor,
and optional EMA. Only the configuration in the linked record is qualified by
the supplied GPU experiment; EMA and other candidate settings are not. CPU fixture
tests cannot establish CUDA, actual Flux, or prepared-Accelerate behavior on the
target machine. Keep the original model, dataset, software, code and settings
immutable between processes. Content fingerprints are relocation-independent;
dependency/backend/configuration changes fail strict compatibility checks.

## Controlled four-attempt trial (strict deterministic profile)

Run on an already provisioned environment; these commands install/download
nothing. Replace all paths. Use the two recorded image-caption pairs (or treat
different data as a new experiment) and
an empty writable output directory. Keep `--steps 4` unchanged on resume.
`--recovery_stop_after` is an absolute stop boundary, not a new LR horizon.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/train_klein_standalone.py \
  --recovery --deterministic_recovery --recovery_model_variant base-9b \
  --model_path /LOCAL/FLUX.2-klein-base-9B \
  --data_dir /LOCAL/image-caption-pairs --output_dir /LOCAL/recovery-trial \
  --target_size 256 --optimizer adafactor --no_ema \
  --batch_size 1 --grad_accum 1 --num_workers 0 \
  --steps 4 --warmup_steps 0 --save_every 2 --log_every 1 \
  --seed 42 --sample_prompts --recovery_stop_after 2

python scripts/train_klein_standalone.py \
  --recovery --deterministic_recovery --recovery_model_variant base-9b \
  --model_path /LOCAL/FLUX.2-klein-base-9B \
  --data_dir /LOCAL/image-caption-pairs --output_dir /LOCAL/recovery-trial \
  --target_size 256 --optimizer adafactor --no_ema \
  --batch_size 1 --grad_accum 1 --num_workers 0 \
  --steps 4 --warmup_steps 0 --save_every 2 --log_every 1 \
  --seed 42 --sample_prompts \
  --recovery_resume /LOCAL/recovery-trial/checkpoint-2
```

Also run the first command without `--recovery_stop_after`, using a different
empty output directory, as the uninterrupted control. Compare attempts 3–4,
parameter/buffer tensors, optimizer tensors and their dtypes, native scheduler
state and actual group LRs, RNG, acknowledged data position, and EMA in a
separate EMA-enabled trial. Checkpoint IDs/timestamps naturally differ. The
four-attempt trial has no warmup (the trainer caps warmup at steps // 10); CPU
continuity tests separately cover manual warmup and simulated skips. Verify
actual Accelerate skips separately before broader qualification.

Compare the complete final checkpoints without loading a model or using CUDA:

```bash
python -B scripts/compare_recovery_checkpoints.py \
  /LOCAL/recovery-control/checkpoint-4 /LOCAL/recovery-trial/checkpoint-4
```

Exit 0 establishes exact comparison under the utility's documented provenance
policy, not proof that the processes followed the protocol. Preserve launch logs.
See the qualification record for the comparison contract and evidence gaps.

## Verify a published transformer artifact

The recovery checkpoint's `model/` directory is independently consumable by
Diffusers. Run this CPU-only structural check against an already published
checkpoint; it loads the public `Flux2Transformer2DModel.from_pretrained` path,
checks every serialized tensor's names, shapes, dtypes and values, and never
rewrites the artifact:

```bash
python -B scripts/verify_inference_artifact.py \
  --artifact /LOCAL/recovery-trial/checkpoint-4/model
```

This proves loader compatibility and serialization integrity for the artifact;
it does not prove Base 9B inference or training quality. The check may require
RAM comparable to the transformer because the public loader constructs a model;
tensor comparison itself reads safetensor shards one at a time.

For a later inference-only GPU qualification, use the original frozen Base 9B
directory separately from the trained transformer. Do not run this command
until both directories already exist:

```bash
python -B scripts/verify_inference_artifact.py \
  --artifact /LOCAL/recovery-trial/checkpoint-4/model \
  --model-path /LOCAL/FLUX.2-klein-base-9B \
  --prompt "a small red house in sunlight" \
  --steps 2 --height 256 --width 256 \
  --output /LOCAL/recovery-trial/artifact-inference.png
```

Successful GPU execution establishes that this trained transformer can be
inserted into the normal Klein pipeline and produce a structurally valid image
on that environment. It does not establish image quality, minimum hardware,
long-run reliability or a broader training qualification.

## Boundaries and limitations

- Recovery Adafactor retains its existing **effective weight decay of zero**.
  A nonzero `--weight_decay` (including the shared CLI default of 0.01) now
  produces a warning because it is not applied. Specify `--weight_decay 0`
  explicitly for Adafactor. AdamW applies the requested value. This clarification
  does not change optimizer configuration or historical qualification results.
- Effective warmup is `min(warmup_steps, steps // 10)`. The first attempt uses
  `--lr`; after zero-based attempt `a` during warmup, the next LR is assigned
  `lr * (a + 1) / effective_warmup`, including skipped attempts. After warmup,
  native CosineAnnealingLR advances only on completed updates, with
  `T_max = steps - effective_warmup` and `eta_min = lr * 0.1`. There is no
  warmup division when effective warmup is zero. Resume restores native scheduler
  state and actual group LRs without replaying warmup. This describes the existing
  after-attempt policy; it is not conventional before-update linear warmup.
- A nonfinite loss or clipping norm aborts before the optimizer update and
  acknowledgement. The norm check reuses the existing clipping reduction;
  failed clipping may mutate gradients, but optimizer, scheduler, EMA, acknowledged
  cursor and completed-update count do not advance, and no checkpoint is published
  for that attempt. Previously committed checkpoints remain available. Reading
  the batch and backward may already have consumed RNG; this is a fatal failure,
  not an in-process retry or RNG rollback.

- The resolver selects standard safetensors, built-in Klein/Qwen classes and
  the fast tokenizer JSON. It rejects ambiguous weights, missing shards, custom
  code, quantization and unsupported tokenizer assets. Slow vocab/merges are
  excluded explicitly. A private local view exposes only the selected files to
  component loaders; weights are hard-linked, metadata copied. This needs a
  writable same-filesystem parent and hard-link support. No huge-copy fallback.
  The Base 9B selector is a user declaration, not proof of weight provenance.
- Recovery encoding moves the transformer to CPU, stages VAE then text encoder
  on CUDA and back, then returns the same transformer parameters to CUDA.
  Optimizer state and EMA remain resident. No latent cache is created. This
  trades transfer time/CPU RAM for reduced simultaneous GPU model residency;
  the smoke test's measured memory does not predict this path's peak.
- A native workers=0 DataLoader uses independent permutation and loader RNGs.
  It is intentionally not Accelerate-wrapped: no sharding/lookahead is needed.
  Only post-update acknowledgement advances the saved cursor. Skips consume
  data and update EMA; cosine advances only on completed updates. Manual warmup
  assignments still occur after attempts, including skips.
- Resume validates integrity and available identities before model allocation,
  checks constructed metadata, prepares model/optimizer, restores tensors in
  place and pristine optimizer dtypes, restores scheduler/EMA/counters/data,
  initializes the iterator, then restores global RNG last.
- Publication failure aborts training. Existing checkpoints are never replaced.
  Failed staging is retained for diagnosis and rejected as a resume root.
  Manifest is last, then an atomic no-replace directory rename. This provides
  atomic visibility on supported local filesystems, **not power-loss durability**.
  Linux/network-volume semantics and resource peaks still need qualification.
- Hashing scans all selected model/data/source bytes; resume also verifies all
  checkpoint payload bytes. No hash cache. Model/EMA staging is shard-bounded,
  optimizer capture is a full CPU copy, and total peak CPU RAM is not bounded.
  Disk preflight includes conservative staging/filesystem reserve; it cannot
  predict quotas, concurrent writes or delayed allocation. No fit guarantee.
