# Opt-in recovery training (unqualified)

`--recovery` uses manifest v2 and the existing version-1 payload codecs. Legacy
training and `--smoke_test` retain their separate behavior. V1 checkpoints remain
readable by the integrity validator, but the recovery trainer rejects them:
execution/source metadata cannot be inferred. There is no migration, latest
discovery, retention, sampling recovery or tracker recovery.

The candidate configuration is one visible CUDA GPU, BF16, full weights, batch
and accumulation 1, workers 0, uncached fixed-size images, AdamW or Adafactor,
and optional EMA. This path is **not qualified for exact recovery**. CPU fixture
tests cannot establish CUDA, actual Flux, or prepared-Accelerate behavior on the
target machine. Keep the original model, dataset, software, code and settings
immutable between processes. Content fingerprints are relocation-independent;
dependency/backend/configuration changes fail strict compatibility checks.

## Controlled four-attempt trial

Run on an already provisioned environment; these commands install/download
nothing. Replace all paths. Start with at least two image-caption pairs and
an empty writable output directory. Keep `--steps 4` unchanged on resume.
`--recovery_stop_after` is an absolute stop boundary, not a new LR horizon.

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train_klein_standalone.py \
  --recovery --recovery_model_variant base-9b \
  --model_path /LOCAL/FLUX.2-klein-base-9B \
  --data_dir /LOCAL/image-caption-pairs --output_dir /LOCAL/recovery-trial \
  --target_size 256 --optimizer adafactor --no_ema \
  --batch_size 1 --grad_accum 1 --num_workers 0 \
  --steps 4 --warmup_steps 0 --save_every 2 --log_every 1 \
  --seed 42 --sample_prompts --recovery_stop_after 2

python scripts/train_klein_standalone.py \
  --recovery --recovery_model_variant base-9b \
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

## Boundaries and limitations

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
