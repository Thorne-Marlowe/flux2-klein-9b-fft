# Dataset transport package (Phase 4.1)

This is a storage-provider-independent transport representation for the flat
image-plus-caption directory consumed by the standalone trainer. It does not
download data, invoke Runpod, alter the trainer, or start training.

The input directory must contain only regular top-level `.jpg`, `.jpeg`,
`.png`, or `.webp` images and exactly one same-stem lowercase `.txt` caption
for each image. Unmatched images, orphan captions, duplicate image stems,
subdirectories, links, and unrelated files are rejected. This is stricter than
the trainer's tolerant scan so packaging cannot silently omit a source file.

## Format

A package directory contains only:

```text
dataset-manifest.json
data-00001-of-000NN.tar
...
```

Tar shards are plain, uncompressed tar files. Their member order and tar
metadata are deterministic: source files are sorted, file mode is `0644`, and
timestamps, owner IDs, and names are normalized. The shard target is the sum of
source member bytes; an image and its caption are always kept in the same shard.
An oversized pair occupies its own shard.

The manifest records the format/schema version, fixed flat-layout contract,
pair count, shard sizes and SHA-256 hashes, and every member's safe flat path,
size, SHA-256, and shard. `dataset_identity` is the SHA-256 of canonical JSON
containing only the layout contract and path-sorted member `{path, size_bytes,
sha256}` records. It identifies reconstructed training content independently of
the chosen shard size.

## Local workflow

```bash
python scripts/pack_training_dataset.py pack \
  --input-dir /data/source --output-dir /data/package --shard-size-mib 1024

python scripts/pack_training_dataset.py validate --package-dir /data/package

python scripts/pack_training_dataset.py extract \
  --package-dir /data/package --output-dir /data/reconstructed
```

Extraction validates the manifest, every archive, and every member before it
uses a private sibling staging directory. It never uses `extractall`, rejects
links, devices, traversal, duplicate or unexpected members, and verifies the
completed staged directory before a rename exposes it. Existing destinations
are refused rather than overwritten. A failed extraction leaves no completed
destination.

The resulting `/data/reconstructed` directory is the normal flat dataset passed
to `--data_dir`; tar shards are transport only. This adds no GPU, training, or
deterministic-recovery qualification. Remote hydration and export belong to
later Phase 4 work.
