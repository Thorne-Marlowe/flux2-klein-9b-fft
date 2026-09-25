#!/usr/bin/env python3
"""Pack, validate, or extract the provider-independent Klein dataset transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Support the repository's documented ``python scripts/...`` invocation as
# well as package imports in CPU tests.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dataset_transport import DatasetTransportError, extract_package, pack_dataset, validate_package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Pack a flat image-caption directory into deterministic tar shards.")
    pack.add_argument("--input-dir", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--shard-size-mib", type=int, default=1024,
                      help="Target sum of source member bytes per shard (default: 1024).")
    validate = commands.add_parser("validate", help="Validate a package without extracting it.")
    validate.add_argument("--package-dir", type=Path, required=True)
    extract = commands.add_parser("extract", help="Validate, stage, and expose a normal flat training dataset.")
    extract.add_argument("--package-dir", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "pack":
            manifest = pack_dataset(args.input_dir, args.output_dir, args.shard_size_mib * 1024 * 1024)
        elif args.command == "validate":
            manifest = validate_package(args.package_dir)
        else:
            manifest = extract_package(args.package_dir, args.output_dir)
    except DatasetTransportError as error:
        print(f"Dataset transport failed: {error}", flush=True)
        return 1
    print(json.dumps({"status": "ok", "dataset_identity": manifest["dataset_identity"],
                      "pair_count": manifest["pair_count"], "shard_count": len(manifest["shards"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
