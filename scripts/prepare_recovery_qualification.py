"""Prepare a reproducible, mid-epoch recovery qualification plan.

This helper performs local dataset/configuration checks and prints commands. It
does not load a model, download anything, start training, or select checkpoints.
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def dataset_inventory(data_dir):
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    pairs = []
    for image in sorted(root.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            caption = image.with_suffix(".txt")
            if caption.is_file():
                pairs.append({"image": image.name, "caption": caption.name})
    if len(pairs) != 3:
        raise ValueError(f"Milestone 3 requires exactly 3 image-caption pairs; found {len(pairs)}")
    return pairs


def _command(script, model, data, output, *, stop_after=None, save_every=7):
    args = ["python", "-B", script, "--recovery", "--deterministic_recovery",
            "--recovery_model_variant", "base-9b", "--model_path", model,
            "--data_dir", data, "--output_dir", output, "--target_size", "256",
            "--optimizer", "adafactor", "--no_ema", "--batch_size", "1",
            "--grad_accum", "1", "--num_workers", "0", "--steps", "7",
            "--warmup_steps", "0", "--lr", "3e-5", "--weight_decay", "0",
            "--max_grad_norm", "1", "--save_every", str(save_every),
            "--log_every", "1", "--seed", "42", "--sample_prompts"]
    if stop_after is not None:
        args.extend(["--recovery_stop_after", str(stop_after)])
    return " ".join(shlex.quote(str(value)) for value in args)


def build_plan(model_path, data_dir, evidence_dir, output_root, *, script="scripts/train_klein_standalone.py"):
    model = Path(model_path)
    if not model.is_dir():
        raise ValueError(f"Base 9B model directory does not exist: {model}")
    pairs = dataset_inventory(data_dir)
    evidence = Path(evidence_dir)
    outputs = Path(output_root)
    control = outputs / "control"
    trial = outputs / "trial"
    return {
        "qualification": "milestone-3-mid-epoch-recovery",
        "dataset": {"pair_count": len(pairs), "pairs": pairs, "batch_size": 1,
                     "workers": 0, "target_size": 256},
        "configuration": {"model": str(model), "precision": "bf16", "optimizer": "adafactor",
                          "effective_weight_decay": 0, "seed": 42, "steps": 7,
                          "warmup_steps": 0, "ema": False, "gradient_checkpointing": True},
        "boundary": {"checkpoint_attempt": 1, "checkpoint": str(trial / "checkpoint-1"),
                      "checkpoint_epoch": 0, "checkpoint_next_batch_index": 1,
                      "final_attempt": 7, "final_epoch": 2, "final_next_batch_index": 1},
        "commands": {
            "control": _command(script, model, data_dir, control),
            "interrupted": _command(script, model, data_dir, trial, stop_after=1, save_every=1),
        "resume": _command(script, model, data_dir, outputs / "resumed", save_every=7) +
                      " --recovery_resume " + shlex.quote(str(trial / "checkpoint-1")),
            "compare": "python -B scripts/compare_recovery_checkpoints.py " +
                       shlex.quote(str(control / "checkpoint-7")) + " " +
                       shlex.quote(str(outputs / "resumed" / "checkpoint-7")),
        },
        "evidence_dir": str(evidence),
        "evidence_procedure": [
            "git rev-parse HEAD; git status --short",
            "python -m pip freeze",
            "nvidia-smi -q",
            "sha256sum model_index.json transformer/config.json",
            "sha256sum control/checkpoint-7/manifest.json resumed/checkpoint-7/manifest.json",
            "run the compare command and save stdout/stderr",
            "preserve control/checkpoint-7, trial/checkpoint-1 and resumed/checkpoint-7 manifests/hashes without duplicating payload archives",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare the Milestone 3 recovery qualification plan")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_plan(args.model_path, args.data_dir, args.evidence_dir,
                                args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
