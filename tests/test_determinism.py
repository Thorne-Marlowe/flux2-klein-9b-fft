"""CPU diagnostics tests; no claim about CUDA determinism or 9B behavior."""
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import klein_determinism as diag
from scripts import klein_checkpoint as ck
from scripts import klein_recovery_training as recovery
from scripts import train_klein_standalone as trainer


class TinyVAE:
    def __init__(self):
        self.bn = SimpleNamespace(running_mean=torch.arange(8) / 10, running_var=torch.ones(8))
        self.config = SimpleNamespace(batch_norm_eps=1e-5)
        self.samples = 0

    def encode(self, images):
        mean = images[:, :2]
        std = torch.ones_like(mean) * 0.2
        def sample():
            self.samples += 1
            return mean + std * torch.randn_like(mean)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean, std=std, sample=sample))


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, hidden_states, **kwargs):
        return (self.dropout(hidden_states) * self.scale,)


PIPE = SimpleNamespace(_prepare_text_ids=lambda x: torch.zeros(x.shape[1], 4),
                       _prepare_latent_ids=lambda x: torch.zeros(x.shape[2] * x.shape[3], 4))


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_hash_dtypes_layout_and_no_mutation(self):
        for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool):
            value = torch.arange(60).reshape(3, 4, 5).to(dtype).transpose(0, 2)
            original = value.clone()
            record = diag.tensor_records([("x", value)], full_cpu=True)[0]
            raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record["coverage"], "complete")
            self.assertTrue(torch.equal(value, original))
        for value in (torch.tensor(1.0), torch.empty(0)):
            self.assertEqual(diag.tensor_records([("x", value)], full_cpu=True)[0]["observed_elements"], value.numel())

    def test_bounded_probes_noncontiguous_and_batched_transfers(self):
        large = torch.arange(100000, dtype=torch.float32).reshape(200, 500).t()
        original = large.clone()
        probe = diag.tensor_records([("large", large)])[0]
        self.assertEqual(probe["coverage"], "sampled")
        self.assertEqual(probe["observed_elements"], diag.PROBE_ELEMENTS)
        self.assertEqual(len(bytes.fromhex(probe["observed_bytes_hex"])), diag.PROBE_ELEMENTS * 4)
        self.assertTrue(torch.equal(large, original))
        small_model_tensor = diag.tensor_records([("bias", torch.ones(1024))], probe_only=True)[0]
        self.assertEqual(small_model_tensor["observed_elements"], diag.PROBE_ELEMENTS)
        self.assertEqual(small_model_tensor["coverage"], "sampled")
        native_cat = torch.cat
        sizes = []
        def bounded_cat(parts, *args, **kwargs):
            sizes.append(sum(p.numel() for p in parts))
            return native_cat(parts, *args, **kwargs)
        with patch.object(diag.torch, "cat", side_effect=bounded_cat):
            diag.tensor_records([(str(i), torch.ones(16384)) for i in range(40)])
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), diag.TRANSFER_BYTES)
        with patch.object(diag, "CHUNK_BYTES", 4096):
            self.assertEqual(diag.tensor_records([("x", large)], full_cpu=True)[0]["sha256"],
                hashlib.sha256(large.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())

    def test_rng_neutral_and_no_cuda_initialization(self):
        recovery.seed_recovery(123)
        data = recovery.AcknowledgedData([0, 1], 123, lambda b: b)
        before = diag.rng_record(data)
        trace = diag.DeterminismTrace(self.root / "rng.jsonl")
        with patch.object(torch.cuda, "is_initialized", return_value=False), \
             patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA initialized")):
            trace.emit("rng", tensors=[("x", torch.arange(100000))], rng=True, data=data)
            trace.finish()
        self.assertEqual(before, diag.rng_record(data))

    def run_training(self, name, enabled, kind):
        # Compare identical dependency initialization: first Transformers import
        # can consume Python RNG in the pinned environment. This is test setup,
        # not a change to production seeding or model-loading order.
        if kind == "adafactor":
            from transformers.optimization import Adafactor
        recovery.seed_recovery(321)
        model, vae = TinyFlow(), TinyVAE()
        if kind == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, foreach=False)
        else:
            optimizer = Adafactor(model.parameters(), lr=3e-5, relative_step=False, scale_parameter=False)
        trace = diag.DeterminismTrace(self.root / name) if enabled else None
        gradients, losses = [], []
        for attempt in range(2):
            images = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 48
            embeds = torch.ones(1, 2, 3)
            if trace:
                trace.attempt = attempt
                trace.emit("begin", tensors=diag.model_tensors(model), full_cpu=True, rng=True)
                before_opt = copy.deepcopy(optimizer.state_dict())
                trace.emit("optimizer_before", details=diag.optimizer_observation(model, optimizer))
                self.assertTrue(ck._equal_state(before_opt, optimizer.state_dict()))
            latents = trainer.encode_images_klein(vae, images, "cpu", torch.float32, trace=trace)
            loss = recovery.flow_loss(model, latents, embeds, PIPE, trainer, trace=trace)
            loss.backward()
            if trace:
                trace.emit("grads", tensors=diag.model_tensors(model, gradients=True))
            gradients.append(model.scale.grad.clone())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if trace:
                trace.emit("updated", tensors=diag.model_tensors(model),
                           details=diag.optimizer_observation(model, optimizer), rng=True)
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        if trace:
            trace.finish()
        self.assertEqual(vae.samples, 2)
        return {"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                "grads": gradients, "loss": losses, "rng": diag.rng_record()}

    def test_enabled_and_disabled_training_identical(self):
        for kind in ("adamw", "adafactor"):
            plain = self.run_training("unused", False, kind)
            traced = self.run_training(kind + ".jsonl", True, kind)
            again = self.run_training(kind + "-again.jsonl", True, kind)
            self.assertTrue(ck._equal_state(plain, traced), kind)
            self.assertTrue(ck._equal_state(traced, again), kind)
            self.assertEqual((self.root / (kind + ".jsonl")).read_bytes(),
                             (self.root / (kind + "-again.jsonl")).read_bytes())

    def test_disabled_calls_do_not_touch_diagnostic_module(self):
        with patch.object(diag, "tensor_records", side_effect=AssertionError("diagnostics invoked")), \
             patch.object(diag, "rng_record", side_effect=AssertionError("diagnostics invoked")):
            vae = TinyVAE()
            latents = trainer.encode_images_klein(vae, torch.ones(1, 3, 4, 4), "cpu", torch.float32)
            recovery.flow_loss(TinyFlow(), latents, torch.ones(1, 2, 3), PIPE, trainer).backward()

    def test_comparator_injected_mismatch_and_incomplete_trace(self):
        left, right = self.root / "a", self.root / "b"
        for path in (left, right):
            trace = diag.DeterminismTrace(path)
            trace.emit("first", tensors=[("x", torch.arange(4))])
            trace.emit("second", details={"lr": 3e-5})
            trace.finish()
        self.assertEqual(diag.compare_traces(left, right)["status"], "no_observed_difference")
        original = right.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        rows[1]["tensors"][0]["sha256"] = "0" * 64
        rows[2]["details"]["lr"] = 0.9
        right.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        result = diag.compare_traces(left, right)
        self.assertEqual(result["status"], "first_observed_difference")
        self.assertEqual(result["stage"], "first")
        self.assertEqual(result["field"], "record.tensors[0].sha256")
        right.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        self.assertEqual(diag.compare_traces(left, right)["status"], "incomplete")
        left.write_text("{broken\n")
        self.assertEqual(diag.compare_traces(left, right)["status"], "invalid")

    def test_sampled_equality_is_not_full_equality(self):
        value = torch.zeros(100000)
        first = diag.tensor_records([("x", value)])[0]
        index = next(i for i in range(100000) if i not in first["indices"])
        value[index] = 1
        self.assertEqual(first, diag.tensor_records([("x", value)])[0])
        self.assertNotEqual(diag.tensor_records([("x", torch.zeros_like(value))], full_cpu=True),
                            diag.tensor_records([("x", value)], full_cpu=True))

    def test_injected_tensor_and_rng_differences(self):
        for fault in ("weights", "rng"):
            paths = [self.root / (fault + str(i)) for i in range(2)]
            for index, path in enumerate(paths):
                recovery.seed_recovery(7)
                trace = diag.DeterminismTrace(path)
                trace.emit("baseline", rng=True)
                value = torch.ones(2)
                if index:
                    if fault == "weights":
                        value[0] = 2
                    else:
                        torch.rand(1)
                trace.emit("suspect", tensors=[("weights", value)], rng=True)
                trace.finish()
            result = diag.compare_traces(*paths)
            self.assertEqual(result["status"], "first_observed_difference")
            self.assertEqual(result["stage"], "suspect")

    def test_fresh_process_traces_match(self):
        paths = [self.root / "process-a.jsonl", self.root / "process-b.jsonl"]
        for path in paths:
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(path)],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(diag.compare_traces(*paths)["status"], "no_observed_difference")

    def test_exclusive_output_and_cli_guards(self):
        path = self.root / "existing"
        diag.DeterminismTrace(path)
        with self.assertRaises(FileExistsError):
            diag.DeterminismTrace(path)
        base = ["--data_dir", "data", "--output_dir", "out"]
        for extra in (["--determinism_trace", "trace"], ["--determinism_trace_steps", "2"]):
            with self.assertRaises(SystemExit):
                trainer.parse_args(base + extra)
        args = trainer.parse_args(base + ["--recovery", "--model_path", "local", "--target_size", "256",
                                 "--optimizer", "adafactor", "--determinism_trace", "trace"])
        self.assertEqual(args.determinism_trace, "trace")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        test = DeterminismTests()
        test.setUp()
        try:
            test.run_training(sys.argv[2], True, "adamw")
        finally:
            test.doCleanups()
    else:
        unittest.main()
