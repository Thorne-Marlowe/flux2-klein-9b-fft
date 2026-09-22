"""CPU-only strict opt-in tests; no claim about CUDA/9B reproducibility."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import klein_checkpoint as ck
from scripts import klein_determinism as diag
from scripts import klein_model_resolver as resolver
from scripts import klein_recovery_metadata as md
from scripts import klein_recovery_training as recovery
from scripts import train_klein_standalone as trainer


class StrictDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.original = (torch.are_deterministic_algorithms_enabled(),
                         torch.is_deterministic_algorithms_warn_only_enabled(),
                         torch.backends.cudnn.benchmark)
        self.addCleanup(self.restore_backends)
        self.env = patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cuda = patch.object(torch.cuda, "is_initialized", return_value=False)
        self.cuda.start()
        self.addCleanup(self.cuda.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enabled = SimpleNamespace(recovery=True, smoke_test=False, deterministic_recovery=True)

    def restore_backends(self):
        torch.use_deterministic_algorithms(self.original[0], warn_only=self.original[1])
        torch.backends.cudnn.benchmark = self.original[2]

    def args(self, *extra):
        return trainer.parse_args(["--recovery", "--model_path", "local-model",
            "--data_dir", "data", "--output_dir", str(self.root / "out"),
            "--target_size", "32", "--optimizer", "adafactor", "--batch_size", "1",
            "--grad_accum", "1", "--num_workers", "0", "--sample_prompts", *extra])

    def test_parser_recovery_only_and_default_off(self):
        self.assertFalse(self.args().deterministic_recovery)
        self.assertTrue(self.args("--deterministic_recovery").deterministic_recovery)
        self.assertTrue(self.args("--deterministic_recovery", "--determinism_trace", "trace").deterministic_recovery)
        for flags in ([], ["--smoke_test"], ["--recovery", "--smoke_test"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(["--output_dir", "out", "--data_dir", "data",
                                    "--deterministic_recovery", *flags])

    def test_disabled_does_not_inspect_cuda_environment_or_mutate_backends(self):
        for args in (SimpleNamespace(), SimpleNamespace(deterministic_recovery=False)):
            with patch.object(torch.cuda, "is_initialized", side_effect=AssertionError("CUDA inspected")), \
                 patch.object(torch, "use_deterministic_algorithms", side_effect=AssertionError("backend changed")), \
                 patch.object(recovery.os, "environ", {}):
                self.assertIsNone(recovery.configure_deterministic_recovery(args))
            self.assertEqual(self.original, (torch.are_deterministic_algorithms_enabled(),
                                            torch.is_deterministic_algorithms_warn_only_enabled(),
                                            torch.backends.cudnn.benchmark))

    def test_opt_in_enforces_strict_flags_preserves_rng_and_other_settings(self):
        model = SimpleNamespace(attn_processors={"test": SimpleNamespace(_attention_backend=None)})
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = True
        before = md.collect_backend_settings(model)
        rng = diag.rng_record()
        settings = recovery.configure_deterministic_recovery(self.enabled)
        after = md.collect_backend_settings(model)
        self.assertTrue(settings["deterministic_algorithms"])
        self.assertFalse(settings["deterministic_warn_only"])
        self.assertFalse(settings["cudnn_benchmark"])
        self.assertEqual(settings["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        for key in before.keys() - {"deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark"}:
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(rng, diag.rng_record())
        trace = diag.DeterminismTrace(self.root / "backend.jsonl")
        trace.emit("configuration", details={"backend_settings": after})
        record = json.loads(trace.path.read_text().splitlines()[1])["details"]["backend_settings"]
        self.assertEqual(record, after)

    def test_missing_or_incompatible_workspace_fails_before_any_setting_change(self):
        for value in (None, "", ":16:8", ":4096:2", " :4096:8"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
                with patch.object(torch, "use_deterministic_algorithms") as setter:
                    with self.assertRaisesRegex(ck.CheckpointValidationError, "before launch"):
                        recovery.train_recovery(self.enabled)
                    setter.assert_not_called()
                self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), value)

    def test_already_initialized_cuda_rejected_even_with_correct_environment(self):
        with patch.object(torch.cuda, "is_initialized", return_value=True), \
             patch.object(torch, "use_deterministic_algorithms") as setter:
            with self.assertRaisesRegex(ck.CheckpointValidationError, "after CUDA initialization"):
                recovery.configure_deterministic_recovery(self.enabled)
            setter.assert_not_called()

    def test_direct_nonrecovery_call_is_rejected(self):
        for recovery_mode, smoke in ((False, False), (True, True)):
            args = SimpleNamespace(recovery=recovery_mode, smoke_test=smoke, deterministic_recovery=True)
            with self.assertRaisesRegex(ValueError, "requires recovery"):
                trainer.train(args)
            with self.assertRaises(ck.CheckpointValidationError):
                recovery.configure_deterministic_recovery(args)

    def test_setup_failure_propagates_without_warn_only_retry(self):
        with patch.object(torch, "use_deterministic_algorithms", side_effect=RuntimeError("injected failure")) as setter:
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                recovery.train_recovery(self.enabled)
            setter.assert_called_once_with(True, warn_only=False)

    def test_real_cpu_unsupported_operation_raises_and_trace_remains_incomplete(self):
        recovery.configure_deterministic_recovery(self.enabled)
        trace = diag.DeterminismTrace(self.root / "unsupported.jsonl")
        trace.emit("before_operation")
        # Pinned PyTorch documents put_(accumulate=False) as unsupported in
        # deterministic mode on CPU too; this is not a simulated CUDA result.
        with self.assertRaisesRegex(RuntimeError, "deterministic"):
            torch.zeros(2).put_(torch.tensor([0, 0]), torch.ones(2), accumulate=False)
        self.assertEqual(diag.compare_traces(trace.path, trace.path)["status"], "incomplete")
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_recovery_entry_configures_before_accelerator_and_loading_and_records_trace(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                torch.use_deterministic_algorithms(False)
                torch.backends.cudnn.benchmark = True
                path = self.root / f"entry-{enabled}.jsonl"
                args = self.args("--determinism_trace", str(path), *(["--deterministic_recovery"] if enabled else []))
                def accelerator(**kwargs):
                    self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                    self.assertEqual(torch.backends.cudnn.benchmark, not enabled)
                    return SimpleNamespace(state=SimpleNamespace())
                with contextlib.ExitStack() as stack:
                    # Restore only our stub, not newly imported torchvision
                    # modules whose native operator registrations stay alive.
                    previous = sys.modules.get("accelerate")
                    sys.modules["accelerate"] = SimpleNamespace(Accelerator=accelerator)
                    stack.callback(lambda saved=previous: sys.modules.pop("accelerate", None)
                                   if saved is None else sys.modules.__setitem__("accelerate", saved))
                    stack.enter_context(patch.object(trainer, "ImageTextDataset", return_value=[1]))
                    stack.enter_context(patch.object(trainer, "preflight_model_config"))
                    stack.enter_context(patch.object(trainer, "preflight_smoke_runtime"))
                    stack.enter_context(patch.object(resolver, "resolve_model_files", return_value=("file",)))
                    loader = stack.enter_context(patch.object(resolver, "load_selected_pipeline", side_effect=RuntimeError("stop before loading")))
                    stack.enter_context(patch.object(ck, "_read_json", return_value={"is_distilled": False}))
                    stack.enter_context(patch.object(recovery, "preallocation_check", return_value={}))
                    stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
                    stack.enter_context(patch.object(torch.cuda, "manual_seed_all"))
                    with self.assertRaisesRegex(RuntimeError, "stop before loading"):
                        trainer.train(args)
                    loader.assert_called_once()
                seeded = json.loads(path.read_text().splitlines()[1])
                self.assertEqual(seeded["stage"], "seeded")
                if enabled:
                    self.assertTrue(seeded["details"]["deterministic_recovery"]["requested"])
                    self.assertFalse(seeded["details"]["deterministic_recovery"]["deterministic_warn_only"])
                else:
                    self.assertEqual(seeded["details"], {"seed": args.seed})


if __name__ == "__main__":
    unittest.main()
