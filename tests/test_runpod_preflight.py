import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import runpod_preflight as preflight


class FakeTorch:
    __version__ = "2.10.0+cu128"
    version = SimpleNamespace(cuda="12.8")
    cuda = SimpleNamespace(is_available=lambda: False)


def portable_rename(source, destination):
    source.rename(destination)


class RunpodPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def write(self, relative, text="x"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def model(self):
        root = self.root / "model"
        index = {"_class_name": "Flux2KleinPipeline",
                 "transformer": ["diffusers", "Flux2Transformer2DModel"],
                 "vae": ["diffusers", "AutoencoderKLFlux2"],
                 "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                 "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                 "tokenizer": ["transformers", "Qwen2Tokenizer"]}
        transformer = {"_class_name": "Flux2Transformer2DModel", "num_layers": 8,
                       "num_single_layers": 24, "num_attention_heads": 32,
                       "attention_head_dim": 128, "joint_attention_dim": 12288,
                       "in_channels": 128, "patch_size": 1, "guidance_embeds": False,
                       "mlp_ratio": 3.0, "axes_dims_rope": [32, 32, 32, 32]}
        (root / "model_index.json").parent.mkdir(parents=True)
        (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        for component, stem in (("transformer", "diffusion_pytorch_model"),
                                ("vae", "diffusion_pytorch_model"),
                                ("text_encoder", "model")):
            folder = root / component
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps(transformer if component == "transformer" else {}), encoding="utf-8")
            (folder / (stem + ".safetensors")).write_bytes(b"fixture")
        for relative in ("scheduler/scheduler_config.json", "tokenizer/tokenizer.json",
                         "tokenizer/tokenizer_config.json"):
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        return root

    def dataset(self):
        root = self.root / "dataset"
        root.mkdir()
        (root / "image.png").write_bytes(b"not decoded by preflight")
        (root / "image.txt").write_text("caption", encoding="utf-8")
        return root

    def test_system_profile_without_gpu_is_a_warning_not_failure(self):
        result = preflight.Result("system")
        with patch.object(preflight, "_locked_core_versions", return_value={}), \
             patch.object(preflight.importlib.metadata, "version", return_value="fixture"), \
             patch.object(preflight.importlib, "import_module", return_value=FakeTorch()):
            preflight.add_runtime_checks(result, self.root, training=False)
        cuda = next(check for check in result.checks if check.identifier == "runtime.cuda")
        self.assertEqual(cuda.status, preflight.WARN)
        self.assertNotEqual(result.status, preflight.FAIL)

    def test_missing_and_valid_build_metadata_are_informational_and_parsed(self):
        runtime = self.root / "runtime"
        for relative in ("scripts/train_klein_standalone.py", "scripts/klein_recovery_training.py",
                         "scripts/klein_model_resolver.py", "scripts/klein_checkpoint.py", "requirements-smoke.txt"):
            path = runtime / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
        result = preflight.Result("system")
        preflight.add_repository_checks(result, runtime)
        self.assertEqual(result.checks[-1].status, preflight.WARN)
        (runtime / ".container-build-info.json").write_text(json.dumps({
            "source_commit": "a" * 40, "environment_version": "test",
            "requirements_lock_sha256": "b" * 64, "repository": "/opt/test"}), encoding="utf-8")
        result = preflight.Result("system")
        preflight.add_repository_checks(result, runtime)
        self.assertEqual(result.checks[-1].status, preflight.PASS)
        self.assertEqual(result.checks[-1].details["source_commit"], "a" * 40)

    def test_workspace_checks_and_temporary_write_cleanup(self):
        for name in preflight.WORKSPACE_DIRECTORIES:
            (self.workspace / name).mkdir()
        with patch.object(preflight, "probe_checkpoint_rename", return_value=preflight.Check(
                "filesystem.checkpoint_rename", preflight.PASS, "ok")):
            result = preflight.Result("system")
            preflight.add_workspace_checks(result, self.workspace, training=False)
        self.assertEqual(next(c for c in result.checks if c.identifier == "workspace.root").status, preflight.PASS)
        self.assertFalse(list(self.workspace.rglob(".klein-preflight-*")))

    def test_hard_link_probe_success_and_cleanup(self):
        check = preflight.probe_hard_link(self.workspace)
        self.assertEqual(check.status, preflight.PASS)
        self.assertFalse(list(self.workspace.glob(".klein-preflight-hardlink-*")))

    def test_hard_link_probe_failure_and_cleanup(self):
        with patch.object(preflight.os, "link", side_effect=OSError("no hard links")):
            check = preflight.probe_hard_link(self.workspace)
        self.assertEqual(check.status, preflight.FAIL)
        self.assertIn("no hard links", check.details["error"])
        self.assertFalse(list(self.workspace.glob(".klein-preflight-hardlink-*")))

    def test_checkpoint_rename_probe_success_and_failure_cleanup(self):
        with patch("scripts.klein_checkpoint._rename_checkpoint_no_replace", side_effect=portable_rename):
            check = preflight.probe_checkpoint_rename(self.workspace)
        self.assertEqual(check.status, preflight.PASS)
        self.assertFalse(list(self.workspace.glob(".klein-preflight-rename-*")))
        with patch("scripts.klein_checkpoint._rename_checkpoint_no_replace", side_effect=OSError("unsupported")):
            check = preflight.probe_checkpoint_rename(self.workspace)
        self.assertEqual(check.status, preflight.FAIL)
        self.assertFalse(list(self.workspace.glob(".klein-preflight-rename-*")))

    @unittest.skipUnless(sys.platform == "linux", "real renameat2 probe requires Linux")
    def test_real_linux_checkpoint_rename_probe_publishes_and_cleans_up(self):
        check = preflight.probe_checkpoint_rename(self.workspace)
        self.assertEqual(check.status, preflight.PASS, check.details)
        self.assertEqual(check.details["primitive"], "publisher no-replace rename")
        self.assertFalse(list(self.workspace.glob(".klein-preflight-rename-*")))

    def test_training_profile_missing_required_paths_fails(self):
        args = preflight.parse_args(["--profile", "training", "--workspace", str(self.workspace)])
        with patch.object(preflight, "add_runtime_checks"), patch.object(preflight, "add_workspace_checks"), \
             patch.object(preflight, "add_repository_checks"), patch.object(preflight, "add_determinism_check"):
            result = preflight.run_preflight(args)
        check = next(c for c in result.checks if c.identifier == "training.arguments")
        self.assertEqual(check.status, preflight.FAIL)
        self.assertEqual(result.exit_code, 1)

    def test_training_assets_validate_without_model_allocation(self):
        model, dataset = self.model(), self.dataset()
        output = self.root / "runs" / "test"
        output.parent.mkdir(parents=True)
        result = preflight.Result("training")
        with patch("scripts.klein_checkpoint._rename_checkpoint_no_replace", side_effect=portable_rename):
            preflight.add_training_checks(result, SimpleNamespace(model_path=model, dataset_path=dataset,
                                                                   output_path=output))
        self.assertTrue(result.checks)
        self.assertTrue(all(check.status == preflight.PASS for check in result.checks), result.checks)
        self.assertFalse(output.exists())
        self.assertFalse(list(model.parent.glob(".klein-preflight-hardlink-*")))
        rename = next(check for check in result.checks if check.identifier == "filesystem.output_checkpoint_rename")
        self.assertEqual(rename.details["publication_parent"], str(output))
        self.assertTrue(rename.details["stand_in_output_root"])
        self.assertFalse(list(output.parent.glob(".klein-preflight-output-*")))

    def test_existing_output_is_the_checkpoint_publication_parent(self):
        output = self.root / "runs" / "existing"
        output.mkdir(parents=True)
        with patch("scripts.klein_checkpoint._rename_checkpoint_no_replace", side_effect=portable_rename):
            publication_parent, check = preflight.validate_output_path(output)
            probe = preflight.probe_output_checkpoint_rename(publication_parent)
        self.assertEqual(check.status, preflight.PASS)
        self.assertEqual(probe.status, preflight.PASS)
        self.assertEqual(probe.details["publication_parent"], str(output))
        self.assertEqual(probe.details["probe_directory"], str(output))
        self.assertFalse(probe.details["stand_in_output_root"])
        self.assertFalse(list(output.glob(".klein-preflight-rename-*")))

    def test_deterministic_environment_is_opt_in_requirement(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, clear=True):
            result = preflight.Result("training")
            preflight.add_determinism_check(result, required=True)
            self.assertEqual(result.status, preflight.PASS)
        for environment in ({}, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                result = preflight.Result("system")
                preflight.add_determinism_check(result, required=False)
                self.assertEqual(result.status, preflight.WARN)
                result = preflight.Result("training")
                preflight.add_determinism_check(result, required=True)
                self.assertEqual(result.status, preflight.FAIL)

    def test_deterministic_recovery_option_is_training_only(self):
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
            preflight.parse_args(["--profile", "system", "--deterministic-recovery"])
        self.assertEqual(caught.exception.code, 2)

    def test_json_shape_status_derivation_exit_code_and_no_secret_leakage(self):
        result = preflight.Result("system")
        result.add("pass", preflight.PASS, "ok")
        result.add("warning", preflight.WARN, "review")
        self.assertEqual(result.status, preflight.WARN)
        self.assertEqual(result.exit_code, 0)
        result.add("failure", preflight.FAIL, "stop")
        self.assertEqual(result.status, preflight.FAIL)
        self.assertEqual(result.exit_code, 1)
        with patch.dict(os.environ, {"HF_TOKEN": "do-not-print"}, clear=True):
            preflight.add_determinism_check(result, required=False)
            document = json.dumps(result.document())
            human = preflight.render_human(result)
        self.assertNotIn("do-not-print", document + human)
        parsed = json.loads(document)
        self.assertEqual(set(parsed), {"profile", "status", "exit_code", "checks"})

    def test_main_json_and_exit_code_are_automation_friendly(self):
        expected = preflight.Result("system")
        expected.add("fixture", preflight.WARN, "ok")
        with patch.object(preflight, "run_preflight", return_value=expected), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            code = preflight.main(["--profile", "system", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stream.getvalue())["status"], preflight.WARN)


if __name__ == "__main__":
    unittest.main()
