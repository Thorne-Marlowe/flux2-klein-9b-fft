"""Bootstrap tests never access a GPU, network, pip or real HF credentials."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import bootstrap_runpod as b


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hub = MagicMock()
        self.hub.get_token.return_value = "fake-cached-secret"
        self.api = self.hub.HfApi.return_value
        self.api.model_info.return_value = SimpleNamespace(
            sha="a" * 40, siblings=[SimpleNamespace(rfilename="model_index.json", size=2),
                                    SimpleNamespace(rfilename="transformer/diffusion_pytorch_model.safetensors", size=3)])

        def download(**kw):
            path = Path(kw["local_dir"]) / kw["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}" if path.suffix == ".json" else b"abc")
        self.hub.hf_hub_download.side_effect = download
        self.env = patch.dict(os.environ, {"HF_TOKEN": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.disk = patch.object(b.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12, total=2*10**12))
        self.disk.start()
        self.addCleanup(self.disk.stop)

    def test_auth_priority_and_immutable_download(self):
        with patch.dict(os.environ, {"HF_TOKEN": "fake-env-secret"}):
            result = b.download_missing(self.root, "main", hub=self.hub)
        self.hub.get_token.assert_not_called()
        self.hub.HfApi.assert_called_once_with(token="fake-env-secret")
        self.api.whoami.assert_called_once()
        self.api.auth_check.assert_called_once_with(repo_id=b.MODEL_ID, repo_type="model")
        self.assertEqual(result["downloaded_files"], 2)
        for call in self.hub.hf_hub_download.call_args_list:
            self.assertEqual(call.kwargs["revision"], "a" * 40)
            self.assertFalse(call.kwargs["force_download"])
        self.assertNotIn("secret", str(result))

    def test_cached_login_reuse_and_idempotence(self):
        b.download_missing(self.root, "main", hub=self.hub)
        self.hub.HfApi.assert_called_with(token="fake-cached-secret")
        self.hub.hf_hub_download.reset_mock()
        result = b.download_missing(self.root, "main", hub=self.hub)
        self.assertEqual(result["downloaded_files"], 0)
        self.hub.hf_hub_download.assert_not_called()

    def test_missing_auth_and_denied_access_stop_before_download(self):
        self.hub.get_token.return_value = None
        with self.assertRaisesRegex(b.BootstrapError, "hf auth login"):
            b.download_missing(self.root, "main", hub=self.hub)
        self.hub.HfApi.assert_not_called()
        self.hub.get_token.return_value = "fake-secret"
        self.api.auth_check.side_effect = RuntimeError("fake-secret")
        with self.assertRaises(b.BootstrapError) as error:
            b.download_missing(self.root, "main", hub=self.hub)
        self.assertNotIn("fake-secret", str(error.exception))
        self.api.model_info.assert_not_called()
        self.hub.hf_hub_download.assert_not_called()

    def test_failure_logs_are_redacted_and_partial_download_can_resume(self):
        normal = self.hub.hf_hub_download.side_effect
        def fail(**kw):
            if kw["filename"] == "model_index.json":
                normal(**kw)
            else:
                print("fake-secret")
                raise RuntimeError("fake-secret")
        self.hub.hf_hub_download.side_effect = fail
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaisesRegex(b.BootstrapError, "rerun setup"):
            b.download_missing(self.root, "main", hub=self.hub)
        self.assertNotIn("fake-secret", stream.getvalue())
        self.hub.hf_hub_download.reset_mock()
        self.hub.hf_hub_download.side_effect = normal
        self.assertEqual(b.download_missing(self.root, "main", hub=self.hub)["downloaded_files"], 1)
        self.hub.hf_hub_download.assert_called_once()

    def test_existing_mismatch_and_disk_shortage_do_not_download(self):
        (self.root / "model_index.json").write_bytes(b"bad")
        with self.assertRaisesRegex(b.BootstrapError, "differs in size"):
            b.download_missing(self.root, "main", hub=self.hub)
        self.assertEqual((self.root / "model_index.json").read_bytes(), b"bad")
        with patch.object(b.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(b.BootstrapError, "Insufficient"):
                b.download_missing(self.root / "other", "main", hub=self.hub)
        self.hub.hf_hub_download.assert_not_called()

    def test_remote_selection_rejects_paths_and_excludes_alternative_weights(self):
        entries = [SimpleNamespace(rfilename=n, size=5) for n in (
            "model_index.json", "transformer/config.json", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json", "transformer/diffusion_pytorch_model.fp16.safetensors",
            "transformer/pytorch_model.bin", "tokenizer/vocab.json", "README.md")]
        self.assertEqual(len(b.selected_remote_files(entries)), 4)
        entries.append(SimpleNamespace(rfilename="../outside", size=2))
        with self.assertRaisesRegex(b.BootstrapError, "Unsafe"):
            b.selected_remote_files(entries)

    def test_git_rejects_wrong_commit_and_dirty_tree(self):
        expected = "a" * 40
        with patch.object(b, "run", side_effect=[str(self.root), expected, "feature/9b-smoke-test", ""]):
            self.assertEqual(b.verify_git(self.root, expected)["commit"], expected)
        for results, message in (([str(self.root), "b" * 40, "branch"], "mismatch"),
                                 ([str(self.root), expected, "branch", " M README.md"], "local changes")):
            with patch.object(b, "run", side_effect=results), self.assertRaisesRegex(b.BootstrapError, message):
                b.verify_git(self.root, expected)
        with self.assertRaisesRegex(b.BootstrapError, "40-character"):
            b.verify_git(self.root, "HEAD")

    def test_effective_lock_markers_and_versions(self):
        lock = self.root / "requirements.txt"
        lock.write_text("--index-url https://example.invalid\n# comment\ntorch==2.10.0+cu128\nabsent==1 ; python_version < '1'\n")
        with patch.object(b.importlib.metadata, "version", return_value="2.10.0+cu128"):
            self.assertEqual(b.dependency_versions(lock), {"torch": "2.10.0+cu128"})
        with patch.object(b.importlib.metadata, "version", return_value="2.10.0+cpu"):
            with self.assertRaisesRegex(b.BootstrapError, "Dependency mismatch"):
                b.dependency_versions(lock)

    def test_real_two_image_dataset_validation(self):
        from PIL import Image
        for name in ("a", "b"):
            Image.new("RGB", (20, 30)).save(self.root / f"{name}.png")
            (self.root / f"{name}.txt").write_text("caption")
        self.assertEqual(b.validate_dataset(self.root, 32)["pairs"], 2)
        (self.root / "b.txt").write_text(" ")
        with self.assertRaisesRegex(b.BootstrapError, "empty caption"):
            b.validate_dataset(self.root, 32)
        (self.root / "b.txt").unlink()
        with self.assertRaisesRegex(b.BootstrapError, "exactly two"):
            b.validate_dataset(self.root, 32)

    def invoke_main(self, mode="preflight", output=None, model_error=None, install_error=None, extra=()):
        repo = Path(b.__file__).resolve().parents[1]
        venv = self.root / "env"
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        (venv / "bin/python").touch()
        arguments = [mode, "--workspace", str(self.root), "--repo", str(repo),
                     "--commit", "a" * 40, "--venv", str(venv), "--model", str(self.root / "model"),
                     "--dataset", str(self.root / "data"), "--output", str(output or self.root / "outputs")]
        arguments.extend(extra)
        stream = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name, result in (("verify_git", {"commit": "a" * 40}), ("dependency_versions", {}),
                                 ("gpu_report", []), ("validate_dataset", {"pairs": 2})):
                stack.enter_context(patch.object(b, name, return_value=result))
            stack.enter_context(patch.object(b, "validate_model", return_value={}, side_effect=model_error))
            downloader = stack.enter_context(patch.object(b, "download_missing", return_value={}))
            self.installer = stack.enter_context(patch.object(b, "install_dependencies", side_effect=install_error))
            runner = stack.enter_context(patch.object(b, "run", return_value=""))
            stack.enter_context(patch.object(b.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(b.platform, "machine", return_value="x86_64"))
            stack.enter_context(patch.object(b.sys, "prefix", str(venv.resolve())))
            stack.enter_context(patch.object(b.sys, "version_info", (3, 12)))
            stack.enter_context(contextlib.redirect_stdout(stream))
            stack.enter_context(contextlib.redirect_stderr(stream))
            code = b.main(arguments)
        return code, stream.getvalue(), downloader, runner

    def test_offline_preflight_never_installs_downloads_or_creates_output(self):
        code, output, download, runner = self.invoke_main()
        self.assertEqual(code, 0, output)
        self.assertIn("environment_ready", output)
        download.assert_not_called()
        self.installer.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args[0][-2:], ["pip", "check"])

    def test_outputs_preserved_and_validation_failure_cannot_report_ready(self):
        output_dir = self.root / "outputs"
        output_dir.mkdir()
        previous = output_dir / "trace.jsonl"
        previous.write_text("previous trace")
        code, output, download, _ = self.invoke_main(mode="setup")
        self.assertEqual(code, 1)
        self.assertNotIn("environment_ready", output)
        self.assertEqual(previous.read_text(), "previous trace")
        download.assert_not_called()
        code, output, download, _ = self.invoke_main(output=self.root / "new", model_error=ValueError("fake-secret"))
        self.assertEqual(code, 1)
        self.assertNotIn("environment_ready", output)
        self.assertNotIn("fake-secret", output)
        download.assert_not_called()

    def test_progress_heartbeat_visible_without_output_and_through_hub_suppression(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            progress = b.Progress(interval=0.01)
            progress.start("Hugging Face access")
            try:
                with b.quiet_hub():
                    print("fake-secret")
                    time.sleep(0.04)
                progress.start("model download")
            finally:
                progress.close()
        output = stream.getvalue()
        self.assertIn("Hugging Face access +", output)
        self.assertIn("still running", output)
        self.assertIn("completed", output)
        self.assertIn("model download", output)
        self.assertNotIn("fake-secret", output)

    def test_safe_pip_events_never_forward_arbitrary_text(self):
        for line in ("Authorization: Bearer fake-secret", "HF_TOKEN=fake-secret", "ERROR: fake-secret"):
            self.assertIsNone(b.safe_pip_event(line, {"torch"}))
        for line in ("Collecting torch==2 (from https://fake-secret@example.invalid)",
                     "Downloading https://fake-secret@example.invalid/artifact.whl",
                     "Using cached fake-secret", "WARNING: Retrying fake-secret",
                     "WARNING: ReadTimeoutError('fake-secret')"):
            event = b.safe_pip_event(line, {"torch"})
            self.assertIsNotNone(event)
            self.assertNotIn("fake-secret", event)
        self.assertIsNone(b.safe_pip_event("Collecting fake-secret", {"torch"}))
        self.assertIn("network socket timeout", b.safe_pip_event("ReadTimeoutError", {"torch"}))

    def pip_fixture(self, source, *, max_seconds=5):
        """Run a tiny local Python child INSTEAD OF pip; no network or install."""
        lock = self.root / "lock.txt"
        lock.write_text("torch==2.10.0+cu128\n")
        native_popen = subprocess.Popen
        commands, children = [], []
        def spawn(command, **kwargs):
            commands.append(command)
            child = native_popen([sys.executable, "-u", "-c", source], **kwargs)
            children.append(child)
            return child
        stream = io.StringIO()
        error = None
        with contextlib.redirect_stdout(stream), patch.object(b.subprocess, "Popen", side_effect=spawn):
            progress = b.Progress(interval=0.01)
            progress.start("dependency installation", "Rerun setup with the same venv; cached artifacts retained.")
            try:
                b.install_dependencies(Path(sys.executable), lock, progress, timeout=120, retries=3, max_seconds=max_seconds)
            except b.BootstrapError as exc:
                error = exc
                progress.fail(str(exc))
            finally:
                progress.close()
        self.assertTrue(all(p.poll() is not None for p in children))
        return commands[0], stream.getvalue(), error

    def test_streamed_safe_progress_and_quiet_slow_operation(self):
        command, output, error = self.pip_fixture(
            "import time; print('Collecting torch==2.10.0+cu128'); "
            "print('Downloading https://fake-secret@example.invalid/wheel'); "
            "print('Authorization: fake-secret'); time.sleep(0.15)")
        self.assertIsNone(error)
        self.assertIn("pip: resolving torch", output)
        self.assertIn("pip: downloading artifact", output)
        self.assertIn("quiet output alone is not evidence of a stall", output)
        self.assertNotIn("fake-secret", output)
        self.assertEqual(command[command.index("--timeout") + 1], "120")
        self.assertEqual(command[command.index("--retries") + 1], "3")
        self.assertNotIn("--force-reinstall", command)
        self.assertNotIn("--upgrade", command)

    def test_total_budget_terminates_silent_child_with_safe_recovery_message(self):
        _, output, error = self.pip_fixture("import time; time.sleep(30)", max_seconds=0.05)
        self.assertIsInstance(error, b.BootstrapError)
        self.assertIn("total runtime budget", output)
        self.assertIn("not proof of a network stall", output)
        self.assertIn("Recovery: Rerun setup", output)
        self.assertRegex(output, r"dependency installation \+\d+s.*FAILED")

    def test_pip_nonzero_exit_is_safe_and_does_not_retry_whole_install(self):
        _, output, error = self.pip_fixture("import sys; print('ERROR: fake-secret'); sys.exit(2)")
        self.assertIsInstance(error, b.BootstrapError)
        self.assertIn("exited with code 2", output)
        self.assertNotIn("fake-secret", output)

    def test_setup_can_retry_existing_environment_without_cleanup(self):
        env = self.root / "env"
        env.mkdir()
        installed = env / "installed-marker"
        installed.write_text("keep")
        code, output, download, _ = self.invoke_main(mode="setup", install_error=b.BootstrapError("Pip exited with code 2"))
        self.assertEqual(code, 1)
        self.assertIn("dependency installation", output)
        self.assertIn("FAILED", output)
        self.assertIn("same venv", output)
        self.assertNotIn("environment_ready", output)
        download.assert_not_called()
        code, output, _, runner = self.invoke_main(mode="setup")
        self.assertEqual(code, 0, output)
        self.installer.assert_called_once()
        self.assertEqual(installed.read_text(), "keep")
        self.assertEqual(runner.call_count, 1)  # pip check only; no venv recreation.
        for stage in ("repository validation", "virtual environment setup", "dependency installation",
                      "dependency verification", "dataset validation", "final model and output checks"):
            self.assertIn(stage, output)

    def test_interrupted_install_reports_stage_and_keeps_environment(self):
        code, output, download, _ = self.invoke_main(mode="setup", install_error=KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertIn("dependency installation", output)
        self.assertIn("Interrupted", output)
        self.assertIn("same venv", output)
        self.assertNotIn("environment_ready", output)
        self.assertTrue((self.root / "env/bin/python").exists())
        download.assert_not_called()

    def test_hub_access_and_download_have_separate_visible_stages(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            progress = b.Progress()
            try:
                b.download_missing(self.root / "model", "main", hub=self.hub, progress=progress)
            finally:
                progress.close()
        output = stream.getvalue()
        self.assertIn("Hugging Face access", output)
        self.assertIn("model download", output)
        self.assertNotIn("fake-cached-secret", output)

    def test_negative_retry_and_timeout_arguments_rejected_without_echo(self):
        for option, value in (("--pip-retries", "-1"), ("--pip-retries", "11"),
                              ("--pip-timeout", "0"), ("--pip-max-seconds", "-1"),
                              ("--progress-interval", "0"), ("--pip-retries", "fake-secret")):
            with self.subTest(option=option, value=value), self.assertRaises(SystemExit):
                self.invoke_main(extra=[option, value])

    def test_accidental_cli_token_is_not_echoed(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), self.assertRaises(SystemExit):
            b.main(["preflight", "--token", "fake-secret"])
        self.assertNotIn("fake-secret", stream.getvalue())

    def test_real_model_layout_headers_and_metadata_without_model_allocation(self):
        import torch
        from safetensors.torch import save_file
        index = {"_class_name": "Flux2KleinPipeline",
                 "transformer": ["diffusers", "Flux2Transformer2DModel"],
                 "vae": ["diffusers", "AutoencoderKLFlux2"],
                 "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                 "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                 "tokenizer": ["transformers", "Qwen2TokenizerFast"]}
        architecture = {"_class_name": "Flux2Transformer2DModel", "num_layers": 8,
                        "num_single_layers": 24, "num_attention_heads": 32,
                        "attention_head_dim": 128, "joint_attention_dim": 12288,
                        "in_channels": 128, "patch_size": 1, "guidance_embeds": False,
                        "mlp_ratio": 3.0, "axes_dims_rope": [32, 32, 32, 32]}
        def write(name, data):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        write("model_index.json", index)
        for component, stem in (("transformer", "diffusion_pytorch_model"),
                                ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")):
            write(component + "/config.json", architecture if component == "transformer" else {})
            save_file({"fixture": torch.zeros(1)}, self.root / component / (stem + ".safetensors"))
        for path in ("scheduler/scheduler_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
            write(path, {})
        self.assertEqual(b.validate_model(self.root)["selected_files"], 10)
        index["is_distilled"] = True
        write("model_index.json", index)
        with self.assertRaisesRegex(ValueError, "is_distilled"):
            b.validate_model(self.root)
        index["is_distilled"] = False
        write("model_index.json", index)
        (self.root / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"corrupt")
        with self.assertRaises(Exception):
            b.validate_model(self.root)


if __name__ == "__main__":
    unittest.main()
