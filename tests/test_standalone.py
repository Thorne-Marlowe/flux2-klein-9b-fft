"""CPU-only tests. The tiny mocked loop does NOT validate Diffusers or 9B execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "standalone", Path(__file__).resolve().parents[1] / "scripts/train_klein_standalone.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


BASE_CONFIG = {"_class_name": "Flux2KleinPipeline", "is_distilled": False,
               "transformer": ["diffusers", "Flux2Transformer2DModel"]}
BASE_9B_ARCHITECTURE = {"num_layers": 8, "num_single_layers": 24,
                        "num_attention_heads": 32, "attention_head_dim": 128,
                        "joint_attention_dim": 12288, "in_channels": 128}


class Config(dict):
    __getattr__ = dict.__getitem__


class TinyTransformer(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 4, dtype=torch.bfloat16)
        self.config = Config(in_channels=4, joint_attention_dim=6, guidance_embeds=False)
        self.requires_grad_(False)  # Emulate a pipeline that loaded frozen weights.
        self.save_pretrained = Mock(side_effect=AssertionError("Smoke must not save weights"))

    def enable_gradient_checkpointing(self):
        pass

    def forward(self, hidden_states, **kwargs):
        return (super().forward(hidden_states),)


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)
        self.config = Config(latent_channels=1, batch_norm_eps=1e-5)

    def encode(self, images):
        latents = torch.nn.functional.avg_pool2d(images[:, :1].float(), 8).to(images.dtype)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latents))


class TinyAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    is_main_process = True
    sync_gradients = True
    optimizer_step_was_skipped = False
    distributed_type = "NO"
    state = SimpleNamespace()

    def __init__(self, **kwargs):
        pass

    def prepare(self, *objects):
        return objects

    def unwrap_model(self, model):
        return model

    def accumulate(self, model):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def wait_for_everyone(self):
        pass


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        Image.new("RGB", (80, 40), (100, 150, 200)).save(self.data / "one.png")
        (self.data / "one.txt").write_text("a test image", encoding="utf-8")

    def args(self, *extra):
        explicit_smoke = (["--model_path", "local-base-9b", "--target_size", "32"]
                          if "--smoke_test" in extra else [])
        return trainer.parse_args(["--data_dir", str(self.data), "--output_dir",
                                   str(self.root / "out"), *explicit_smoke, *extra])

    def test_smoke_requires_explicit_model_and_size_and_preserves_normal_defaults(self):
        required = ["--data_dir", str(self.data), "--output_dir", str(self.root / "out")]
        for extra in [[], ["--model_path", "local-base-9b"], ["--target_size", "256"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args([*required, "--smoke_test", *extra])
        self.assertEqual(self.args().model_path, "black-forest-labs/FLUX.2-klein-base-4B")
        self.assertEqual(self.args().target_size, 1024)

    def test_ema_defaults_and_explicit_disable(self):
        self.assertTrue(self.args().use_ema)
        self.assertTrue(self.args("--use_ema").use_ema)
        self.assertFalse(self.args("--no_ema").use_ema)

    def test_smoke_uses_fixed_size_and_uncached_pair(self):
        args = self.args("--smoke_test", "--target_size", "64")
        with patch.object(trainer, "find_bucket", side_effect=AssertionError("bucket called")), \
                patch.object(trainer.ImageTextDataset, "_find_cached_latent",
                             side_effect=AssertionError("cache called")):
            dataset = trainer.preflight_dataset(args)
            batch = trainer.collate_fn([dataset[0]])
        self.assertEqual(batch["pixel_values"].shape, (1, 3, 64, 64))
        self.assertEqual(batch["captions"], ["a test image"])
        self.assertEqual((args.batch_size, args.grad_accum, args.steps, args.num_workers), (1, 1, 1, 0))
        self.assertFalse(args.use_ema)
        self.assertTrue((batch["pixel_values"].abs() <= 1).all())

    def test_ordinary_dataset_still_uses_buckets(self):
        with patch.object(trainer, "find_bucket", return_value=(32, 16)) as bucket:
            sample = trainer.ImageTextDataset(self.data)[0]
        bucket.assert_called_once_with(80, 40)
        self.assertEqual(sample["pixel_values"].shape, (3, 16, 32))

    def test_dataset_rejects_invalid_smoke_inputs(self):
        for extra in [("--use_cached_latents",), ("--resume_from", "checkpoint"),
                      ("--target_size", "17")]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                trainer.preflight_dataset(self.args("--smoke_test", *extra))
        (self.data / "one.txt").write_text(" ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "caption must not be empty"):
            trainer.preflight_dataset(self.args("--smoke_test", "--target_size", "32"))
        (self.data / "one.txt").unlink()
        with self.assertRaisesRegex(ValueError, "full batch"):
            trainer.preflight_dataset(self.args("--smoke_test"))

    def test_dataset_rejects_multiple_pairs_and_corrupt_image(self):
        Image.new("RGB", (32, 32)).save(self.data / "two.png")
        (self.data / "two.txt").write_text("second", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            trainer.preflight_dataset(self.args("--smoke_test"))
        (self.data / "two.txt").unlink()
        (self.data / "one.png").write_bytes(b"invalid image")
        with self.assertRaises(OSError):
            trainer.preflight_dataset(self.args("--smoke_test"))

    def test_model_preflight_rejects_distilled_unknown_and_wrong_pipeline(self):
        trainer.preflight_model_config(BASE_CONFIG)
        for config in [{}, {**BASE_CONFIG, "is_distilled": True},
                       {**BASE_CONFIG, "is_distilled": None},
                       {**BASE_CONFIG, "_class_name": "FluxPipeline"},
                       {**BASE_CONFIG, "transformer": ["diffusers", "Other"]}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                trainer.preflight_model_config(config)

    def test_component_width_preflight(self):
        pipe = self.tiny_pipe()
        trainer.preflight_components(pipe, 32)
        pipe.transformer.config.joint_attention_dim = 999
        with self.assertRaisesRegex(ValueError, "text width"):
            trainer.preflight_components(pipe, 32)

    def test_9b_architecture_rejects_4b_and_missing_or_conflicting_metadata(self):
        trainer.preflight_9b_architecture(BASE_9B_ARCHITECTURE)
        trainer.preflight_9b_architecture({**BASE_9B_ARCHITECTURE, "out_channels": None})
        for key in BASE_9B_ARCHITECTURE:
            config = dict(BASE_9B_ARCHITECTURE)
            del config[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                trainer.preflight_9b_architecture(config)
        for overrides in [{"num_layers": 5, "num_single_layers": 20, "num_attention_heads": 24,
                           "joint_attention_dim": 7680}, {"out_channels": 64}, {"guidance_embeds": True},
                          {"patch_size": 2}, {"axes_dims_rope": [16, 16, 16, 16]}]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                trainer.preflight_9b_architecture({**BASE_9B_ARCHITECTURE, **overrides})

    def test_runtime_rejects_cpu_and_multiple_processes(self):
        with self.assertRaisesRegex(ValueError, "single CUDA GPU"):
            trainer.preflight_smoke_runtime(TinyAccelerator())
        with self.assertRaisesRegex(ValueError, "single CUDA GPU"):
            trainer.preflight_smoke_runtime(SimpleNamespace(num_processes=2, device=torch.device("cuda"),
                                                           distributed_type="NO", state=SimpleNamespace()))

    def test_runtime_rejects_distributed_backends_even_with_one_process(self):
        for mode in ["FSDP", "DEEPSPEED", "MULTI_GPU", "MEGATRON_LM"]:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "distributed execution"):
                trainer.preflight_smoke_runtime(SimpleNamespace(
                    num_processes=1, device=torch.device("cuda"),
                    distributed_type=SimpleNamespace(value=mode), state=SimpleNamespace()))
        for plugin in ["fsdp_plugin", "deepspeed_plugin"]:
            with self.subTest(plugin=plugin), self.assertRaisesRegex(ValueError, "distributed execution"):
                trainer.preflight_smoke_runtime(SimpleNamespace(
                    num_processes=1, device=torch.device("cuda"), distributed_type="NO",
                    state=SimpleNamespace(**{plugin: object()})))

    def test_smoke_rejects_environment_backends_before_importing_accelerate(self):
        for variable in ["ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"]:
            with self.subTest(variable=variable), patch.dict(trainer.os.environ, {variable: "true"}), \
                    self.assertRaisesRegex(ValueError, "FSDP/DeepSpeed"):
                trainer.train(self.args("--smoke_test"))

    def test_optimizer_coverage_rejects_frozen_missing_extra_duplicate(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        self.assertEqual(trainer.parameter_coverage(model, optimizer)["total_parameters"], 6)
        model.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "frozen"):
            trainer.parameter_coverage(model, optimizer)
        model.requires_grad_(True)
        for params in [[model.weight], [model.weight, model.bias, torch.nn.Parameter(torch.ones(1))],
                       [model.weight, model.bias, model.weight]]:
            with self.subTest(params=len(params)), self.assertRaises(ValueError):
                trainer.parameter_coverage(model, SimpleNamespace(param_groups=[{"params": params}]))

    def test_gradient_and_update_diagnostics(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = trainer.parameter_probes(model)
        model(torch.ones(1, 2)).sum().backward()
        diagnostics = trainer.gradient_diagnostics(model)
        self.assertEqual(diagnostics["missing_gradients"], [])
        self.assertAlmostEqual(diagnostics["gradient_l2_norm_before_clip"], 3 ** 0.5)
        optimizer.step()
        changes = trainer.parameter_change_diagnostics(before, trainer.parameter_probes(model))
        self.assertEqual(changes["changed_sampled_values"], 3)
        model.weight.grad[0, 0] = float("nan")
        model.bias.grad = None
        diagnostics = trainer.gradient_diagnostics(model)
        self.assertEqual(diagnostics["nonfinite_gradients"], ["weight"])
        self.assertEqual(diagnostics["missing_gradients"], ["bias"])

    def test_gradient_coverage_distinguishes_zero_missing_and_nonfinite(self):
        model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(3)) for _ in range(4)])
        model[0].grad = torch.tensor([1., 0., -1.])
        model[1].grad = torch.zeros(3)
        model[2].grad = torch.tensor([float("inf"), 0., 0.])
        diagnostics = trainer.gradient_diagnostics(model)
        self.assertEqual(diagnostics["gradient_parameter_tensors"], 3)
        self.assertEqual(diagnostics["nonzero_gradient_parameter_tensors"], 1)
        self.assertEqual(diagnostics["gradient_details"]["0"]["nonzero_values"], 2)
        self.assertTrue(diagnostics["gradient_details"]["1"]["finite"])
        self.assertFalse(diagnostics["gradient_details"]["2"]["finite"])
        self.assertFalse(diagnostics["gradient_details"]["3"]["present"])

    def test_frozen_checks_detect_parameter_buffer_and_trainability_changes(self):
        for change in ["parameter", "unsampled_parameter", "buffer", "trainability", "eval"]:
            with self.subTest(change=change):
                model = torch.nn.Linear(1024, 1).requires_grad_(False).eval()
                model.register_buffer("running_count", torch.tensor(2 ** 40))
                before = trainer.frozen_snapshot(model)
                self.assertTrue(trainer.frozen_diagnostics(model, before)["unchanged_under_checks"])
                self.assertTrue(all(v.device.type == "cpu" for v in before["samples"].values()))
                with torch.no_grad():
                    if change == "parameter":
                        model.weight.data[0, 0] += 1  # Deliberately bypass the version counter.
                    elif change == "unsampled_parameter":
                        model.weight[0, 1] += 1  # Detected by the mutation counter.
                    elif change == "buffer":
                        model.running_count += 1
                    elif change == "trainability":
                        model.requires_grad_(True)
                    else:
                        model.train()
                self.assertFalse(trainer.frozen_diagnostics(model, before)["unchanged_under_checks"])

    def test_optimizer_membership_and_nested_state_dtypes(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW([{"params": [model.weight]}, {"params": [model.bias]}])
        self.assertEqual(trainer.parameter_coverage(model, optimizer)["optimizer_membership"],
                         {"weight": 0, "bias": 1})
        optimizer.state[model.weight] = {"state": [torch.ones(1), torch.ones(1, dtype=torch.bfloat16)],
                                         "step": 1}
        self.assertEqual(trainer.optimizer_state_dtypes(optimizer), {"torch.float32": 1, "torch.bfloat16": 1})

    def test_patch_and_pack_round_trip(self):
        latent = torch.randn(1, 32, 8, 12)
        packed = trainer.pack_latents(trainer.patchify(latent))
        restored = trainer.unpatchify(trainer.unpack_latents(packed, 4, 6))
        torch.testing.assert_close(restored, latent)

    def test_probes_handle_indices_beyond_float32_integer_precision(self):
        model = torch.nn.Module()
        model.register_parameter("large", torch.nn.Parameter(torch.zeros(16_777_220)))
        with torch.no_grad():
            model.large[-1] = 7
        probes = trainer.parameter_probes(model)
        self.assertEqual(probes["large"].numel(), 256)
        self.assertEqual(probes["large"][-1].item(), 7)

    def tiny_pipe(self):
        text = torch.nn.Linear(2, 2)
        text.config = Config(hidden_size=2, num_hidden_layers=27)
        return SimpleNamespace(transformer=TinyTransformer(), vae=TinyVAE(),
                               text_encoder=text, tokenizer=object(), vae_scale_factor=8)

    def run_mocked_loop(self, *, failure=None, ordinary=False, mutate_buffer=False,
                        partial_update=False, invalid_metadata=False):
        pipe = self.tiny_pipe()
        pipeline = Mock()
        pipeline.load_config.return_value = BASE_CONFIG
        pipeline.from_pretrained.return_value = pipe
        pipeline._get_qwen3_prompt_embeds.return_value = torch.ones(1, 4, 6, dtype=torch.bfloat16)
        pipeline._prepare_latent_ids.return_value = torch.zeros(1, 4, 4)
        pipeline._prepare_text_ids.return_value = torch.zeros(1, 4, 4)
        transformer_class = Mock()
        transformer_class.load_config.return_value = BASE_9B_ARCHITECTURE
        if invalid_metadata:
            transformer_class.load_config.return_value = {**BASE_9B_ARCHITECTURE, "num_layers": 5}
            pipeline.from_pretrained.side_effect = AssertionError("Must reject metadata before loading weights")
        modules = {"accelerate": SimpleNamespace(Accelerator=TinyAccelerator),
                   "diffusers": SimpleNamespace(Flux2KleinPipeline=pipeline,
                                                 Flux2Transformer2DModel=transformer_class,
                                                 FlowMatchEulerDiscreteScheduler=Mock())}
        flags = (["--steps", "1", "--batch_size", "1", "--grad_accum", "1", "--no_ema",
                  "--num_workers", "0"] if ordinary else ["--smoke_test"])
        args = self.args(*flags, "--target_size", "32", "--optimizer", "adamw",
                         "--lr", "0.1", "--save_every", "2" if ordinary else "1",
                         "--sample_every", "2" if ordinary else "1")
        if ordinary:
            pipeline.load_config.side_effect = AssertionError("Ordinary training must not require optional metadata")
            pipe.transformer.save_pretrained.side_effect = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(trainer.os.environ, {"ACCELERATE_USE_FSDP": "false",
                                                              "ACCELERATE_USE_DEEPSPEED": "false"}))
            stack.enter_context(patch.object(trainer, "preflight_smoke_runtime"))
            # Validate on-disk metadata normally; substitute only the loaded tiny model check.
            validate_architecture = trainer.preflight_9b_architecture
            stack.enter_context(patch.object(trainer, "preflight_9b_architecture", side_effect=lambda config:
                                            None if config is pipe.transformer.config else validate_architecture(config)))
            stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats"))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=123))
            stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=456))
            stack.enter_context(patch.object(trainer, "EMAModel", side_effect=AssertionError("EMA allocated")))
            stack.enter_context(patch.object(trainer, "generate_sample", side_effect=AssertionError("sample generated")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if failure == "model_loading":
                pipeline.from_pretrained.side_effect = torch.OutOfMemoryError("synthetic load OOM")
            elif failure == "backward":
                stack.enter_context(patch.object(TinyAccelerator, "backward",
                                                side_effect=torch.OutOfMemoryError("synthetic backward OOM")))
            elif failure == "optimizer_step":
                stack.enter_context(patch.object(torch.optim.AdamW, "step",
                                                side_effect=torch.OutOfMemoryError("synthetic optimizer OOM")))
            if mutate_buffer:
                original_encode = pipe.vae.encode
                def encode(images):
                    pipe.vae.bn.num_batches_tracked += 1
                    return original_encode(images)
                stack.enter_context(patch.object(pipe.vae, "encode", side_effect=encode))
            if partial_update:
                def update_only_bias(optimizer, *args, **kwargs):
                    with torch.no_grad():
                        pipe.transformer.bias.add_(0.125)
                stack.enter_context(patch.object(torch.optim.AdamW, "step", update_only_bias))
            trainer.train(args)
        return args, pipe, pipeline, transformer_class

    def test_mocked_one_step_smoke_loop(self):
        args, pipe, pipeline, transformer_class = self.run_mocked_loop()
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertGreater(report["changed_sampled_values"], 0)
        self.assertEqual(report["peak_gpu_allocated_bytes"], 123)
        self.assertEqual(report["missing_gradients"], [])
        self.assertFalse(report["ema_enabled"])
        self.assertEqual(report["learning_rates_used"], [0.1])
        self.assertIn("torch.bfloat16", report["optimizer_state_dtypes"])
        self.assertTrue(report["frozen_components"]["vae"]["unchanged_under_checks"])
        self.assertGreater(report["frozen_components"]["vae"]["buffer_tensors_checked"], 0)
        self.assertTrue(report["frozen_components"]["text_encoder"]["unchanged_under_checks"])
        self.assertIn("backward", report["stage_memory"])
        self.assertIn("optimizer_step", report["stage_memory"])
        self.assertTrue(all(p.requires_grad for p in pipe.transformer.parameters()))
        self.assertTrue(all(not p.requires_grad for p in pipe.vae.parameters()))
        self.assertTrue(all(not p.requires_grad for p in pipe.text_encoder.parameters()))
        pipeline.load_config.assert_called_once_with(args.model_path, local_files_only=True)
        pipeline.from_pretrained.assert_called_once_with(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        transformer_class.load_config.assert_called_once_with(
            args.model_path, subfolder="transformer", local_files_only=True)
        pipe.transformer.save_pretrained.assert_not_called()
        self.assertFalse((self.root / "out" / "final").exists())

    def test_mocked_smoke_rejects_4b_metadata_before_loading_weights(self):
        with self.assertRaisesRegex(ValueError, "requires Klein Base 9B"):
            self.run_mocked_loop(invalid_metadata=True)

    def test_stage_memory_preserves_global_peak_across_resets(self):
        report = trainer.SmokeReport(self.args("--smoke_test"))
        report.device = torch.device("cuda")
        report.stage = "backward"
        with patch.object(torch.cuda, "max_memory_allocated", side_effect=[500, 200]), \
                patch.object(torch.cuda, "max_memory_reserved", side_effect=[700, 400]), \
                patch.object(torch.cuda, "reset_peak_memory_stats") as reset:
            report.start_stage("optimizer_step")
            report.write()
        reset.assert_called_once()
        self.assertEqual(report.data["peak_gpu_allocated_bytes"], 500)
        self.assertEqual(report.data["peak_gpu_reserved_bytes"], 700)
        self.assertEqual(report.data["stage_memory"]["optimizer_step"]["peak_allocated_bytes"], 200)

    def test_mocked_oom_reports_stage_and_memory_and_reraises(self):
        for stage in ["model_loading", "backward", "optimizer_step"]:
            with self.subTest(stage=stage), self.assertRaises(torch.OutOfMemoryError):
                self.run_mocked_loop(failure=stage)
            report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"], "out_of_memory")
            self.assertEqual(report["failed_stage"], stage)
            self.assertEqual(report["peak_gpu_allocated_bytes"], 123)
            self.assertEqual(report["stage_memory"][stage]["peak_reserved_bytes"], 456)

    def test_mocked_smoke_fails_when_vae_buffer_changes(self):
        with self.assertRaisesRegex(RuntimeError, "frozen component changed"):
            self.run_mocked_loop(mutate_buffer=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["frozen_components"]["vae"]["changed_buffers"], ["bn.num_batches_tracked"])

    def test_mocked_smoke_does_not_require_every_tensor_to_change(self):
        self.run_mocked_loop(partial_update=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["parameter_change_details"]["weight"]["changed_values"], 0)
        self.assertGreater(report["parameter_change_details"]["bias"]["changed_values"], 0)

    def test_ordinary_training_does_not_require_optional_metadata(self):
        _, pipe, pipeline, transformer_class = self.run_mocked_loop(ordinary=True)
        pipeline.load_config.assert_not_called()
        transformer_class.load_config.assert_not_called()
        pipe.transformer.save_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
