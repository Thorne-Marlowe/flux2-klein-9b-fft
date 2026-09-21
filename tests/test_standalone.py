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


@contextlib.contextmanager
def stub_dependencies(modules):
    # Restore only our stubs. Clearing newly imported torch/torchvision modules
    # would leave their native operator registrations alive and break later tests.
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


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
        explicit_smoke = (["--model_path", "local-base-9b", "--target_size", "32", "--optimizer", "adamw"]
                          if "--smoke_test" in extra else [])
        return trainer.parse_args(["--data_dir", str(self.data), "--output_dir",
                                   str(self.root / "out"), *explicit_smoke, *extra])

    def test_smoke_requires_explicit_model_and_size_and_preserves_normal_defaults(self):
        required = ["--data_dir", str(self.data), "--output_dir", str(self.root / "out")]
        for extra in [[], ["--model_path", "local-base-9b"], ["--target_size", "256"],
                      ["--model_path", "local-base-9b", "--target_size", "256"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args([*required, "--smoke_test", *extra])
        self.assertEqual(self.args().model_path, "black-forest-labs/FLUX.2-klein-base-4B")
        self.assertEqual(self.args().target_size, 1024)
        self.assertEqual(self.args().optimizer, "adamw8bit")

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
                        partial_update=False, invalid_metadata=False, no_update=False,
                        bad_clip=False, optimizer_name="adamw", rounded_updates=False,
                        failure_type=torch.OutOfMemoryError):
        pipe = self.tiny_pipe()
        if rounded_updates:
            with torch.no_grad():
                for p in pipe.transformer.parameters():
                    p.fill_(1.)
        pipe.vae.encode = Mock(wraps=pipe.vae.encode)
        pipeline = Mock()
        pipeline.load_config.return_value = BASE_CONFIG
        pipeline.from_pretrained.return_value = pipe
        pipeline._get_qwen3_prompt_embeds.return_value = torch.ones(1, 4, 6, dtype=torch.bfloat16)
        pipe._get_qwen3_prompt_embeds = pipeline._get_qwen3_prompt_embeds
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
        args = self.args(*flags, "--target_size", "32", "--optimizer", optimizer_name,
                         "--save_every", "2" if ordinary else "1",
                         "--sample_every", "2" if ordinary else "1")
        if ordinary:
            pipeline.load_config.side_effect = AssertionError("Ordinary training must not require optional metadata")
            pipe.transformer.save_pretrained.side_effect = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(stub_dependencies(modules))
            stack.enter_context(patch.dict(trainer.os.environ, {"ACCELERATE_USE_FSDP": "false",
                                                              "ACCELERATE_USE_DEEPSPEED": "false"}))
            stack.enter_context(patch.object(trainer, "preflight_smoke_runtime"))
            # Validate on-disk metadata normally; substitute only the loaded tiny model check.
            validate_architecture = trainer.preflight_9b_architecture
            stack.enter_context(patch.object(trainer, "preflight_9b_architecture", side_effect=lambda config:
                                            None if config is pipe.transformer.config else validate_architecture(config)))
            stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats"))
            stack.enter_context(patch.object(torch.cuda, "mem_get_info", return_value=(1024, 2048)))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="mock CUDA device"))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            stack.enter_context(patch.object(torch.cuda, "empty_cache"))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=123))
            stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=456))
            stack.enter_context(patch.object(trainer, "EMAModel", side_effect=AssertionError("EMA allocated")))
            stack.enter_context(patch.object(trainer, "generate_sample", side_effect=AssertionError("sample generated")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if failure == "model_loading":
                pipeline.from_pretrained.side_effect = failure_type("synthetic load failure")
            elif failure == "backward":
                stack.enter_context(patch.object(TinyAccelerator, "backward",
                                                side_effect=failure_type("synthetic backward failure")))
            elif failure == "optimizer_step":
                original_step = torch.optim.AdamW.step
                def step(optimizer, *args, **kwargs):
                    if any(p is pipe.transformer.weight for g in optimizer.param_groups for p in g["params"]):
                        raise failure_type("synthetic optimizer failure")
                    return original_step(optimizer, *args, **kwargs)
                stack.enter_context(patch.object(torch.optim.AdamW, "step", step))
            elif failure in ("vae_encoding", "text_encoding"):
                target = pipe.vae if failure == "vae_encoding" else pipeline
                name = "encode" if failure == "vae_encoding" else "_get_qwen3_prompt_embeds"
                if failure == "text_encoding":
                    pipe._get_qwen3_prompt_embeds.side_effect = failure_type("synthetic encoding failure")
                else:
                    stack.enter_context(patch.object(target, name, side_effect=failure_type("synthetic encoding failure")))
            if mutate_buffer:
                original_encode = pipe.vae.encode
                def encode(images):
                    pipe.vae.bn.num_batches_tracked += 1
                    return original_encode(images)
                stack.enter_context(patch.object(pipe.vae, "encode", side_effect=encode))
            if partial_update or no_update:
                original_step = torch.optim.AdamW.step
                def update_only_bias(optimizer, *args, **kwargs):
                    if not any(p is pipe.transformer.weight for g in optimizer.param_groups for p in g["params"]):
                        return original_step(optimizer, *args, **kwargs)
                    with torch.no_grad():
                        if not no_update:
                            pipe.transformer.bias.add_(0.125)
                stack.enter_context(patch.object(torch.optim.AdamW, "step", update_only_bias))
            if bad_clip:
                def clip(self, parameters, max_norm):
                    for p in parameters:
                        p.grad.fill_(float("nan"))
                stack.enter_context(patch.object(TinyAccelerator, "clip_grad_norm_", clip))
            if not ordinary:
                def prepare(self, *objects):
                    pipeline._get_qwen3_prompt_embeds.assert_called_once()
                    trainer.require_cpu(pipe.vae, "VAE before prepare")
                    trainer.require_cpu(pipe.text_encoder, "text encoder before prepare")
                    return objects
                stack.enter_context(patch.object(TinyAccelerator, "prepare", prepare))
            trainer.train(args)
        return args, pipe, pipeline, transformer_class

    def test_mocked_one_step_smoke_loop(self):
        args, pipe, pipeline, transformer_class = self.run_mocked_loop()
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["cleanup_completed"])
        self.assertTrue(report["step_checks_passed"])
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertGreater(report["changed_sampled_values"], 0)
        self.assertEqual(report["peak_gpu_allocated_bytes"], 123)
        self.assertEqual(report["missing_gradients"], [])
        self.assertFalse(report["ema_enabled"])
        self.assertEqual(report["learning_rates_used"], [3e-5])
        self.assertFalse(report["optimizer"]["configuration"][0]["foreach"])
        self.assertEqual(report["optimizer"]["configuration"][0]["weight_decay"], 0.01)
        self.assertTrue(report["optimizer_step_completed"])
        self.assertTrue(report["optimizer_step_returned"])
        self.assertTrue(report["optimizer_step_evidence"]["sufficient"])
        self.assertIn("gradients_after_clip", report)
        self.assertGreater(report["memory_after_step"]["optimizer_state"]["total_bytes"], 0)
        self.assertEqual(report["optimizer_details_after_step"]["states_by_parameter"]["weight"]["step"], 1)
        self.assertEqual(report["gpu_at_start"]["capacity_bytes"], 2048)
        self.assertEqual(report["seed"], 0)
        pipe.vae.encode.assert_called_once()
        pipeline._get_qwen3_prompt_embeds.assert_called_once()
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
                patch.object(torch.cuda, "mem_get_info", return_value=(1024, 2048)), \
                patch.object(torch.cuda, "reset_peak_memory_stats") as reset:
            report.start_stage("optimizer_step")
            report.write()
        reset.assert_called_once()
        self.assertEqual(report.data["peak_gpu_allocated_bytes"], 500)
        self.assertEqual(report.data["peak_gpu_reserved_bytes"], 700)
        self.assertEqual(report.data["stage_memory"]["optimizer_step"]["peak_allocated_bytes"], 200)

    def test_mocked_oom_reports_stage_and_memory_and_reraises(self):
        for stage in ["model_loading", "vae_encoding", "text_encoding", "backward", "optimizer_step"]:
            with self.subTest(stage=stage), self.assertRaises(torch.OutOfMemoryError):
                self.run_mocked_loop(failure=stage)
            report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"], "out_of_memory")
            self.assertEqual(report["failed_stage"], stage)
            self.assertEqual(report["peak_gpu_allocated_bytes"], 123)
            self.assertEqual(report["stage_memory"][stage]["peak_reserved_bytes"], 456)

    def test_unexpected_exceptions_replace_stale_success_and_identify_stage(self):
        path = self.root / "out" / "smoke_diagnostics.json"
        path.parent.mkdir(exist_ok=True)
        for stage in ("model_loading", "vae_encoding", "text_encoding", "backward", "optimizer_step"):
            path.write_text('{"passed": true, "stale": true}')
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "synthetic"):
                self.run_mocked_loop(failure=stage, failure_type=ValueError)
            report = json.loads(path.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_stage"], stage)
            self.assertEqual(report["exception_type"], "ValueError")
            self.assertIn("synthetic", report["error_message"])
            self.assertIn("ValueError", report["traceback"])
            self.assertNotIn("stale", report)

    def test_preflight_and_import_failures_get_fresh_reports(self):
        for stage, target, error in (
            ("preflight", "preflight_dataset", ValueError("bad dataset")),
            ("dependency_imports", None, ImportError("missing accelerate")),
        ):
            with contextlib.ExitStack() as stack:
                if target:
                    stack.enter_context(patch.object(trainer, target, side_effect=error))
                else:
                    import builtins
                    original_import = builtins.__import__
                    def importing(name, *args, **kwargs):
                        if name == "accelerate":
                            raise error
                        return original_import(name, *args, **kwargs)
                    stack.enter_context(patch.object(builtins, "__import__", importing))
                with self.subTest(stage=stage), self.assertRaises(type(error)):
                    self.run_mocked_loop()
            report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["failed_stage"], stage)
            self.assertEqual(report["exception_type"], type(error).__name__)

    def test_cleanup_failure_cannot_publish_success(self):
        path = self.root / "out" / "smoke_diagnostics.json"
        def fail_cleanup(*args):
            self.assertFalse(json.loads(path.read_text())["passed"])
            raise RuntimeError("synthetic cleanup failure")
        for method in ("wait_for_everyone", "close"):
            with contextlib.ExitStack() as stack:
                stack.enter_context(self.subTest(method=method))
                if method == "close":
                    # Avoid making tqdm's destructor raise the injected exception.
                    progress = Mock()
                    progress.close.side_effect = fail_cleanup
                    stack.enter_context(patch.object(trainer, "tqdm", return_value=progress))
                else:
                    stack.enter_context(patch.object(TinyAccelerator, method, side_effect=fail_cleanup))
                with self.assertRaisesRegex(RuntimeError, "synthetic cleanup failure"):
                    self.run_mocked_loop()
            report = json.loads(path.read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["cleanup_completed"])
            self.assertTrue(report["optimizer_step_completed"])
            self.assertTrue(report["step_checks_passed"])
            self.assertEqual(report["failed_stage"], "cleanup")

    def test_memory_query_failure_does_not_hide_original_exception(self):
        args = self.args("--smoke_test", "--target_size", "32", "--optimizer", "adamw")
        error = ValueError("original failure")
        with patch.object(trainer, "_train", side_effect=error), \
             patch.object(trainer.SmokeReport, "measure", side_effect=RuntimeError("CUDA query failed")):
            with self.assertRaises(ValueError) as caught:
                trainer.train(args)
        self.assertIs(caught.exception, error)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["exception_type"], "ValueError")
        self.assertIn("CUDA query failed", report["memory_reporting_error"])

    def test_unwritable_report_preserves_original_exception(self):
        args = self.args("--smoke_test", "--target_size", "32", "--optimizer", "adamw")
        error = PermissionError("output denied")
        with patch.object(trainer.SmokeReport, "write", side_effect=error), \
             patch.object(trainer, "_train") as run, contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(PermissionError) as caught:
                trainer.train(args)
        self.assertIs(caught.exception, error)
        run.assert_not_called()
        self.assertIn("Failed to write smoke diagnostics", stderr.getvalue())

    def test_mocked_smoke_fails_when_vae_buffer_changes(self):
        with self.assertRaisesRegex(RuntimeError, "frozen component changed"):
            self.run_mocked_loop(mutate_buffer=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["frozen_encoding_checks"]["vae"]["changed_buffers"], ["bn.num_batches_tracked"])

    def test_weight_change_alone_is_insufficient_step_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "insufficient optimizer-step evidence"):
            self.run_mocked_loop(partial_update=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertFalse(report["passed"])
        self.assertFalse(report["optimizer_step_completed"])
        self.assertEqual(report["parameter_change_details"]["weight"]["changed_values"], 0)
        self.assertGreater(report["parameter_change_details"]["bias"]["changed_values"], 0)

    def test_ordinary_training_does_not_require_optional_metadata(self):
        _, pipe, pipeline, transformer_class = self.run_mocked_loop(ordinary=True)
        pipeline.load_config.assert_not_called()
        transformer_class.load_config.assert_not_called()
        pipe.transformer.save_pretrained.assert_called_once()

    def test_staged_encoding_moves_pipeline_components_sequentially(self):
        pipe = self.tiny_pipe()
        pipe.vae.requires_grad_(False).eval()
        pipe.text_encoder.requires_grad_(False).eval()
        events = []
        latents = torch.ones(1, 1, 4, 4, dtype=torch.bfloat16)
        embeds = torch.ones(1, 4, 6, dtype=torch.bfloat16)
        def encode(*args, **kwargs):
            events.append("encode image")
            return latents
        def prompt(**kwargs):
            events.append("encode caption")
            return embeds
        pipe._get_qwen3_prompt_embeds = Mock(side_effect=prompt)
        smoke = Mock()
        smoke.data = {}
        with patch.object(pipe.transformer, "to", side_effect=AssertionError("transformer moved during encoding")), \
                patch.object(pipe.vae, "to", side_effect=lambda device, **kwargs: events.append(f"vae {device}") or pipe.vae), \
                patch.object(pipe.text_encoder, "to", side_effect=lambda device, **kwargs: events.append(f"text {device}") or pipe.text_encoder), \
                patch.object(trainer, "encode_images_klein", side_effect=encode) as image_encoder, \
                patch.object(torch.cuda, "empty_cache"):
            result = trainer.stage_smoke_encoding(pipe, trainer.ImageTextDataset(self.data, fixed_size=True, target_size=32),
                                                 torch.device("cuda"), torch.bfloat16, smoke)
        self.assertEqual(events, ["vae cuda", "encode image", "vae cpu", "text cuda", "encode caption", "text cpu"])
        self.assertIs(result[0], latents)
        self.assertIs(result[1], embeds)
        image_encoder.assert_called_once()
        pipe._get_qwen3_prompt_embeds.assert_called_once()
        self.assertTrue(all(c["unchanged_under_checks"] for c in smoke.data["frozen_encoding_checks"].values()))

    def test_tensor_and_training_memory_match_tensor_bytes(self):
        a = torch.zeros(3, dtype=torch.bfloat16)
        b = torch.zeros(2, dtype=torch.float32)
        report = trainer.tensor_memory([a, b, a, None])
        self.assertEqual(report["total_bytes"], 14)
        self.assertEqual(report["by_device_dtype"]["cpu/torch.bfloat16"]["bytes"], 6)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
        model(torch.ones(1, 2)).sum().backward()
        self.assertEqual(trainer.training_memory(model, optimizer)["optimizer_state"]["total_bytes"], 0)
        optimizer.step()
        report = trainer.training_memory(model, optimizer)
        self.assertEqual(report["parameters"]["total_bytes"], 12)
        self.assertEqual(report["gradients"]["total_bytes"], 12)
        self.assertEqual(report["optimizer_state"]["total_bytes"], 32)
        self.assertEqual(report["total_bytes"], 56)
        details = trainer.optimizer_details(model, optimizer)
        self.assertEqual(details["states_by_parameter"]["weight"]["step"], 1)
        self.assertEqual(details["states_by_parameter"]["weight"]["keys"], ["exp_avg", "exp_avg_sq", "step"])

    def test_default_lr_precision_probe_detects_rounding_without_mutation(self):
        samples = {"weight": torch.tensor([0.001, 0.01, 1.])}
        before = samples["weight"].clone()
        probe = trainer.precision_probe(samples, 3e-5, 0.01)
        self.assertEqual(len(probe["results"]), 1)
        result = probe["results"][0]
        self.assertEqual(result["lr"], 3e-5)
        self.assertGreater(result["lost_on_bf16_storage_cast"], 0)
        self.assertGreater(result["unchanged_in_bf16_optimizer"], 0)
        self.assertGreater(result["fp32_updates"], result["lost_on_bf16_storage_cast"])
        torch.testing.assert_close(samples["weight"], before)
        self.assertEqual([r["lr"] for r in trainer.precision_probe(samples, 1e-5, 0.0)["results"]], [3e-5, 1e-5])

    def test_noop_step_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, "insufficient optimizer-step evidence"):
            self.run_mocked_loop(no_update=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertTrue(report["optimizer_step_returned"])
        self.assertFalse(report["optimizer_step_completed"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["passed"])
        self.assertFalse(report["optimizer_step_evidence"]["sufficient"])

    def test_real_bf16_step_can_pass_without_stored_weight_changes(self):
        self.run_mocked_loop(rounded_updates=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertTrue(report["optimizer_step_completed"])
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertFalse(report["stored_weight_changes_observed"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["learning_rates_used"], [3e-5])

    def test_adamw_evidence_checks_real_counters_and_rejects_noop_on_existing_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = trainer.optimizer_step_snapshot(model, optimizer, "adamw")
        self.assertFalse(before["parameters"]["weight"]["state_initialized"])
        self.assertEqual(len(optimizer.state), 0)
        optimizer.step()
        after = trainer.optimizer_step_snapshot(model, optimizer, "adamw")
        self.assertTrue(trainer.verify_optimizer_step(before, after, False)["sufficient"])
        self.assertIsNone(before["parameters"]["weight"]["step"])  # Snapshot cannot alias the live counter.
        self.assertFalse(trainer.verify_optimizer_step(after, after, False)["sufficient"])
        self.assertFalse(trainer.verify_optimizer_step(before, after, True)["sufficient"])
        optimizer.step()
        second = trainer.optimizer_step_snapshot(model, optimizer, "adamw")
        self.assertTrue(trainer.verify_optimizer_step(after, second, False)["sufficient"])
        self.assertFalse(trainer.verify_optimizer_step(before, second, False)["sufficient"])

    def test_missing_malformed_and_partial_optimizer_evidence_fails_closed(self):
        for problem in ("counter", "moments", "counter_only", "partial", "bad_shape", "nan", "vector", "fraction"):
            with self.subTest(problem=problem):
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
                for p in model.parameters():
                    p.grad = torch.ones_like(p)
                before = trainer.optimizer_step_snapshot(model, optimizer, "adamw")
                optimizer.step()
                state = optimizer.state[model.weight]
                if problem == "counter":
                    del state["step"]
                elif problem == "moments":
                    del state["exp_avg_sq"]
                elif problem == "counter_only":
                    optimizer.state[model.weight] = {"step": 1}
                elif problem == "partial":
                    optimizer.state[model.bias].clear()
                elif problem == "bad_shape":
                    state["exp_avg"] = torch.zeros(1)
                else:
                    state["step"] = {"nan": float("nan"), "vector": torch.ones(2), "fraction": 1.5}[problem]
                after = trainer.optimizer_step_snapshot(model, optimizer, "adamw")
                self.assertFalse(trainer.verify_optimizer_step(before, after, False)["sufficient"])

    def test_unknown_optimizer_schema_is_explicitly_unsupported(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = trainer.optimizer_step_snapshot(model, optimizer, "sgd")
        evidence = trainer.verify_optimizer_step(before, before, False)
        self.assertFalse(evidence["schema_supported"])
        self.assertFalse(evidence["sufficient"])
        self.assertEqual(evidence["reason"], "unsupported optimizer schema")

    def test_adamw8bit_documented_state_schema_fixture_not_cuda_execution(self):
        # Actual bitsandbytes is unavailable. These are documented Optimizer2State
        # layouts, testing only parsing/transition checks, not an optimizer kernel.
        for dtype in (torch.uint8, torch.float32):
            with self.subTest(dtype=dtype):
                model = torch.nn.Linear(2, 1)
                optimizer = SimpleNamespace(param_groups=[{"params": list(model.parameters())}], state={})
                for p in model.parameters():
                    p.grad = torch.ones_like(p)
                before = trainer.optimizer_step_snapshot(model, optimizer, "adamw8bit")
                for p in model.parameters():
                    optimizer.state[p] = {"state1": torch.zeros_like(p, dtype=dtype),
                                          "state2": torch.zeros_like(p, dtype=dtype), "step": 1}
                after = trainer.optimizer_step_snapshot(model, optimizer, "adamw8bit")
                self.assertTrue(trainer.verify_optimizer_step(before, after, False)["sufficient"])
                self.assertFalse(trainer.verify_optimizer_step(after, after, False)["sufficient"])
                del optimizer.state[model.weight]["step"]
                missing = trainer.optimizer_step_snapshot(model, optimizer, "adamw8bit")
                self.assertFalse(trainer.verify_optimizer_step(before, missing, False)["sufficient"])

    def test_bad_post_clip_gradients_fail_before_optimizer_step(self):
        with self.assertRaisesRegex(RuntimeError, "after clipping"):
            self.run_mocked_loop(bad_clip=True)
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["gradients_before_clip"]["nonfinite_gradients"], [])
        self.assertTrue(report["gradients_after_clip"]["nonfinite_gradients"])

    def test_smoke_adafactor_reports_effective_configuration(self):
        try:
            from transformers.optimization import Adafactor
        except ImportError:
            self.skipTest("Transformers Adafactor not installed")
        self.run_mocked_loop(optimizer_name="adafactor")
        report = json.loads((self.root / "out" / "smoke_diagnostics.json").read_text())
        self.assertEqual(report["optimizer"]["configuration"][0]["weight_decay"], 0.01)
        self.assertFalse(report["optimizer"]["configuration"][0]["relative_step"])
        self.assertEqual(report["optimizer_details_after_step"]["states_by_parameter"]["weight"]["step"], 1)
        self.assertIn("exp_avg_sq_row", report["optimizer_details_after_step"]["states_by_parameter"]["weight"]["keys"])
        self.assertTrue(report["optimizer_step_evidence"]["sufficient"])
        after = report["optimizer_step_evidence"]["after"]["parameters"]
        self.assertIn("exp_avg_sq_col", after["weight"]["state_keys"])
        self.assertIn("exp_avg_sq", after["bias"]["state_keys"])

    def test_missing_dependency_versions_are_explicit(self):
        with patch.object(trainer.importlib.metadata, "version", side_effect=trainer.importlib.metadata.PackageNotFoundError):
            versions = trainer.dependency_versions()
        self.assertIsNone(versions["diffusers"])
        self.assertEqual(versions["torch"], torch.__version__)


if __name__ == "__main__":
    unittest.main()
