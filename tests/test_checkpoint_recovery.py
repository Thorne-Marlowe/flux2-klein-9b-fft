"""Real filesystem contract tests; opaque fixture bytes do not prove restoration."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.klein_checkpoint import (
    CheckpointCompatibilityError, CheckpointValidationError, LegacyCheckpointError,
    validate_checkpoint, validate_configuration_compatibility,
    validate_object_compatibility,
)


class CheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "arbitrary-name"
        self.root.mkdir()
        self.manifest = {
            "format": "klein-recovery", "schema_version": 1, "checkpoint_id": "checkpoint-a",
            "run_id": "run-a", "parent_checkpoint_id": None, "created_at": "2026-09-21T12:00:00Z",
            "qualification": "unqualified",
            "progress": {"completed_optimizer_steps": 2, "attempted_optimizer_steps": 3,
                         "skipped_optimizer_steps": 1, "loop_iterations": 3, "epoch": 0,
                         "next_batch_index": 3, "accumulation_step": 0},
            "configuration": {
                "model_fingerprint": "a" * 64, "dataset_fingerprint": "b" * 64,
                "preprocessing_fingerprint": "c" * 64, "dataset_size": 4,
                "world_size": 1, "batch_size": 1, "gradient_accumulation_steps": 1,
                "num_workers": 0, "precision": "bf16", "use_cached_latents": False,
                "drop_last": True, "sampling_enabled": False, "optimizer": "adafactor",
                "optimizer_options": {"lr": 3e-5, "weight_decay": 0.0, "relative_step": False},
                "total_optimizer_steps": 4, "warmup_steps": 0,
                "schedule_policy": "standalone_warmup_cosine_v1",
                "ema_enabled": False, "ema_decay": 0.9999, "seed": 0,
            },
            "environment": {
                "python": "3.12.12", "trainer_commit": "1bb55ea", "cuda": "12.8",
                "platform": "linux-x86_64", "device_type": "cuda", "device_name": "fixture GPU",
                "backend_settings": {"deterministic_algorithms": True, "cudnn_benchmark": False,
                                     "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False,
                                     "cudnn_allow_tf32": False, "float32_matmul_precision": "highest"},
                "dependencies": {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128",
                                 "transformers": "5.3.0", "diffusers": "0.37.0",
                                 "accelerate": "1.12.0", "safetensors": "0.7.0",
                                 "numpy": "2.5.3", "pillow": "12.1.1"},
            },
            "objects": {
                "model_class": "diffusers.Flux2Transformer2DModel",
                "model_tensors": [
                    {"name": "weight", "shape": [2, 3], "dtype": "bfloat16", "kind": "parameter"},
                    {"name": "counter", "shape": [], "dtype": "int64", "kind": "buffer"},
                ],
                "optimizer_class": "transformers.Adafactor", "parameter_groups": [["weight"]],
                "scheduler_class": "torch.optim.lr_scheduler.CosineAnnealingLR",
            },
            "payloads": [],
        }
        self.payload("model/config.json", b'{"_class_name":"Flux2Transformer2DModel"}')
        self.payload("model/diffusion_pytorch_model.safetensors", b"opaque model bytes")
        self.payload("trainer.json", b'{}')
        for name in ("optimizer.pt", "scheduler.pt", "rng.pt", "data_order.pt"):
            self.payload(name, b"opaque fixture " + name.encode())
        self.write_manifest()

    def payload(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.manifest["payloads"] = [p for p in self.manifest["payloads"] if p["path"] != name]
        self.manifest["payloads"].append({"path": name, "size_bytes": len(data),
                                          "sha256": hashlib.sha256(data).hexdigest()})

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def reject(self, pattern=None):
        self.write_manifest()
        with self.assertRaisesRegex(CheckpointValidationError, pattern or "."):
            validate_checkpoint(self.root)

    def test_valid_root_renamed_and_counters_not_inferred_from_name(self):
        validated = validate_checkpoint(self.root)
        self.assertEqual(validated.manifest["progress"]["completed_optimizer_steps"], 2)
        self.assertEqual(validated.manifest["progress"]["loop_iterations"], 3)
        renamed = self.root.with_name("checkpoint-999999")
        self.root.rename(renamed)
        moved = validate_checkpoint(renamed)
        self.assertEqual(moved.manifest, validated.manifest)
        self.assertEqual(moved.qualification, "unqualified")
        with self.assertRaises(TypeError):
            moved.manifest["progress"]["completed_optimizer_steps"] = 100

    def test_validation_does_not_change_files(self):
        before = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        validate_checkpoint(self.root)
        after = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_file_and_missing_required_entry(self):
        (self.root / "optimizer.pt").unlink()
        self.reject("inventory mismatch")
        self.manifest["payloads"] = [p for p in self.manifest["payloads"] if p["path"] != "optimizer.pt"]
        self.reject("Missing required payloads")

    def test_truncated_and_same_size_corruption(self):
        path = self.root / "rng.pt"
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        self.reject("size mismatch")
        path.write_bytes(b"X" * len(original))
        self.reject("checksum mismatch")

    def test_duplicate_unexpected_and_case_colliding_payloads(self):
        original = copy.deepcopy(self.manifest["payloads"])
        for name in (original[0]["path"], original[0]["path"].upper(), "extra.pt", "manifest.json"):
            self.manifest["payloads"] = copy.deepcopy(original)
            self.manifest["payloads"].append({"path": name, "size_bytes": 1, "sha256": "a" * 64})
            with self.subTest(name=name):
                self.reject()

    def test_unlisted_files_and_export_directories_rejected(self):
        extra = self.root / "extra.pt"
        extra.write_bytes(b"extra")
        self.reject("unexpected")
        extra.unlink()
        (self.root / "exports").mkdir()
        self.reject("Unexpected checkpoint directory")

    def test_unsafe_relative_paths(self):
        original = self.manifest["payloads"][0]["path"]
        for name in ("../outside", "/absolute", "C:/drive", "C:relative", "model\\config.json",
                     "model/../rng.pt", "model//config.json", "./rng.pt", "rng.pt:stream", "rng.pt ", "x\x00y"):
            self.manifest["payloads"][0]["path"] = name
            with self.subTest(name=name):
                self.reject("Unsafe payload path")
        self.manifest["payloads"][0]["path"] = original

    def test_unknown_schema_versions_and_format(self):
        for version in (0, 3, "1", True):
            self.manifest["schema_version"] = version
            with self.subTest(version=version):
                self.reject("Unsupported checkpoint format/schema_version")
        self.manifest["schema_version"] = 1
        self.manifest["format"] = "inference-export"
        self.reject("Unsupported checkpoint format/schema_version")

    def test_malformed_json_duplicate_keys_and_nonfinite_numbers(self):
        path = self.root / "manifest.json"
        for text in ('{', '[]', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(CheckpointValidationError):
                validate_checkpoint(self.root)

    def test_missing_extra_and_malformed_manifest_fields(self):
        original = copy.deepcopy(self.manifest)
        cases = [("checkpoint_id", ""), ("created_at", "2026-09-21T00:00:00"),
                 ("payloads", {}), ("configuration", None), ("objects", []), ("qualification", "exact")]
        for key, value in cases:
            self.manifest = copy.deepcopy(original)
            self.manifest[key] = value
            with self.subTest(key=key):
                self.reject()
        self.manifest = copy.deepcopy(original)
        del self.manifest["progress"]
        self.reject()
        self.manifest = copy.deepcopy(original)
        self.manifest["unexpected"] = True
        self.reject()

    def test_legacy_and_wrong_root_rejected(self):
        with self.assertRaises(LegacyCheckpointError):
            validate_checkpoint(self.root / "model")
        (self.root / "accelerator_state").mkdir()
        with self.assertRaises(LegacyCheckpointError):
            validate_checkpoint(self.root / "accelerator_state")
        (self.root / "manifest.json").unlink()
        with self.assertRaises(LegacyCheckpointError):
            validate_checkpoint(self.root)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint(self.root / "does-not-exist")

    def test_complete_manifest_in_staging_directory_is_not_committed(self):
        staging = self.root.with_name(".incomplete-a")
        self.root.rename(staging)
        with self.assertRaisesRegex(CheckpointValidationError, "Incomplete staging"):
            validate_checkpoint(staging)

    def test_payload_symlink_rejected(self):
        outside = self.root.parent / "outside.bin"
        outside.write_bytes((self.root / "rng.pt").read_bytes())
        (self.root / "rng.pt").unlink()
        try:
            (self.root / "rng.pt").symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"OS does not permit symlink creation: {error}")
        self.reject("Linked checkpoint entry")

    def test_unsupported_configuration_and_counter_consistency(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (("world_size", 2), ("batch_size", 2), ("gradient_accumulation_steps", 2),
                           ("num_workers", 1), ("use_cached_latents", True), ("sampling_enabled", True),
                           ("optimizer", "adamw8bit"), ("precision", "fp32"), ("world_size", True)):
            self.manifest = copy.deepcopy(original)
            self.manifest["configuration"][key] = value
            with self.subTest(key=key):
                self.reject("Unsupported recovery")
        for key, value in (("completed_optimizer_steps", 3), ("loop_iterations", 2), ("epoch", 1),
                           ("next_batch_index", 4), ("accumulation_step", 1), ("skipped_optimizer_steps", -1)):
            self.manifest = copy.deepcopy(original)
            self.manifest["progress"][key] = value
            with self.subTest(key=key):
                self.reject()

    def test_normalized_epoch_boundary(self):
        self.manifest["progress"].update(completed_optimizer_steps=3, attempted_optimizer_steps=4,
                                         loop_iterations=4, epoch=1, next_batch_index=0)
        self.write_manifest()
        validate_checkpoint(self.root)

    def test_attempt_horizon_applies_even_when_updates_are_skipped(self):
        # The old completed-only check accepted this impossible trainer position.
        self.manifest["progress"].update(completed_optimizer_steps=2,
                                         attempted_optimizer_steps=5,
                                         skipped_optimizer_steps=3,
                                         loop_iterations=5, epoch=1, next_batch_index=1)
        self.reject("Inconsistent optimizer/loop counters")

    def test_all_skipped_attempts_can_reach_the_loop_limit(self):
        self.manifest["progress"].update(completed_optimizer_steps=0,
                                         attempted_optimizer_steps=4,
                                         skipped_optimizer_steps=4,
                                         loop_iterations=4, epoch=1, next_batch_index=0)
        self.write_manifest()
        checked = validate_checkpoint(self.root)
        self.assertEqual(checked.manifest["progress"]["completed_optimizer_steps"], 0)
        self.assertEqual(checked.qualification, "unqualified")

    def test_warmup_must_be_resolved_using_the_actual_trainer_cap(self):
        self.manifest["configuration"]["warmup_steps"] = 1
        self.reject("resolved warmup")  # steps=4 resolves any positive request to 0.
        self.manifest["configuration"].update(total_optimizer_steps=20, warmup_steps=2)
        self.write_manifest()
        validate_checkpoint(self.root)
        self.manifest["configuration"]["warmup_steps"] = 3
        self.reject("resolved warmup")

    def test_native_cosine_state_does_not_track_manual_warmup_lr(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable for native CPU schedule evidence")
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=3e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=18, eta_min=3e-6)
        used_lrs = []
        for attempt in range(3):
            used_lrs.append(optimizer.param_groups[0]["lr"])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            if attempt < 2:
                optimizer.param_groups[0]["lr"] = 3e-5 * (attempt + 1) / 2
            else:
                scheduler.step()
            optimizer.zero_grad()
            if attempt == 0:
                self.assertEqual(scheduler.state_dict()["_last_lr"], [3e-5])
                self.assertEqual(optimizer.param_groups[0]["lr"], 1.5e-5)
        self.assertEqual(used_lrs, [3e-5, 1.5e-5, 3e-5])
        self.assertEqual(scheduler.state_dict()["last_epoch"], 1)
        self.assertEqual(scheduler.state_dict()["_step_count"], 2)

    def test_native_adafactor_load_casts_state_and_cannot_preserve_precision(self):
        # Evidence for the future dtype-preserving restore requirement, not a
        # recovery implementation or a test of Accelerate/CUDA behavior.
        try:
            import torch
            from transformers.optimization import Adafactor
        except ImportError:
            self.skipTest("Torch/Transformers unavailable for native CPU evidence")
        parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
        optimizer = Adafactor([parameter], lr=3e-5, relative_step=False,
                              scale_parameter=False)
        parameter.grad = torch.full_like(parameter, 0.3)
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        moments = ("exp_avg_sq_row", "exp_avg_sq_col", "RMS")
        saved_state = next(iter(saved["state"].values()))
        for key in moments:
            self.assertEqual(saved_state[key].dtype, torch.float32)
        optimizer.load_state_dict(saved)
        for key in moments:
            self.assertEqual(optimizer.state[parameter][key].dtype, torch.bfloat16)
        self.assertTrue(any(not torch.equal(saved_state[key],
                                           optimizer.state[parameter][key].float())
                            for key in moments))

    def test_ema_payload_required_iff_enabled(self):
        self.manifest["configuration"]["ema_enabled"] = True
        self.reject("incomplete ema")
        self.payload("ema/weights.safetensors", b"opaque ema")
        self.write_manifest()
        validate_checkpoint(self.root)
        self.manifest["configuration"]["ema_enabled"] = False
        self.reject("Unexpected EMA")

    def test_sharded_weights_and_index_coverage(self):
        old = "model/diffusion_pytorch_model.safetensors"
        (self.root / old).unlink()
        self.manifest["payloads"] = [p for p in self.manifest["payloads"] if p["path"] != old]
        first = "diffusion_pytorch_model-00001-of-00002.safetensors"
        second = "diffusion_pytorch_model-00002-of-00002.safetensors"
        self.payload("model/" + first, b"first")
        self.payload("model/" + second, b"second")
        index = {"metadata": {}, "weight_map": {"weight": first, "counter": second}}
        index_path = "model/diffusion_pytorch_model.safetensors.index.json"
        self.payload(index_path, json.dumps(index).encode())
        self.write_manifest()
        validate_checkpoint(self.root)
        for mapping in ({"weight": first}, {"weight": first, "counter": "../outside"},
                        {"weight": first, "counter": first}):
            index["weight_map"] = mapping
            self.payload(index_path, json.dumps(index).encode())
            with self.subTest(mapping=mapping):
                self.reject()

    def test_configuration_and_constructed_objects_are_separate_checks(self):
        checked = validate_checkpoint(self.root)
        validate_configuration_compatibility(checked, configuration=self.manifest["configuration"],
                                              environment=self.manifest["environment"])
        validate_object_compatibility(checked, self.manifest["objects"])
        config = copy.deepcopy(self.manifest["configuration"])
        config["total_optimizer_steps"] = 10
        with self.assertRaisesRegex(CheckpointCompatibilityError, "configuration differs"):
            validate_configuration_compatibility(checked, configuration=config, environment=self.manifest["environment"])
        environment = copy.deepcopy(self.manifest["environment"])
        environment["dependencies"]["torch"] = "different"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "environment differs"):
            validate_configuration_compatibility(checked, configuration=self.manifest["configuration"], environment=environment)
        objects = copy.deepcopy(self.manifest["objects"])
        objects["model_tensors"][0]["shape"] = [3, 2]
        with self.assertRaisesRegex(CheckpointCompatibilityError, "training objects differ"):
            validate_object_compatibility(checked, objects)

    def test_compatibility_does_not_equate_booleans_and_integers(self):
        checked = validate_checkpoint(self.root)
        configuration = copy.deepcopy(self.manifest["configuration"])
        configuration["world_size"] = True
        with self.assertRaises(CheckpointCompatibilityError):
            validate_configuration_compatibility(checked, configuration=configuration,
                                                  environment=self.manifest["environment"])

    def test_invalid_payload_sizes_and_digests(self):
        original = copy.deepcopy(self.manifest["payloads"])
        for key, value in (("size_bytes", True), ("size_bytes", -1), ("size_bytes", "12"),
                           ("sha256", "not-a-hash"), ("sha256", "A" * 64)):
            self.manifest["payloads"] = copy.deepcopy(original)
            self.manifest["payloads"][0][key] = value
            with self.subTest(key=key, value=value):
                self.reject()

    def test_invalid_tensor_metadata_and_optimizer_membership(self):
        original = copy.deepcopy(self.manifest["objects"])
        for key, value in (("dtype", []), ("dtype", "float32"), ("shape", [-1]), ("kind", "unknown")):
            self.manifest["objects"] = copy.deepcopy(original)
            self.manifest["objects"]["model_tensors"][0][key] = value
            with self.subTest(key=key):
                self.reject()
        for groups in ([["weight", "weight"]], [["counter"]], [["unknown"]], []):
            self.manifest["objects"] = copy.deepcopy(original)
            self.manifest["objects"]["parameter_groups"] = groups
            with self.subTest(groups=groups):
                self.reject()
        self.manifest["objects"] = copy.deepcopy(original)
        self.manifest["objects"]["optimizer_class"] = "torch.optim.AdamW"
        self.reject("conflicting training object class")

    def test_parameter_group_order_is_compatibility_relevant(self):
        self.manifest["objects"]["model_tensors"].append(
            {"name": "bias", "shape": [2], "dtype": "bfloat16", "kind": "parameter"})
        self.manifest["objects"]["parameter_groups"] = [["weight", "bias"]]
        self.write_manifest()
        checked = validate_checkpoint(self.root)
        objects = copy.deepcopy(self.manifest["objects"])
        objects["parameter_groups"] = [["bias", "weight"]]
        with self.assertRaises(CheckpointCompatibilityError):
            validate_object_compatibility(checked, objects)

    def test_payload_json_schema_cannot_be_bypassed_with_updated_hash(self):
        self.payload("trainer.json", b"invalid JSON")
        self.reject("Cannot read JSON")
        self.payload("trainer.json", b"{}")
        self.payload("model/config.json", b'{"_class_name":"OtherModel"}')
        self.reject("Conflicting model/config.json class")


class RecoveryCodecTests(unittest.TestCase):
    """Native CPU codecs; not prepared-Accelerate or CUDA qualification."""

    def setUp(self):
        import torch
        from scripts import klein_checkpoint as ck
        self.torch, self.ck = torch, ck
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        self.model.register_buffer("counter", torch.tensor(7, dtype=torch.int64))

    def optimizer(self, kind="adamw", **options):
        groups = [{"params": [self.model.bias], "weight_decay": 0.04},
                  {"params": [self.model.weight], "weight_decay": 0.02}]
        if kind == "adamw":
            return self.torch.optim.AdamW(groups, lr=3e-5, foreach=False, **options)
        from transformers.optimization import Adafactor
        return Adafactor(groups, lr=3e-5, relative_step=False, scale_parameter=False, **options)

    def step(self, optimizer):
        for p in self.model.parameters():
            p.grad = self.torch.arange(1, p.numel() + 1).reshape(p.shape).to(p) * 0.137
        optimizer.step()
        optimizer.zero_grad()

    def roundtrip(self, filename, state, **context):
        path = self.root / filename
        checkpoint_id = context.get("manifest", {}).get("checkpoint_id", "test-a")
        self.ck.write_state_payload(path, checkpoint_id, state, **context)
        return self.ck.read_state_payload(path, checkpoint_id, **context)

    def test_model_shards_parameters_buffers_and_parameter_identity(self):
        expected = copy.deepcopy(self.model.state_dict())
        identities = [id(p) for p in self.model.parameters()]
        path = self.root / "model"
        files = self.ck.write_model_state(path, self.model, max_shard_bytes=12)
        self.assertGreater(len(files), 2)
        with self.torch.no_grad():
            for t in self.model.state_dict().values():
                t.zero_()
        self.ck.restore_model_state(path, self.model, expected_inventory=self.ck.tensor_inventory(self.model))
        self.assertEqual(identities, [id(p) for p in self.model.parameters()])
        self.assertTrue(self.ck._equal_state(expected, self.model.state_dict()))

    def test_model_oversize_fails_before_writing(self):
        path = self.root / "model"
        with self.assertRaisesRegex(CheckpointValidationError, "budget"):
            self.ck.write_model_state(path, self.model, max_shard_bytes=1)
        self.assertFalse(path.exists())

    def test_mapping_model_config_is_serialized_without_to_dict(self):
        from types import MappingProxyType
        self.model.config = MappingProxyType({"in_features": 3, "out_features": 2})
        path = self.root / "model"
        self.ck.write_model_state(path, self.model)
        config = json.loads((path / "config.json").read_text())
        self.assertEqual(config, {"_class_name": "Linear", "in_features": 3, "out_features": 2})

    def test_nonpersistent_buffer_is_included_in_recovery(self):
        self.model.register_buffer("runtime_buffer", self.torch.tensor([4.0]), persistent=False)
        path = self.root / "model"
        self.ck.write_model_state(path, self.model)
        self.model.runtime_buffer.zero_()
        self.ck.restore_model_state(path, self.model)
        self.assertEqual(self.model.runtime_buffer.item(), 4.0)

    def test_aliased_weights_are_explicitly_unsupported(self):
        self.model.register_buffer("alias", self.model.weight.detach())
        with self.assertRaisesRegex(CheckpointValidationError, "Aliased"):
            self.ck.write_model_state(self.root / "model", self.model)

    def test_model_schema_mismatch_in_later_shard_does_not_mutate(self):
        from safetensors.torch import save_file
        path = self.root / "model"
        self.ck.write_model_state(path, self.model, max_shard_bytes=12)
        index = json.loads((path / "diffusion_pytorch_model.safetensors.index.json").read_text())
        filename = index["weight_map"]["counter"]
        save_file({"counter": self.torch.ones(2, dtype=self.torch.int64)}, str(path / filename))
        with self.torch.no_grad():
            for t in self.model.state_dict().values():
                t.zero_()
        before = copy.deepcopy(self.model.state_dict())
        with self.assertRaises(CheckpointValidationError):
            self.ck.restore_model_state(path, self.model)
        self.assertTrue(self.ck._equal_state(before, self.model.state_dict()))

    def test_model_wrong_dtype_and_unknown_tensor_fail(self):
        from safetensors.torch import save_file
        for bad in ("dtype", "name"):
            with self.subTest(bad=bad):
                path = self.root / bad
                self.ck.write_model_state(path, self.model)
                tensors = {k: v.clone() for k, v in self.model.state_dict().items()}
                if bad == "dtype":
                    tensors["weight"] = tensors["weight"].float()
                else:
                    tensors["unknown"] = tensors.pop("weight")
                save_file(tensors, str(path / "diffusion_pytorch_model.safetensors"))
                with self.assertRaises(CheckpointValidationError):
                    self.ck.restore_model_state(path, self.model)

    def test_adamw_real_roundtrip_groups_options_and_parameter_identity(self):
        optimizer = self.optimizer(amsgrad=True)
        self.step(optimizer)
        state = self.ck.capture_optimizer_state(self.model, optimizer, completed_steps=1)
        loaded = self.roundtrip("optimizer.pt", state, model=self.model, optimizer=optimizer, completed_steps=1)
        expected = copy.deepcopy(optimizer.state_dict())
        identities = [id(p) for p in self.model.parameters()]
        optimizer.state.clear()
        optimizer.param_groups[0]["lr"] = 0.9
        optimizer.param_groups[0]["weight_decay"] = 0.8
        self.ck.restore_optimizer_state(self.model, optimizer, loaded, completed_steps=1)
        self.assertTrue(self.ck._equal_state(expected, optimizer.state_dict()))
        self.assertEqual(identities, [id(p) for p in self.model.parameters()])
        self.assertEqual(loaded["parameter_groups"], [["bias"], ["weight"]])

    def test_adafactor_fp32_factored_unfactored_and_first_moments_exact(self):
        optimizer = self.optimizer("adafactor", beta1=0.9)
        for _ in range(3):
            self.step(optimizer)
        state = self.ck.capture_optimizer_state(self.model, optimizer, completed_steps=3)
        loaded = self.roundtrip("optimizer.pt", state, model=self.model, optimizer=optimizer, completed_steps=3)
        pristine = copy.deepcopy(loaded)
        optimizer.state.clear()
        self.ck.restore_optimizer_state(self.model, optimizer, loaded, completed_steps=3)
        restored = self.ck.capture_optimizer_state(self.model, optimizer, completed_steps=3)
        self.assertTrue(self.ck._equal_state(pristine, restored))
        self.assertTrue(self.ck._equal_state(pristine, loaded))
        for entry in optimizer.state.values():
            self.assertEqual(entry["step"], 3)
            for key, tensor in entry.items():
                if key != "step":
                    self.assertEqual(tensor.dtype, self.torch.float32)
        # Recovered objects can execute another native CPU update.
        self.step(optimizer)
        self.assertTrue(all(s["step"] == 4 for s in optimizer.state.values()))

    def test_optimizer_wrong_groups_moments_counter_and_options_rejected(self):
        optimizer = self.optimizer("adafactor")
        self.step(optimizer)
        state = self.ck.capture_optimizer_state(self.model, optimizer, completed_steps=1)
        for fault in ("groups", "shape", "dtype", "counter", "options", "unknown", "inventory"):
            bad = copy.deepcopy(state)
            native = bad["state_dict"]
            if fault == "groups":
                bad["parameter_groups"].reverse()
            elif fault == "shape":
                native["state"][0]["exp_avg_sq"] = self.torch.ones(5)
            elif fault == "dtype":
                native["state"][0]["exp_avg_sq"] = native["state"][0]["exp_avg_sq"].bfloat16()
            elif fault == "counter":
                native["state"][0]["step"] = 5
            elif fault == "options":
                native["param_groups"][0]["relative_step"] = True
            elif fault == "unknown":
                native["state"][0]["unknown"] = 1
            else:
                bad["tensor_metadata"] = []
            with self.subTest(fault=fault), self.assertRaises(CheckpointValidationError):
                self.ck.restore_optimizer_state(self.model, optimizer, bad, completed_steps=1)

    def test_optimizer_uninitialized_state_is_explicit(self):
        optimizer = self.optimizer()
        state = self.ck.capture_optimizer_state(self.model, optimizer, completed_steps=0)
        self.assertEqual(state["step_counters"], {"bias": None, "weight": None})
        self.ck.restore_optimizer_state(self.model, optimizer, state, completed_steps=0)
        with self.assertRaises(CheckpointValidationError):
            self.ck.validate_optimizer_state(state, model=self.model, optimizer=optimizer, completed_steps=1)

    def test_scheduler_roundtrip_preserves_stale_warmup_last_lr(self):
        optimizer = self.optimizer()
        scheduler = self.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=3e-6)
        self.step(optimizer)
        for group in optimizer.param_groups:
            group["lr"] = 1.5e-5
        trainer_state = {"global_step": 1, "cosine_updates": 0, "group_lrs": [1.5e-5, 1.5e-5]}
        config = {"total_optimizer_steps": 20, "warmup_steps": 2, "optimizer_options": {"lr": 3e-5}}
        state = self.ck.capture_scheduler_state(scheduler, 0)
        loaded = self.roundtrip("scheduler.pt", state, scheduler=scheduler,
                                configuration=config, trainer_state=trainer_state)
        self.assertEqual(loaded["state_dict"]["_last_lr"], [3e-5, 3e-5])
        scheduler.last_epoch = 999
        self.ck.restore_scheduler_state(scheduler, loaded, configuration=config, trainer_state=trainer_state)
        self.assertTrue(self.ck._equal_state(loaded["state_dict"], scheduler.state_dict()))
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1.5e-5, 1.5e-5])
        bad = copy.deepcopy(loaded)
        bad["state_dict"]["last_epoch"] = 1
        with self.assertRaises(CheckpointValidationError):
            self.ck.restore_scheduler_state(scheduler, bad, configuration=config, trainer_state=trainer_state)

    def test_trainer_counters_identity_and_envelope(self):
        fixture = CheckpointContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest = fixture.manifest
        state = {"progress": copy.deepcopy(manifest["progress"]), "global_step": 3,
                 "ema_updates": 0, "cosine_updates": 2, "group_lrs": [3e-5],
                 "accelerator_step": 3, "loss_accumulator": 0.0, "log_steps": 0}
        loaded = self.roundtrip("trainer.json", state, manifest=manifest)
        self.assertEqual(self.ck.restore_trainer_state(loaded, manifest=manifest), state)
        with self.assertRaises(CheckpointValidationError):
            self.ck.read_state_payload(self.root / "trainer.json", "other", manifest=manifest)
        for key in ("global_step", "ema_updates", "cosine_updates"):
            bad = copy.deepcopy(state)
            bad[key] += 1
            with self.subTest(key=key), self.assertRaises(CheckpointValidationError):
                self.ck.validate_trainer_state(bad, manifest=manifest)

    def test_rng_reproduces_python_numpy_torch_draws(self):
        import random
        import numpy as np
        before = self.ck.capture_rng_state(cuda_devices=0)
        self.addCleanup(self.ck.restore_rng_state, before, cuda_devices=0)
        random.seed(99)
        np.random.seed(44)
        self.torch.manual_seed(55)
        random.gauss(0, 1)  # Exercise cached Gaussian state too.
        np.random.normal()
        state = self.ck.capture_rng_state(cuda_devices=0)
        loaded = self.roundtrip("rng.pt", state, cuda_devices=0)
        def draws():
            return (random.random(), random.gauss(0, 1), np.random.rand(),
                    np.random.normal(), self.torch.rand(4))
        expected = draws()
        self.ck.restore_rng_state(loaded, cuda_devices=0)
        self.assertTrue(self.ck._equal_state(expected, draws()))
        bad = copy.deepcopy(loaded)
        bad["numpy"]["position"] = 900
        with self.assertRaises(CheckpointValidationError):
            self.ck.restore_rng_state(bad, cuda_devices=0)

    def test_data_initial_mid_epoch_and_boundary_roundtrip(self):
        for epoch, cursor in ((0, 0), (0, 2), (1, 0)):
            with self.subTest(epoch=epoch, cursor=cursor):
                order = self.torch.Generator().manual_seed(123)
                loader = self.torch.Generator().manual_seed(456)
                permutation = (self.torch.randperm(4, generator=order) if epoch or cursor else
                               self.torch.empty(0, dtype=self.torch.int64))
                attempts = epoch * 4 + cursor
                progress = {"epoch": epoch, "next_batch_index": cursor,
                            "loop_iterations": attempts, "attempted_optimizer_steps": attempts}
                state = self.ck.capture_data_order_state(progress=progress, dataset_fingerprint="a"*64,
                                                        permutation=permutation, order_generator=order,
                                                        loader_generator=loader)
                path = self.root / str(attempts)
                path.mkdir()
                context = dict(progress=progress, dataset_size=4, dataset_fingerprint="a"*64)
                self.ck.write_state_payload(path / "data_order.pt", "test-a", state, **context)
                loaded = self.ck.read_state_payload(path / "data_order.pt", "test-a", **context)
                expected_order = self.torch.randperm(4, generator=order)
                expected_loader = self.torch.rand(3, generator=loader)
                e, c, p = self.ck.restore_data_order_state(loaded, order, loader, **context)
                self.assertEqual((e, c), (epoch, cursor))
                self.assertTrue(self.torch.equal(p, permutation))
                self.assertTrue(self.torch.equal(expected_order, self.torch.randperm(4, generator=order)))
                self.assertTrue(self.torch.equal(expected_loader, self.torch.rand(3, generator=loader)))
                bad = copy.deepcopy(loaded)
                bad["permutation_epoch"] += 1
                with self.assertRaises(CheckpointValidationError):
                    self.ck.validate_data_order_state(bad, **context)

    def test_ema_roundtrip_never_replaces_live_model_weights(self):
        # Exercise the actual standalone EMA class, without loading dependencies/models.
        import importlib.util
        spec = importlib.util.spec_from_file_location("ema_trainer", Path(__file__).parents[1] / "scripts/train_klein_standalone.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ema = module.EMAModel(self.model, decay=0.9)
        before = copy.deepcopy(self.model.state_dict())
        expected = copy.deepcopy(ema.shadow)
        path = self.root / "ema"
        self.ck.write_ema_state(path, ema, self.model, decay=0.9, max_shard_bytes=12)
        for t in ema.shadow.values():
            t.zero_()
        self.ck.restore_ema_state(path, ema, self.model, decay=0.9)
        self.assertTrue(self.ck._equal_state(expected, ema.shadow))
        self.assertTrue(self.ck._equal_state(before, self.model.state_dict()))
        with self.assertRaises(CheckpointValidationError):
            self.ck.restore_ema_state(path, ema, self.model, decay=0.99)

    def test_restricted_load_rejects_arbitrary_objects_and_corruption(self):
        path = self.root / "rng.pt"
        from types import SimpleNamespace
        self.torch.save(SimpleNamespace(untrusted=1), path)
        with self.assertRaisesRegex(CheckpointValidationError, "Cannot decode"):
            self.ck.read_state_payload(path, "test-a", cuda_devices=0)
        path.write_bytes(b"corrupt payload")
        with self.assertRaises(CheckpointValidationError):
            self.ck.read_state_payload(path, "test-a", cuda_devices=0)

    def test_state_envelope_unknown_version_keys_and_identity_rejected(self):
        state = self.ck.capture_rng_state(cuda_devices=0)
        envelope = {"schema_version": 1, "checkpoint_id": "test-a", "state": state}
        for change in ({"schema_version": True}, {"schema_version": 2},
                       {"checkpoint_id": "other"}, {"unknown": 1}, {"state": {}}):
            with self.subTest(change=change), self.assertRaises(CheckpointValidationError):
                self.ck.validate_state_payload("rng.pt", dict(envelope, **change), "test-a", cuda_devices=0)

    def test_rng_invalid_bytes_and_cuda_count_fail_without_global_mutation(self):
        before = self.ck.capture_rng_state(cuda_devices=0)
        for fault in ("cpu", "cuda", "python", "numpy"):
            state = copy.deepcopy(before)
            if fault == "cpu":
                state["torch_cpu"] = self.torch.zeros(1, dtype=self.torch.uint8)
            elif fault == "cuda":
                state["torch_cuda"] = [state["torch_cpu"]]
            elif fault == "python":
                state["python"] = (2, state["python"][1], None)
            else:
                state["numpy"]["keys"][0] = -1
            with self.subTest(fault=fault), self.assertRaises(CheckpointValidationError):
                self.ck.restore_rng_state(state, cuda_devices=0)
            self.assertTrue(self.ck._equal_state(before, self.ck.capture_rng_state(cuda_devices=0)))

    def test_data_duplicate_order_wrong_identity_and_bad_generator_fail(self):
        progress = {"epoch": 0, "next_batch_index": 1, "loop_iterations": 1, "attempted_optimizer_steps": 1}
        state = self.ck.capture_data_order_state(progress=progress, dataset_fingerprint="a"*64,
                    permutation=self.torch.arange(4), order_generator=self.torch.Generator(),
                    loader_generator=self.torch.Generator())
        for fault in ("permutation", "identity", "rng", "cursor"):
            bad = copy.deepcopy(state)
            if fault == "permutation":
                bad["permutation"][1] = 0
            elif fault == "identity":
                bad["dataset_fingerprint"] = "b"*64
            elif fault == "rng":
                bad["loader_generator_state"] = self.torch.zeros(1, dtype=self.torch.uint8)
            else:
                bad["next_batch_index"] = 2
            with self.subTest(fault=fault), self.assertRaises(CheckpointValidationError):
                self.ck.validate_data_order_state(bad, progress=progress, dataset_size=4, dataset_fingerprint="a"*64)

    def test_production_read_barrier_does_not_accept_a_tiny_model_class(self):
        fixture = CheckpointContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        checkpoint = self.ck.validate_checkpoint(fixture.root)
        optimizer = self.optimizer()
        scheduler = self.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "training objects differ"):
            self.ck.read_recovery_states(checkpoint, model=self.model, optimizer=optimizer, scheduler=scheduler)

    def test_state_writes_refuse_overwrite(self):
        state = self.ck.capture_rng_state(cuda_devices=0)
        self.roundtrip("rng.pt", state, cuda_devices=0)
        before = (self.root / "rng.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            self.ck.write_state_payload(self.root / "rng.pt", "test-a", state, cuda_devices=0)
        self.assertEqual(before, (self.root / "rng.pt").read_bytes())

    def test_cosine_roundtrip_with_a_skipped_attempt(self):
        optimizer = self.optimizer()
        scheduler = self.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=3e-6)
        config = {"total_optimizer_steps": 20, "warmup_steps": 2, "optimizer_options": {"lr": 3e-5}}
        # Reproduce the current manual warmup, then one skipped cosine request.
        for attempt in range(5):
            if attempt != 3:
                self.step(optimizer)
            if attempt < 2:
                for group in optimizer.param_groups:
                    group["lr"] = 3e-5 * (attempt + 1) / 2
            elif attempt != 3:
                scheduler.step()  # Native equivalent of wrapper's skip suppression.
        state = self.ck.capture_scheduler_state(scheduler, 2)
        trainer_state = {"global_step": 5, "cosine_updates": 2, "group_lrs": state["group_lrs"]}
        loaded = self.roundtrip("scheduler.pt", state, scheduler=scheduler,
                                configuration=config, trainer_state=trainer_state)
        self.ck.restore_scheduler_state(scheduler, loaded, configuration=config, trainer_state=trainer_state)
        self.assertEqual(scheduler.last_epoch, 2)
        bad = copy.deepcopy(loaded)
        bad["state_dict"]["_last_lr"] = [1.0, 1.0]
        with self.assertRaises(CheckpointValidationError):
            self.ck.restore_scheduler_state(scheduler, bad, configuration=config, trainer_state=trainer_state)

    def test_native_scheduler_and_weights_continue_exactly_after_recovery(self):
        """Real CPU steps; skips are explicit simulation, not Accelerate evidence."""
        torch, ck = self.torch, self.ck
        horizon, warmup, base_lr = 30, 3, 3e-5
        skipped = {1, 5}  # One warmup skip and one cosine skip, zero-based attempts.
        config = {"total_optimizer_steps": horizon, "warmup_steps": warmup,
                  "optimizer_options": {"lr": base_lr}}
        for kind in ("adamw", "adafactor"):
            for dtype in (torch.float32, torch.bfloat16):
                initial = copy.deepcopy(self.model).to(dtype=dtype)

                def fresh():
                    model = copy.deepcopy(initial)
                    groups = [{"params": [model.bias], "weight_decay": 0.04},
                              {"params": [model.weight], "weight_decay": 0.02}]
                    if kind == "adamw":
                        optimizer = torch.optim.AdamW(groups, lr=base_lr, foreach=False)
                    else:
                        from transformers.optimization import Adafactor
                        optimizer = Adafactor(groups, lr=base_lr, relative_step=False,
                                              scale_parameter=False)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=horizon - warmup, eta_min=base_lr * 0.1)
                    return model, optimizer, scheduler

                def run(objects, start, end):
                    model, optimizer, scheduler = objects
                    trace = []
                    for attempt in range(start, end):
                        used_lrs = [g["lr"] for g in optimizer.param_groups]
                        inputs = torch.linspace(-0.5, 0.5, 6).reshape(2, 3).to(dtype)
                        prediction = model(inputs).float()
                        loss = (prediction - (attempt % 4) * 0.1).square().mean()
                        loss.backward()
                        if attempt not in skipped:
                            optimizer.step()
                        if attempt < warmup:
                            # Match the trainer's evaluation order exactly.
                            factor = (attempt + 1) / warmup
                            for group in optimizer.param_groups:
                                group["lr"] = base_lr * factor
                        elif attempt not in skipped:
                            scheduler.step()
                        optimizer.zero_grad()
                        updates = sum(a not in skipped for a in range(warmup, attempt + 1))
                        state = ck.capture_scheduler_state(scheduler, updates)
                        trainer = {"global_step": attempt + 1, "cosine_updates": updates,
                                   "group_lrs": state["group_lrs"]}
                        # Test the reconstructed formula against native state at
                        # EVERY boundary, including warmup and both skipped steps.
                        ck.validate_scheduler_state(state, scheduler=scheduler,
                                                    configuration=config, trainer_state=trainer)
                        trace.append({"used_lrs": used_lrs, "group_lrs": state["group_lrs"],
                                      "scheduler": copy.deepcopy(scheduler.state_dict()),
                                      "weights": copy.deepcopy(model.state_dict())})
                    return trace

                uninterrupted_objects = fresh()
                expected = run(uninterrupted_objects, 0, horizon)
                # Within warmup, immediately after warmup, after a cosine step,
                # and immediately after the simulated cosine skip.
                for boundary in (1, 3, 4, 6):
                    with self.subTest(optimizer=kind, dtype=dtype, boundary=boundary):
                        interrupted = fresh()
                        actual = run(interrupted, 0, boundary)
                        model, optimizer, scheduler = interrupted
                        completed = sum(a not in skipped for a in range(boundary))
                        updates = sum(a not in skipped for a in range(warmup, boundary))
                        directory = self.root / f"{kind}-{dtype}-{boundary}"
                        directory.mkdir()
                        ck.write_model_state(directory / "model", model)
                        optimizer_state = ck.capture_optimizer_state(model, optimizer, completed_steps=completed)
                        scheduler_state = ck.capture_scheduler_state(scheduler, updates)
                        trainer = {"global_step": boundary, "cosine_updates": updates,
                                   "group_lrs": scheduler_state["group_lrs"]}
                        ck.write_state_payload(directory / "optimizer.pt", "continuity", optimizer_state,
                                               model=model, optimizer=optimizer, completed_steps=completed)
                        ck.write_state_payload(directory / "scheduler.pt", "continuity", scheduler_state,
                                               scheduler=scheduler, configuration=config, trainer_state=trainer)
                        resumed = fresh()  # Scheduler constructed BEFORE optimizer restore.
                        restored_model, restored_optimizer, restored_scheduler = resumed
                        loaded_optimizer = ck.read_state_payload(directory / "optimizer.pt", "continuity",
                            model=restored_model, optimizer=restored_optimizer, completed_steps=completed,
                            group_lrs=trainer["group_lrs"])
                        loaded_scheduler = ck.read_state_payload(directory / "scheduler.pt", "continuity",
                            scheduler=restored_scheduler, configuration=config, trainer_state=trainer)
                        ck.restore_model_state(directory / "model", restored_model)
                        ck.restore_optimizer_state(restored_model, restored_optimizer, loaded_optimizer,
                                                   completed_steps=completed, group_lrs=trainer["group_lrs"])
                        ck.restore_scheduler_state(restored_scheduler, loaded_scheduler,
                                                   configuration=config, trainer_state=trainer)
                        self.assertTrue(ck._equal_state(scheduler.state_dict(), restored_scheduler.state_dict()))
                        self.assertEqual([g["lr"] for g in restored_optimizer.param_groups], trainer["group_lrs"])
                        actual.extend(run(resumed, boundary, horizon))
                        # Exact equality, not tolerance: all attempt LRs, group
                        # LRs, native scheduler states and stored weights.
                        self.assertTrue(ck._equal_state(expected, actual))
                        self.assertTrue(ck._equal_state(uninterrupted_objects[1].state_dict(),
                                                        restored_optimizer.state_dict()))
                if dtype == torch.float32:
                    self.assertFalse(torch.equal(initial.weight, uninterrupted_objects[0].weight))

    def test_parameter_counters_do_not_identify_completed_update_count(self):
        # One update touching both parameters and two updates touching one each
        # have identical optimizer counters. The manifest's completed count must
        # therefore come from authoritative trainer outcomes, not this inventory.
        torch, ck = self.torch, self.ck
        initial = copy.deepcopy(self.model)
        snapshots = []
        for separate in (False, True):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, foreach=False)
            parameters = list(model.parameters())
            batches = [[p] for p in parameters] if separate else [parameters]
            for active in batches:
                for p in active:
                    p.grad = torch.ones_like(p)
                optimizer.step()
                optimizer.zero_grad()
            snapshot = ck.capture_optimizer_state(model, optimizer, completed_steps=len(batches))
            snapshots.append(snapshot["step_counters"])
        self.assertEqual(snapshots[0], snapshots[1])


class CheckpointPublicationTests(unittest.TestCase):
    """Real files/codecs/renames; tiny Flux-named fixture, NOT a Flux execution.

    Production v1 requires CUDA RNG. Tests adapt only its device-count check to
    zero and run the real CPU RNG codec; no production validation is relaxed.
    """

    def setUp(self):
        import torch
        from types import SimpleNamespace
        from scripts import klein_checkpoint as ck
        self.torch, self.ck = torch, ck
        fixture = CheckpointContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.parent = fixture.root.parent
        self.manifest = copy.deepcopy(fixture.manifest)

        class Flux2Transformer2DModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
                self.register_buffer("counter", torch.tensor(7))

        self.model = Flux2Transformer2DModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5, foreach=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=4, eta_min=3e-5 * 0.1)
        self.model.weight.grad = torch.full_like(self.model.weight, 0.2)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.manifest["payloads"] = []
        self.manifest["progress"].update(completed_optimizer_steps=1, attempted_optimizer_steps=1,
                                         skipped_optimizer_steps=0, loop_iterations=1, next_batch_index=1)
        self.manifest["configuration"].update(optimizer="adamw", optimizer_options={"lr": 3e-5, "weight_decay": 0.01},
                                              ema_enabled=True)
        self.manifest["objects"].update(model_tensors=ck.tensor_inventory(self.model), optimizer_class="torch.optim.AdamW")
        self.ema = SimpleNamespace(decay=0.9999, shadow={"weight": self.model.weight.detach().clone()})
        lrs = [g["lr"] for g in self.optimizer.param_groups]
        progress = self.manifest["progress"]
        self.states = {
            "trainer": {"progress": copy.deepcopy(progress), "global_step": 1, "ema_updates": 1,
                        "cosine_updates": 1, "group_lrs": lrs, "accelerator_step": 1,
                        "loss_accumulator": 0.0, "log_steps": 0},
            "optimizer": ck.capture_optimizer_state(self.model, self.optimizer, completed_steps=1),
            "scheduler": ck.capture_scheduler_state(self.scheduler, 1),
            "rng": ck.capture_rng_state(cuda_devices=0),
            "data_order": ck.capture_data_order_state(progress=progress,
                            dataset_fingerprint=self.manifest["configuration"]["dataset_fingerprint"],
                            permutation=torch.arange(4), order_generator=torch.Generator().manual_seed(8),
                            loader_generator=torch.Generator().manual_seed(9)),
        }
        self.native_validate_rng = ck.validate_rng_state

    def publish(self, name, **kwargs):
        from unittest.mock import patch
        def cpu_rng(state, *, cuda_devices):
            self.assertEqual(cuda_devices, 1)  # Production still asks for one.
            self.native_validate_rng(state, cuda_devices=0)
        with patch.object(self.ck, "validate_rng_state", side_effect=cpu_rng):
            return self.ck.publish_checkpoint(self.parent / name, manifest=self.manifest, states=self.states,
                       model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                       ema=self.ema, max_shard_bytes=12, **kwargs)

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def test_success_inventory_hashes_single_export_and_manifest_last(self):
        events = []
        result = self.publish("committed", fault_injector=events.append)
        self.assertEqual(result.outcome, self.ck.PublicationOutcome.PUBLISHED)
        self.assertEqual(result.retention, "deferred")
        self.assertFalse(result.power_loss_durable)
        self.assertIsNone(result.staging_root)
        checked = self.ck.validate_checkpoint(result.destination)
        files = self.snapshot(result.destination)
        entries = checked.manifest["payloads"]
        self.assertEqual(set(files), {p["path"] for p in entries} | {"manifest.json"})
        for p in entries:
            self.assertEqual(p["size_bytes"], len(files[p["path"]]))
            self.assertEqual(p["sha256"], hashlib.sha256(files[p["path"]]).hexdigest())
        writes = [e.removeprefix("write:") for e in events if e.startswith("write:")]
        self.assertEqual(len(writes), len(set(writes)))
        self.assertEqual(set(writes), set(files) - {"manifest.json"})
        self.assertTrue(all(events.index("write:" + name) < events.index("manifest_write") for name in writes))
        self.assertLess(events.index("manifest_write"), events.index("integrity_validation"))
        self.assertLess(events.index("semantic_validation"), events.index("rename"))
        self.assertFalse(any("transformer/" in n or "export" in n or "accelerator_state" in n for n in files))
        self.assertEqual(len(list(result.destination.glob("model/*.safetensors"))), 2)
        self.assertFalse(list(self.parent.glob(".incomplete-*")))
        self.assertEqual(self.manifest["payloads"], [])  # Input not rewritten.

    def test_existing_checkpoint_and_empty_directory_are_not_overwritten(self):
        committed = self.publish("existing")
        before = self.snapshot(committed.destination)
        with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
            self.publish("existing")
        self.assertEqual(caught.exception.result.outcome, self.ck.PublicationOutcome.FAILED)
        self.assertEqual(caught.exception.result.exception_type, "FileExistsError")
        self.assertEqual(before, self.snapshot(committed.destination))
        (self.parent / "empty").mkdir()
        with self.assertRaises(self.ck.CheckpointPublicationError):
            self.publish("empty")
        self.assertEqual(list((self.parent / "empty").iterdir()), [])

    def test_every_prepublication_failure_retains_staging_and_previous_checkpoint(self):
        events = []
        previous = self.publish("previous", fault_injector=events.append)
        before = self.snapshot(previous.destination)
        # Includes every model/EMA shard, index, config and state payload write.
        stages = [e for e in events if e != "after_rename"]
        for index, target in enumerate(stages):
            def fail(stage):
                if stage == target:
                    raise OSError("injected failure " + stage)
            with self.subTest(stage=target), self.assertRaises(self.ck.CheckpointPublicationError) as caught:
                self.publish(f"failure-{index}", fault_injector=fail)
            result = caught.exception.result
            self.assertEqual(result.outcome, self.ck.PublicationOutcome.FAILED)
            self.assertEqual(result.failed_stage, target)
            self.assertEqual(result.exception_type, "OSError")
            self.assertFalse(result.destination.exists())
            self.assertEqual(before, self.snapshot(previous.destination))
            if result.staging_root is not None:
                self.assertTrue(result.staging_root.is_dir())  # Deliberately retained, never pruned.
                self.assertEqual(result.staging_root.parent, result.destination.parent)
                with self.assertRaisesRegex(CheckpointValidationError, "Incomplete staging"):
                    self.ck.validate_checkpoint(result.staging_root)

    def test_postrename_diagnostic_error_is_reported_as_published(self):
        def fail(stage):
            if stage == "after_rename":
                raise RuntimeError("diagnostic failed")
        with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
            self.publish("published-with-error", fault_injector=fail)
        result = caught.exception.result
        self.assertEqual(result.outcome, self.ck.PublicationOutcome.PUBLISHED)
        self.assertEqual(result.failed_stage, "after_rename")
        self.assertEqual(result.retention, "deferred")
        self.assertIsNone(result.staging_root)
        self.ck.validate_checkpoint(result.destination)

    def test_destination_created_during_rename_is_not_replaced(self):
        def race(stage):
            if stage == "rename":
                (self.parent / "racing").mkdir()  # POSIX plain rename would replace this.
        with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
            self.publish("racing", fault_injector=race)
        self.assertEqual(caught.exception.result.failed_stage, "rename")
        self.assertEqual(caught.exception.result.outcome, self.ck.PublicationOutcome.FAILED)
        self.assertEqual(list((self.parent / "racing").iterdir()), [])

    def test_disk_preflight_includes_model_optimizer_ema_and_reserve(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        estimate = self.ck.estimate_checkpoint_bytes(self.manifest, self.states, max_shard_bytes=12)
        self.assertEqual(estimate["model_bytes"], 20)
        self.assertEqual(estimate["ema_bytes"], 12)
        self.assertGreater(estimate["optimizer_bytes"], 0)
        self.assertGreater(estimate["other_state_bytes"], 0)
        self.assertGreater(estimate["staging_reserve_bytes"], estimate["model_bytes"])
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=estimate["required_free_bytes"] - 1)):
            with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
                self.publish("no-space")
        self.assertEqual(caught.exception.result.failed_stage, "disk_preflight")
        self.assertIsNone(caught.exception.result.staging_root)
        self.assertFalse((self.parent / "no-space").exists())

    def test_corrupt_written_payload_fails_integrity_before_rename(self):
        def corrupt(stage):
            if stage == "integrity_validation":
                staging = next(self.parent.glob(".incomplete-corrupt-*"))
                (staging / "rng.pt").write_bytes(b"corruption")
        with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
            self.publish("corrupt", fault_injector=corrupt)
        self.assertEqual(caught.exception.result.failed_stage, "integrity_validation")
        self.assertIn("size mismatch", caught.exception.result.exception_message)
        self.assertFalse((self.parent / "corrupt").exists())

    def test_production_publisher_does_not_accept_zero_cuda_rng_states(self):
        with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
            self.ck.publish_checkpoint(self.parent / "not-production", manifest=self.manifest, states=self.states,
                        model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, ema=self.ema)
        self.assertEqual(caught.exception.result.failed_stage, "inputs")
        self.assertIn("CUDA RNG state count", caught.exception.result.exception_message)
        self.assertIsNone(caught.exception.result.staging_root)

    def test_partial_payload_shard_and_manifest_writes_are_never_committed(self):
        import errno
        from unittest.mock import patch
        from safetensors.torch import save_file
        previous = self.publish("previous-partial-tests")
        before = self.snapshot(previous.destination)
        native_write = self.ck.write_state_payload
        native_dump = json.dump
        for fault in ("payload", "shard", "manifest"):
            def partial_payload(path, *args, **kwargs):
                if Path(path).name == "optimizer.pt":
                    Path(path).write_bytes(b"partial optimizer")
                    raise OSError(errno.ENOSPC, "injected disk exhaustion")
                return native_write(path, *args, **kwargs)
            def partial_shard(tensors, filename, *args, **kwargs):
                if "00002-of-00002" in str(filename):
                    Path(filename).write_bytes(b"partial shard")
                    raise OSError(errno.ENOSPC, "injected disk exhaustion")
                return save_file(tensors, filename, *args, **kwargs)
            def partial_manifest(value, stream, *args, **kwargs):
                if Path(stream.name).name == "manifest.json":
                    stream.write('{"format":')
                    raise OSError(errno.ENOSPC, "injected disk exhaustion")
                return native_dump(value, stream, *args, **kwargs)
            patches = {"payload": patch.object(self.ck, "write_state_payload", side_effect=partial_payload),
                       "shard": patch("safetensors.torch.save_file", side_effect=partial_shard),
                       "manifest": patch("json.dump", side_effect=partial_manifest)}
            with self.subTest(fault=fault), patches[fault], self.assertRaises(self.ck.CheckpointPublicationError) as caught:
                self.publish("partial-" + fault)
            result = caught.exception.result
            self.assertEqual(result.outcome, self.ck.PublicationOutcome.FAILED)
            self.assertFalse(result.destination.exists())
            self.assertTrue(result.staging_root.is_dir())
            files = self.snapshot(result.staging_root)
            self.assertTrue(any(b"partial" in v or v == b'{"format":' for v in files.values()))
            self.assertEqual(before, self.snapshot(previous.destination))
            with self.assertRaisesRegex(CheckpointValidationError, "Incomplete staging"):
                self.ck.validate_checkpoint(result.staging_root)

    def test_linux_rename_wrapper_requests_no_replace_and_propagates_failure(self):
        # ABI/flag regression only; does not establish Linux filesystem behavior
        # when this suite runs on Windows. Real publish tests use the host syscall.
        import ctypes
        import errno
        from unittest.mock import Mock, patch
        libc = Mock()
        libc.renameat2.return_value = 0
        with patch("os.name", "posix"), patch("sys.platform", "linux"), patch.object(ctypes, "CDLL", return_value=libc):
            self.ck._rename_checkpoint_no_replace("/source", "/destination")
            libc.renameat2.assert_called_once_with(-100, b"/source", -100, b"/destination", 1)
            libc.renameat2.return_value = -1
            with patch.object(ctypes, "get_errno", return_value=errno.EEXIST), self.assertRaises(FileExistsError):
                self.ck._rename_checkpoint_no_replace("/source", "/destination")

    def test_staging_creation_os_failure_has_no_staging_or_destination(self):
        from unittest.mock import patch
        with patch("tempfile.mkdtemp", side_effect=OSError("cannot create staging")):
            with self.assertRaises(self.ck.CheckpointPublicationError) as caught:
                self.publish("mkdir-failure")
        self.assertEqual(caught.exception.result.failed_stage, "staging_create")
        self.assertIsNone(caught.exception.result.staging_root)
        self.assertFalse(caught.exception.result.destination.exists())


if __name__ == "__main__":
    unittest.main()
