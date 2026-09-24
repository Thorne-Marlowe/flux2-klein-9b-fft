"""Real CPU metadata tests; the selected CUDA Accelerator is a runtime fixture."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts import klein_recovery_metadata as metadata
from scripts.klein_checkpoint import CheckpointValidationError, CheckpointCompatibilityError, SCHEMA_VERSION


class Processor:
    _attention_backend = None


class RecoveryMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_root = self.root / "model"
        self.files = ["model_index.json", "transformer/config.json", "vae/config.json", "text_encoder/config.json",
                      "transformer/weights.safetensors", "vae/weights.safetensors", "text_encoder/weights.safetensors",
                      "tokenizer/tokenizer.json"]
        for name in self.files:
            path = self.model_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}" if name.endswith(".json") else b"real fixture bytes")
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        from PIL import Image
        for i in range(2):
            Image.new("RGB", (32, 32), (i, 0, 0)).save(self.data_root / f"{i}.png")
            (self.data_root / f"{i}.txt").write_text(f"caption {i}\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location("metadata_trainer", Path(__file__).parents[1] / "scripts/train_klein_standalone.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        self.dataset = trainer.ImageTextDataset(self.data_root)
        self.buckets = trainer.BUCKET_SIZES
        self.model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        self.model.is_gradient_checkpointing = True
        self.model.attn_processors = {"attention.processor": Processor()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5, foreach=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=18, eta_min=3e-5 * 0.1)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, num_workers=0, drop_last=True)
        self.accelerator = SimpleNamespace(num_processes=1, device=torch.device("cuda:0"), distributed_type="NO",
                                           mixed_precision="bf16", gradient_accumulation_steps=1, scaler=None)
        self.args = SimpleNamespace(smoke_test=False, resume_from=None, model_path=str(self.model_root),
                                    optimizer="adamw", lr=3e-5, weight_decay=0.99, steps=20, warmup_steps=9,
                                    batch_size=1, num_workers=0, gradient_checkpointing=True, use_ema=False,
                                    ema_decay=0.9999, sample_prompts=[], seed=0, max_grad_norm=1.0)

    def build(self):
        return metadata.build_recovery_configuration(args=self.args, dataset=self.dataset, dataloader=self.loader,
                    model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, accelerator=self.accelerator,
                    resolved_model_files=self.files, bucket_sizes=self.buckets)

    def test_canonical_json_contract_and_nonfinite_rejection(self):
        expected = hashlib.sha256('{"a":"é","z":1}'.encode("utf-8")).hexdigest()
        self.assertEqual(metadata.canonical_sha256({"z": 1, "a": "é"}), expected)
        with self.assertRaises(CheckpointValidationError):
            metadata.canonical_sha256({"x": float("nan")})

    def test_model_determinism_sorted_selection_and_unknown_identity(self):
        first = metadata.fingerprint_model(self.model_root, self.files)
        self.assertEqual(first, metadata.fingerprint_model(self.model_root, reversed(self.files)))
        self.assertIsNone(first["document"]["revision"])
        self.assertIsNone(first["document"]["repo_id"])
        (self.model_root / "vae/weights.safetensors").write_bytes(b"different contents")
        self.assertNotEqual(first["sha256"], metadata.fingerprint_model(self.model_root, self.files)["sha256"])

    def test_relocation_preserves_model_dataset_and_source_fingerprints(self):
        relocated = self.root / "moved-model"
        shutil.copytree(self.model_root, relocated)
        self.assertEqual(metadata.fingerprint_model(self.model_root, self.files),
                         metadata.fingerprint_model(relocated, self.files))
        moved_data = self.root / "moved-data"
        shutil.copytree(self.data_root, moved_data)
        dataset = copy.copy(self.dataset)
        dataset.data_dir = moved_data
        dataset.samples = [(str(moved_data / Path(i).name), str(moved_data / Path(c).name)) for i, c in dataset.samples]
        self.assertEqual(metadata.fingerprint_dataset(self.dataset), metadata.fingerprint_dataset(dataset))
        self.assertEqual(metadata.fingerprint_source_code(self.model_root, self.files),
                         metadata.fingerprint_source_code(relocated, self.files))

    def test_dataset_order_and_caption_image_changes_are_identity_relevant(self):
        first = metadata.fingerprint_dataset(self.dataset)
        self.dataset.samples.reverse()
        self.assertNotEqual(first["sha256"], metadata.fingerprint_dataset(self.dataset)["sha256"])
        self.dataset.samples.reverse()
        (self.data_root / "0.txt").write_text("changed caption", encoding="utf-8")
        second = metadata.fingerprint_dataset(self.dataset)
        self.assertNotEqual(first["sha256"], second["sha256"])
        (self.data_root / "0.png").write_bytes(b"changed image bytes")
        self.assertNotEqual(second["sha256"], metadata.fingerprint_dataset(self.dataset)["sha256"])

    def test_preprocessing_tracks_effective_policy_not_unused_target(self):
        first = metadata.fingerprint_preprocessing(self.dataset, bucket_sizes=self.buckets)
        self.dataset.target_size = 512  # Ignored by current bucket-selection path.
        self.assertEqual(first, metadata.fingerprint_preprocessing(self.dataset, bucket_sizes=self.buckets))
        self.assertNotEqual(first["sha256"], metadata.fingerprint_preprocessing(self.dataset, bucket_sizes=list(reversed(self.buckets)))["sha256"])
        self.dataset.fixed_size = True
        second = metadata.fingerprint_preprocessing(self.dataset, bucket_sizes=self.buckets)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.dataset.target_size = 256
        self.assertNotEqual(second["sha256"], metadata.fingerprint_preprocessing(self.dataset, bucket_sizes=self.buckets)["sha256"])

    def test_source_identity_detects_uncommitted_bytes_and_is_order_independent(self):
        first = metadata.fingerprint_source_code(self.model_root, self.files)
        self.assertEqual(first, metadata.fingerprint_source_code(self.model_root, reversed(self.files)))
        (self.model_root / self.files[0]).write_text("changed code fixture", encoding="utf-8")
        self.assertNotEqual(first["sha256"], metadata.fingerprint_source_code(self.model_root, self.files)["sha256"])

    def test_missing_duplicate_unsafe_and_omitted_shard_paths_rejected(self):
        for files in (self.files + [self.files[0]], ["../outside"], self.files[1:]):
            with self.subTest(files=files), self.assertRaises(CheckpointValidationError):
                metadata.fingerprint_model(self.model_root, files)
        index = "transformer/weights.safetensors.index.json"
        (self.model_root / index).write_text(json.dumps({"weight_map": {"w": "missing.safetensors"}}))
        with self.assertRaisesRegex(CheckpointValidationError, "Unselected/unsafe shard"):
            metadata.fingerprint_model(self.model_root, self.files + [index])

    def test_builder_uses_native_options_resolved_warmup_and_actual_sources(self):
        built = self.build()
        config = built["document"]["configuration"]
        self.assertEqual(config["optimizer_options"][0]["weight_decay"], 0.01)
        self.assertNotEqual(config["optimizer_options"][0]["weight_decay"], self.args.weight_decay)
        self.assertEqual(config["warmup_steps"], 2)
        self.assertEqual(config["max_grad_norm"], 1.0)
        self.assertTrue(config["gradient_checkpointing"])
        self.assertEqual(built, self.build())
        source = built["document"]["fingerprints"]["source_code"]["document"]["files"]
        self.assertEqual([p["path"] for p in source], sorted(metadata.SOURCE_FILES))
        metadata.validate_metadata_compatibility(built, self.build())

    def test_adafactor_does_not_copy_unused_cli_weight_decay(self):
        from transformers.optimization import Adafactor
        self.args.optimizer = "adafactor"
        self.optimizer = Adafactor(self.model.parameters(), lr=3e-5, relative_step=False, scale_parameter=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=18, eta_min=3e-5 * 0.1)
        self.assertEqual(self.build()["document"]["configuration"]["optimizer_options"][0]["weight_decay"], 0.0)

    def test_policy_changes_fail_compatibility_and_v1_is_unchanged(self):
        first = self.build()
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(first["document"]["required_checkpoint_schema"], 2)
        self.assertNotIn("schema_version", first["document"])  # Not a publishable manifest.
        self.args.max_grad_norm = 0.5
        with self.assertRaises(CheckpointCompatibilityError):
            metadata.validate_metadata_compatibility(first, self.build())

    def test_builder_rejects_unsupported_runtime(self):
        for key, bad in (("num_processes", 2), ("device", torch.device("cpu")), ("mixed_precision", "fp16"),
                         ("gradient_accumulation_steps", 2), ("distributed_type", "FSDP"), ("scaler", object())):
            old = getattr(self.accelerator, key)
            setattr(self.accelerator, key, bad)
            with self.subTest(key=key), self.assertRaises(CheckpointValidationError):
                self.build()
            setattr(self.accelerator, key, old)

    def test_smoke_sampling_clipping_and_actual_checkpointing_mismatch_rejected(self):
        for key, bad in (("smoke_test", True), ("sample_prompts", ["prompt"]),
                         ("max_grad_norm", float("nan")), ("gradient_checkpointing", False), ("lr", 0.1)):
            old = getattr(self.args, key)
            setattr(self.args, key, bad)
            with self.subTest(key=key), self.assertRaises(CheckpointValidationError):
                self.build()
            setattr(self.args, key, old)

    def test_attention_policy_and_backend_flags_are_recorded_without_changes(self):
        before = metadata.collect_backend_settings(self.model)
        built = self.build()
        self.assertEqual(built["document"]["environment"]["backend_settings"], before)
        self.assertEqual(before, metadata.collect_backend_settings(self.model))
        self.model.attn_processors["attention.processor"]._attention_backend = "unsupported"
        with self.assertRaisesRegex(CheckpointValidationError, "attention backend"):
            self.build()

    def test_collection_does_not_seed_iterate_or_change_model_optimizer(self):
        import random
        import numpy as np
        from unittest.mock import patch
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        parameters = {k: v.clone() for k, v in self.model.state_dict().items()}
        groups = copy.deepcopy(self.optimizer.state_dict())
        with patch.object(type(self.dataset), "__getitem__", side_effect=AssertionError("must not decode")):
            self.build()
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_rng[1]))
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(groups, self.optimizer.state_dict())
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(parameters[k], v))

    def test_backend_change_is_compatibility_relevant(self):
        from unittest.mock import patch
        original = self.build()
        with patch.object(torch.backends.cuda, "flash_sdp_enabled", return_value=not torch.backends.cuda.flash_sdp_enabled()):
            changed = self.build()
        with self.assertRaises(CheckpointCompatibilityError):
            metadata.validate_metadata_compatibility(original, changed)

    def test_source_edit_changes_complete_foundation(self):
        code_root = self.root / "code"
        for name in metadata.SOURCE_FILES:
            path = code_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).parents[1] / name, path)
        def build():
            return metadata.build_recovery_configuration(args=self.args, dataset=self.dataset, dataloader=self.loader,
                        model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, accelerator=self.accelerator,
                        resolved_model_files=self.files, bucket_sizes=self.buckets, source_root=code_root)
        original = build()
        with (code_root / metadata.SOURCE_FILES[0]).open("a", encoding="utf-8") as stream:
            stream.write("\n# uncommitted source edit\n")
        with self.assertRaises(CheckpointCompatibilityError):
            metadata.validate_metadata_compatibility(original, build())

    def test_mutated_fingerprint_and_unknown_metadata_version_rejected(self):
        original = self.build()
        for fault in ("hash", "version"):
            bad = copy.deepcopy(original)
            if fault == "hash":
                bad["document"]["fingerprints"]["dataset"]["sha256"] = "0" * 64
            else:
                bad["document"]["foundation_version"] = True
            bad["sha256"] = metadata.canonical_sha256(bad["document"])
            with self.subTest(fault=fault), self.assertRaises(CheckpointValidationError):
                metadata.validate_metadata_compatibility(bad, bad)

    def test_missing_dependency_version_is_unknown_not_fabricated(self):
        from unittest.mock import patch
        native_version = metadata.importlib.metadata.version
        def version(name):
            if name == "accelerate":
                raise metadata.importlib.metadata.PackageNotFoundError(name)
            return native_version(name)
        with patch.object(metadata.importlib.metadata, "version", side_effect=version):
            built = self.build()
        self.assertIsNone(built["document"]["environment"]["dependencies"]["accelerate"])

    def assert_invalid_foundation(self, value, valid):
        # Exercise both argument positions. Incidental exceptions fail the test.
        for saved, current in ((value, valid), (valid, value), (value, value)):
            with self.assertRaises(CheckpointValidationError):
                metadata.validate_metadata_compatibility(saved, current)

    def rehash(self, value):
        for fingerprint in value["document"]["fingerprints"].values():
            if type(fingerprint) is dict and "document" in fingerprint:
                fingerprint["sha256"] = metadata.canonical_sha256(fingerprint["document"])
        value["sha256"] = metadata.canonical_sha256(value["document"])

    def test_malformed_outer_documents_raise_validation_error(self):
        valid = self.build()
        for value in (None, [], "metadata", 1, True, {}, {"sha256": valid["sha256"]},
                      {"document": valid["document"]}, {"document": None, "sha256": "a" * 64},
                      {**valid, "extra": 1}):
            with self.subTest(value_type=type(value).__name__):
                self.assert_invalid_foundation(value, valid)

    def test_every_required_nested_field_is_checked_before_comparison(self):
        valid = self.build()
        paths = []
        def walk(value, path):
            if type(value) is dict:
                for key, item in value.items():
                    paths.append(path + [key])
                    walk(item, path + [key])
            elif type(value) is list:
                # One representative file/group suffices; all document shapes
                # and every dictionary field are covered without O(dataset) work.
                if value:
                    walk(value[0], path + [0])
        walk(valid, [])
        for path in paths:
            if path[-1] == "decoupled_weight_decay":
                continue  # Optional native field; compatibility still compares presence.
            bad = copy.deepcopy(valid)
            parent = bad
            for key in path[:-1]:
                parent = parent[key]
            del parent[path[-1]]
            with self.subTest(path=path):
                self.assert_invalid_foundation(bad, valid)

    def test_incorrect_nested_types_fail_even_with_recomputed_checksums(self):
        valid = self.build()
        cases = [
            (["configuration"], []), (["environment"], None), (["fingerprints"], []),
            (["configuration", "parameter_groups"], "weight"),
            (["configuration", "parameter_groups"], [[None]]),
            (["configuration", "optimizer_options"], [None]),
            (["configuration", "optimizer_options", 0, "betas"], [True, 0.9]),
            (["configuration", "optimizer_options", 0, "foreach"], 0),
            (["configuration", "dataset_size"], True),
            (["configuration", "gradient_checkpointing"], 1),
            (["configuration", "ema_decay"], 0.9),
            (["environment", "dependencies", "torch"], []),
            (["environment", "backend_settings", "sdpa"], []),
            (["environment", "backend_settings", "sdpa", "flash"], 1),
            (["environment", "backend_settings", "attention_processors"], []),
            (["environment", "backend_settings", "attention_processors", "attention.processor"], None),
            (["fingerprints", "model"], []),
            (["fingerprints", "model", "document", "component_files"], {}),
            (["fingerprints", "model", "document", "component_files", 0, "size_bytes"], True),
            (["fingerprints", "model", "document", "component_files", 0, "path"], "../outside"),
            (["fingerprints", "dataset", "document", 0], None),
            (["fingerprints", "dataset", "document", 0, "image"], []),
            (["fingerprints", "source_code", "document", "files"], []),
            (["fingerprints", "preprocessing", "document", "resolution", "ordered_buckets"], [[True, 256]]),
            (["fingerprints", "preprocessing", "document", "caption", "strip"], 1),
            (["required_checkpoint_schema"], 2.0),
        ]
        for path, replacement in cases:
            bad = copy.deepcopy(valid)
            parent = bad["document"]
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = replacement
            if type(bad["document"]["fingerprints"]) is dict:
                self.rehash(bad)
            else:
                bad["sha256"] = metadata.canonical_sha256(bad["document"])
            with self.subTest(path=path):
                self.assert_invalid_foundation(bad, valid)

    def test_invalid_digest_values_at_every_level(self):
        valid = self.build()
        paths = [["sha256"], ["document", "fingerprints", "model", "sha256"],
                 ["document", "fingerprints", "model", "document", "component_files", 0, "sha256"]]
        for path in paths:
            for digest in (None, 1, [], {}, "", "A" * 64, "g" * 64, "a" * 63, "a" * 64 + "\n"):
                bad = copy.deepcopy(valid)
                parent = bad
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = digest
                with self.subTest(path=path, digest=digest):
                    self.assert_invalid_foundation(bad, valid)

    def test_nonjson_values_cycles_and_unknown_fields_fail_explicitly(self):
        valid = self.build()
        for value in (object(), (1, 2), float("inf"), float("nan")):
            bad = copy.deepcopy(valid)
            bad["document"]["configuration"]["max_grad_norm"] = value
            self.assert_invalid_foundation(bad, valid)
        bad = copy.deepcopy(valid)
        bad["document"]["environment"]["cycle"] = bad
        self.assert_invalid_foundation(bad, valid)
        for section in ("configuration", "environment"):
            bad = copy.deepcopy(valid)
            bad["document"][section]["unknown"] = 1
            self.rehash(bad)
            self.assert_invalid_foundation(bad, valid)

    def test_incomplete_model_selections_fail_minimum_requirements(self):
        for missing in self.files:
            with self.subTest(missing=missing), self.assertRaises(CheckpointValidationError):
                metadata.fingerprint_model(self.model_root, [p for p in self.files if p != missing])

    def test_selection_validation_cannot_detect_unlisted_tokenizer_assets(self):
        first = metadata.fingerprint_model(self.model_root, self.files)
        asset = "tokenizer/chat_template.jinja"
        (self.model_root / asset).write_text("template fixture", encoding="utf-8")
        # Only a future resolver can know this unlisted asset is needed. The
        # supplied-selection producer deliberately does not discover extra files.
        self.assertEqual(first, metadata.fingerprint_model(self.model_root, self.files))
        self.assertNotEqual(first, metadata.fingerprint_model(self.model_root, self.files + [asset]))

    def test_validation_preserves_valid_input_bytes_and_hashes(self):
        valid = self.build()
        original = json.dumps(valid, ensure_ascii=False, sort_keys=True)
        metadata.validate_metadata_compatibility(valid, copy.deepcopy(valid))
        self.assertEqual(original, json.dumps(valid, ensure_ascii=False, sort_keys=True))
        # Existing producers still use exactly the original canonical algorithm.
        for fingerprint in valid["document"]["fingerprints"].values():
            encoded = json.dumps(fingerprint["document"], sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.assertEqual(fingerprint["sha256"], hashlib.sha256(encoded).hexdigest())

    def test_json_roundtrip_accepts_native_torch_version_and_adafactor_options(self):
        from transformers.optimization import Adafactor
        for kind in ("adamw", "adafactor"):
            if kind == "adafactor":
                self.args.optimizer = kind
                self.optimizer = Adafactor(self.model.parameters(), lr=3e-5, relative_step=False, scale_parameter=False)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=18, eta_min=3e-5 * 0.1)
            valid = self.build()
            loaded = json.loads(json.dumps(valid))
            with self.subTest(kind=kind):
                metadata.validate_metadata_compatibility(valid, loaded)
                self.assertEqual(valid["sha256"], loaded["sha256"])

    def test_valid_numeric_representation_difference_is_not_silently_coerced(self):
        valid = self.build()
        changed = copy.deepcopy(valid)
        changed["document"]["configuration"]["max_grad_norm"] = 1  # valid, distinct from 1.0
        self.rehash(changed)
        with self.assertRaises(CheckpointCompatibilityError):
            metadata.validate_metadata_compatibility(valid, changed)


if __name__ == "__main__":
    unittest.main()
