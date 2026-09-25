"""Tiny real CPU checkpoint files; fixture CUDA RNG bytes are not GPU evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch

import torch
from scripts import compare_recovery_checkpoints as comparison
from scripts import klein_checkpoint as ck
import test_recovery_metadata as fixtures
from test_recovery_training import make_session, advance, cpu_rng_fixture


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RecoveryMetadataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.a = self.fixture.root / "control"
        self.b = self.fixture.root / "renamed-resumed"
        session, _, _ = make_session(self.fixture)
        with cpu_rng_fixture():
            advance(session, 4)
            session.save(self.a)
        # Production manifests have one CUDA RNG entry. Structural comparison
        # can inspect its bytes on CPU without pretending to restore that GPU.
        self.mutate_state(self.a, "rng.pt", lambda s: s.update(torch_cuda=[torch.zeros(16, dtype=torch.uint8)]))
        shutil.copytree(self.a, self.b)
        manifest = self.manifest(self.b)
        manifest.update(checkpoint_id="b" * 32, run_id="c" * 32,
                        parent_checkpoint_id="d" * 32, created_at="2026-09-24T00:00:00+00:00")
        self.write_manifest(self.b, manifest)
        for name in comparison.STATE_KEYS:
            envelope = (json.loads((self.b / name).read_text()) if name.endswith(".json") else
                        torch.load(self.b / name, weights_only=True))
            envelope["checkpoint_id"] = manifest["checkpoint_id"]
            self.write_payload(self.b, name, envelope)
        for root, path in ((self.a, "/tmp/selection-a"), (self.b, "/tmp/selection-b")):
            config = json.loads((root / "model/config.json").read_text())
            config["_name_or_path"] = path
            self.write_payload(root, "model/config.json", config)

    def manifest(self, root):
        return json.loads((root / "manifest.json").read_text())

    def write_manifest(self, root, manifest):
        (root / "manifest.json").write_text(json.dumps(manifest))

    def write_payload(self, root, name, value):
        path = root / name
        if name.endswith(".json"):
            path.write_text(json.dumps(value))
        else:
            torch.save(value, path)
        self.refresh(root, name)

    def refresh(self, root, name):
        manifest = self.manifest(root)
        raw = (root / name).read_bytes()
        entry = next(p for p in manifest["payloads"] if p["path"] == name)
        entry.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.write_manifest(root, manifest)

    def mutate_state(self, root, name, change):
        value = (json.loads((root / name).read_text()) if name.endswith(".json") else
                 torch.load(root / name, weights_only=True))
        change(value["state"])
        self.write_payload(root, name, value)

    def test_exact_match_provenance_and_no_cuda_or_file_mutation(self):
        def hashes():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for root in (self.a, self.b) for p in root.rglob("*") if p.is_file()}
        before = hashes()
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
             patch.object(torch.cuda, "device_count", side_effect=AssertionError("CUDA forbidden")):
            report = comparison.compare_checkpoints(self.a, self.b)
        self.assertEqual(report["status"], "exact_match", report)
        self.assertFalse(report["strict_determinism_recorded"])
        self.assertIn("model/config.json/_name_or_path", report["provenance_differences"])
        self.assertIn("optimizer.pt/checkpoint_id", report["provenance_differences"])
        self.assertEqual(len(report["compared_payloads"]), len(self.manifest(self.a)["payloads"]))
        self.assertEqual(before, hashes())

    def test_exact_tensor_dtype_shape_values_and_scalar_types(self):
        for other, suffix in ((torch.tensor([1.0], dtype=torch.float64), "dtype"),
                              (torch.tensor([[1.0]]), "shape"), (torch.tensor([1.000001]), "values")):
            self.assertEqual(comparison.first_difference(torch.tensor([1.0]), other, "x"), "x/" + suffix)
        self.assertEqual(comparison.first_difference(1, True, "x"), "x/type")
        self.assertEqual(comparison.first_difference([1], (1,), "x"), "x/type")

    def test_rng_and_optimizer_changes_are_not_provenance(self):
        for name in ("rng.pt", "optimizer.pt"):
            with self.subTest(name=name):
                original = (self.b / name).read_bytes()
                def change(s):
                    if name == "rng.pt":
                        s["torch_cuda"][0][0] = 1
                    else:
                        next(iter(s["state_dict"]["state"].values()))["exp_avg"].add_(1)
                self.mutate_state(self.b, name, change)
                report = comparison.compare_checkpoints(self.a, self.b)
                self.assertEqual(report["status"], "different", report)
                self.assertTrue(report["first_difference"].startswith(name))
                (self.b / name).write_bytes(original)
                self.refresh(self.b, name)

    def test_counters_and_cursor_must_agree_with_manifest(self):
        for name, field in (("trainer.json", "global_step"), ("scheduler.pt", "cosine_updates"),
                            ("data_order.pt", "next_batch_index")):
            with self.subTest(name=name):
                original = (self.b / name).read_bytes()
                self.mutate_state(self.b, name, lambda s: s.update({field: s[field] + 1}))
                self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")
                (self.b / name).write_bytes(original)
                self.refresh(self.b, name)

    def test_integrity_missing_files_schema_and_incomplete_roots_fail(self):
        path = self.b / "optimizer.pt"
        data = path.read_bytes()
        path.write_bytes(data + b"corrupt")
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")
        path.unlink()
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")
        path.write_bytes(data)
        m = self.manifest(self.b)
        m["schema_version"] = 999
        self.write_manifest(self.b, m)
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b.parent / ".incomplete-copy")["status"], "invalid")

    def test_envelope_identity_checked_before_provenance_exclusion(self):
        value = torch.load(self.b / "optimizer.pt", weights_only=True)
        value["checkpoint_id"] = "e" * 32
        self.write_payload(self.b, "optimizer.pt", value)
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")

    def test_unexpected_keys_and_unsafe_pickle_do_not_pass(self):
        original = (self.b / "rng.pt").read_bytes()
        value = torch.load(self.b / "rng.pt", weights_only=True)
        value["unexpected"] = 1
        self.write_payload(self.b, "rng.pt", value)
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")
        self.write_payload(self.b, "rng.pt", {"object": Path("not-a-state-payload")})
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "invalid")
        (self.b / "rng.pt").write_bytes(original)

    def test_model_changes_and_unknown_config_fields_are_compared(self):
        name = next(p["path"] for p in self.manifest(self.b)["payloads"] if p["path"].startswith("model/") and p["path"].endswith(".safetensors"))
        from safetensors.torch import load_file, save_file
        tensors = {k: v.clone() for k, v in load_file(self.b / name).items()}
        next(iter(tensors.values())).add_(1)
        save_file(tensors, self.b / name)
        self.refresh(self.b, name)
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["first_difference"], name + "/sha256")
        shutil.copyfile(self.a / name, self.b / name)
        self.refresh(self.b, name)
        config = json.loads((self.b / "model/config.json").read_text())
        config["unrecognized_path"] = "/tmp/not-provenance"
        self.write_payload(self.b, "model/config.json", config)
        self.assertEqual(comparison.compare_checkpoints(self.a, self.b)["status"], "different")

    def test_incompatible_effective_configuration_is_compared(self):
        from scripts.klein_recovery_metadata import canonical_sha256
        m = self.manifest(self.b)
        m["metadata"]["document"]["configuration"]["max_grad_norm"] = 2.0
        m["metadata"]["sha256"] = canonical_sha256(m["metadata"]["document"])
        self.write_manifest(self.b, m)
        report = comparison.compare_checkpoints(self.a, self.b)
        self.assertEqual(report["status"], "different", report)

    def test_cli_exit_codes_and_json(self):
        command = [sys.executable, "-B", str(Path(comparison.__file__)), str(self.a), str(self.b)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "exact_match")
        command[-1] = str(self.b / "missing")
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)


class PreservedEvidenceTests(unittest.TestCase):
    # The production image deliberately excludes the preserved archive. This
    # remains a required source-repository integrity test in CI.
    requires_source_repository = True

    def test_original_archive_hashes_metadata_and_boundary_are_preserved(self):
        root = Path(__file__).resolve().parents[1] / "docs/qualification/evidence/2026-09-23"
        inventory = json.loads((root / "inventory.json").read_text())
        archive = root / inventory["archive"]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),
                         "c889f28adf4a917ff5240eebc4af38cfe3cdbe66236538a13b810f1aa35da35d")
        self.assertEqual(archive.stat().st_size, inventory["size_bytes"])
        with tarfile.open(archive) as bundle:
            members = {m.name: m for m in bundle.getmembers() if m.isfile()}
            self.assertEqual(set(members), {m["path"] for m in inventory["members"]})
            for item in inventory["members"]:
                raw = bundle.extractfile(members[item["path"]]).read()
                self.assertEqual(len(raw), item["size_bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
            prefix = "recovery-qualification-evidence/"
            def read(name):
                return bundle.extractfile(prefix + name).read()
            self.assertIn(b"Deterministic algorithms: False", read("environment.txt"))
            self.assertEqual(read("git-commit.txt").decode().strip(), inventory["tested_commit"])
            manifests = [json.loads(read(s + "-json/manifest.json")) for s in ("control", "resumed")]
            for m in manifests:
                ck._schema(m)
                self.assertTrue(m["environment"]["backend_settings"]["deterministic_algorithms"])
                self.assertEqual(m["progress"]["completed_optimizer_steps"], 4)
            self.assertEqual(manifests[0]["metadata"], manifests[1]["metadata"])
            hashes = [{line.split(None, 1)[1].removeprefix("./"): line.split(None, 1)[0]
                       for line in read(s + "-checkpoint-4.sha256").decode().splitlines()}
                      for s in ("control", "resumed")]
            model_names = [n for n in hashes[0] if n.endswith((".safetensors", ".safetensors.index.json"))]
            self.assertEqual(len(model_names), 43)
            for name in model_names:
                self.assertEqual(hashes[0][name], hashes[1][name])
            trace = [json.loads(line) for line in read("resumed-phase1-trace.jsonl").splitlines()]
            boundary = [r for r in trace if r.get("stage") == "acknowledged"][-1]["details"]["progress"]
            self.assertEqual((boundary["attempted_optimizer_steps"], boundary["epoch"], boundary["next_batch_index"]), (2, 1, 0))


if __name__ == "__main__":
    unittest.main()
