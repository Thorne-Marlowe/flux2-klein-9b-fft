"""CPU-only transport tests using real checkpoint validation and a fake Hub."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts import export_runpod_artifacts as artifacts
from scripts import klein_checkpoint as checkpoint
from tests import test_checkpoint_recovery as checkpoint_fixtures


class FakeHub:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.downloads: list[str] = []
        self.download_revisions: list[str] = []
        self.uploads: list[str] = []
        self.upload_parents: list[str | None] = []
        self.upload_count = 0
        self.fail_upload_at: int | None = None
        self.upload_hook = None
        self.revision = "a" * 40
        self.HfApi = lambda token: FakeApi(self, token)

    def hf_hub_download(self, *, filename, local_dir, revision, **kwargs):
        self.downloads.append(filename)
        self.download_revisions.append(revision)
        if filename not in self.files:
            raise FileNotFoundError(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


class FakeApi:
    def __init__(self, hub: FakeHub, token: str):
        self.hub, self.token = hub, token

    def whoami(self):
        return {"name": "fixture"}

    def auth_check(self, **kwargs):
        return None

    def model_info(self, repository, **kwargs):
        return SimpleNamespace(sha=self.hub.revision,
                               siblings=[SimpleNamespace(rfilename=name) for name in sorted(self.hub.files)])

    def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
        self.hub.upload_count += 1
        if self.hub.fail_upload_at == self.hub.upload_count:
            raise RuntimeError("credential-bearing-url-should-not-escape")
        self.hub.files[path_in_repo] = Path(path_or_fileobj).read_bytes()
        self.hub.uploads.append(path_in_repo)
        self.hub.upload_parents.append(kwargs.get("parent_commit"))
        self.hub.revision = f"{self.hub.upload_count:040x}"
        if self.hub.upload_hook is not None:
            self.hub.upload_hook(path_in_repo)
        return SimpleNamespace(oid=self.hub.revision)


class ArtifactTransportTests(unittest.TestCase):
    repository = "owner/private-klein-checkpoints"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hub = FakeHub()
        self.environment = {"HF_OUTPUT_TOKEN": "fake-output-secret"}

    def valid_checkpoint(self) -> Path:
        fixture = checkpoint_fixtures.CheckpointContractTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture.root

    def export(self, root: Path, label="checkpoint"):
        return artifacts.export_checkpoint(root, repository=self.repository, revision="main", label=label,
                                           hub=self.hub, environment=self.environment)

    def record(self, result):
        path = result["remote_namespace"] + "/export-record.json"
        return artifacts.parse_completion_record(self.hub.files[path], repository=self.repository,
                                                 export_id=result["export_id"])

    def remote_checkpoint_path(self, result, relative):
        return result["remote_namespace"] + "/checkpoint/" + relative

    def test_valid_checkpoint_uses_authoritative_validator_and_exports_record_last(self):
        root = self.valid_checkpoint()
        original = checkpoint.validate_checkpoint(root)
        result = self.export(root)
        record = self.record(result)
        self.assertEqual(record["checkpoint"]["checkpoint_id"], original.manifest["checkpoint_id"])
        self.assertEqual(record["transport_identity"], result["transport_identity"])
        self.assertEqual(record["requested_revision"], "main")
        self.assertEqual(record["resolved_revision_before_export"], "a" * 40)
        self.assertEqual(result["resolved_revision_before_export"], "a" * 40)
        self.assertEqual(self.hub.uploads[-1], result["remote_namespace"] + "/export-record.json")
        self.assertTrue(all(path.startswith(result["remote_namespace"] + "/checkpoint/")
                            for path in self.hub.uploads[:-1]))
        self.assertEqual(self.hub.upload_parents[0], "a" * 40)
        self.assertTrue(all(parent == f"{index:040x}" for index, parent in
                            enumerate(self.hub.upload_parents[1:], start=1)))
        serialized = json.dumps(record)
        self.assertNotIn("fake-output-secret", serialized)
        self.assertEqual(result["published_revision"], self.hub.revision)

    def test_invalid_checkpoint_is_rejected_before_remote_publication(self):
        root = self.valid_checkpoint()
        (root / "optimizer.pt").unlink()
        with self.assertRaises(checkpoint.CheckpointValidationError):
            self.export(root)
        self.assertEqual(self.hub.uploads, [])

    def test_inventory_identity_is_deterministic(self):
        root = self.valid_checkpoint()
        _, first = artifacts.checkpoint_inventory(root)
        _, second = artifacts.checkpoint_inventory(root)
        self.assertEqual(first, second)
        self.assertEqual(artifacts.transport_identity(first), artifacts.transport_identity(second))

    def test_interrupted_export_has_no_completion_record(self):
        root = self.valid_checkpoint()
        self.hub.fail_upload_at = 2
        with self.assertRaisesRegex(artifacts.ArtifactTransportError, "No completion record") as raised:
            self.export(root)
        self.assertNotIn("fake-output-secret", str(raised.exception))
        export_id = artifacts._export_id(checkpoint._plain(checkpoint.validate_checkpoint(root).manifest),
                                         artifacts.transport_identity(artifacts.checkpoint_inventory(root)[1]), "checkpoint")
        self.assertNotIn(artifacts._record_path(export_id), self.hub.files)
        self.assertTrue(self.hub.files)

    def test_local_checkpoint_mutation_after_inventory_prevents_certification(self):
        root = self.valid_checkpoint()
        _, inventory = artifacts.checkpoint_inventory(root)
        target_relative = inventory[0]["path"]
        self.assertNotEqual(target_relative, "manifest.json")
        payload = root / target_relative
        manifest_path = root / "manifest.json"
        observed = {}

        # The content-derived namespace is not known until export starts. Use
        # the fixed checkpoint suffix, and mutate only after the first sorted
        # payload has been uploaded from the original inventory.
        def mutate_after_upload(remote_path):
            if observed or not remote_path.endswith(f"/checkpoint/{target_relative}"):
                return
            replacement = b"mutated-after-transport-inventory"
            payload.write_bytes(replacement)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest["payloads"]:
                if entry["path"] == target_relative:
                    entry["size_bytes"] = len(replacement)
                    entry["sha256"] = hashlib.sha256(replacement).hexdigest()
            manifest_path.write_bytes(artifacts._canonical_bytes(manifest))
            observed["payload"] = replacement
            observed["manifest"] = manifest_path.read_bytes()

        self.hub.upload_hook = mutate_after_upload
        with self.assertRaisesRegex(artifacts.ArtifactTransportError, "Checkpoint changed before upload") as raised:
            self.export(root, label="mutating")
        self.assertTrue(observed)
        self.assertNotIn("fake-output-secret", str(raised.exception))
        self.assertFalse(any(path.endswith("/export-record.json") for path in self.hub.files))
        self.assertEqual(payload.read_bytes(), observed["payload"])
        self.assertEqual(manifest_path.read_bytes(), observed["manifest"])
        checkpoint.validate_checkpoint(root)

    def test_completed_export_is_idempotently_reused_and_conflict_is_refused(self):
        root = self.valid_checkpoint()
        result = self.export(root)
        uploads = len(self.hub.uploads)
        repeated = self.export(root)
        self.assertEqual(repeated["export_id"], result["export_id"])
        self.assertTrue(repeated["reused"])
        self.assertEqual(len(self.hub.uploads), uploads)
        # A completion record makes its namespace immutable.  A subsequently
        # missing payload is corruption, not an invitation to repair it.
        missing = self.remote_checkpoint_path(result, self.record(result)["files"][0]["path"])
        del self.hub.files[missing]
        with self.assertRaisesRegex(artifacts.ArtifactTransportError, "could not be downloaded"):
            self.export(root)
        self.assertEqual(len(self.hub.uploads), uploads)
        # Restore it so the conflict assertion separately exercises record
        # content rather than the missing-payload guard.
        self.hub.files[missing] = (root / self.record(result)["files"][0]["path"]).read_bytes()
        record_path = result["remote_namespace"] + "/export-record.json"
        document = json.loads(self.hub.files[record_path])
        document["requested_revision"] = "conflicting-branch"
        self.hub.files[record_path] = artifacts._canonical_bytes(document)
        with self.assertRaisesRegex(artifacts.ArtifactTransportError, "conflicts"):
            self.export(root)
        self.assertEqual(len(self.hub.uploads), uploads)

    def test_import_reads_record_first_downloads_only_declared_files_and_validates_before_publication(self):
        result = self.export(self.valid_checkpoint())
        record = self.record(result)
        self.hub.downloads.clear()
        destination = self.root / "imported"
        imported = artifacts.import_checkpoint(repository=self.repository, export_id=result["export_id"],
                                                destination=destination, hub=self.hub, environment=self.environment)
        self.assertFalse(imported["reused"])
        self.assertEqual(self.hub.downloads[0], result["remote_namespace"] + "/export-record.json")
        expected = {result["remote_namespace"] + "/export-record.json",
                    *(self.remote_checkpoint_path(result, entry["path"]) for entry in record["files"])}
        self.assertEqual(set(self.hub.downloads), expected)
        self.assertTrue(all(revision == self.hub.revision for revision in self.hub.download_revisions))
        self.assertEqual(artifacts.checkpoint_inventory(destination)[1], record["files"])
        checkpoint.validate_checkpoint(destination)
        self.assertEqual(imported["resolved_revision"], self.hub.revision)

    def test_corrupt_or_missing_remote_file_never_exposes_destination(self):
        for action in ("corrupt", "missing"):
            with self.subTest(action=action):
                result = self.export(self.valid_checkpoint(), label="copy-" + action)
                record = self.record(result)
                remote = self.remote_checkpoint_path(result, record["files"][0]["path"])
                if action == "corrupt":
                    self.hub.files[remote] = b"corrupt"
                else:
                    del self.hub.files[remote]
                destination = self.root / ("import-" + action)
                with self.assertRaises(artifacts.ArtifactTransportError):
                    artifacts.import_checkpoint(repository=self.repository, export_id=result["export_id"],
                                                destination=destination, hub=self.hub, environment=self.environment)
                self.assertFalse(destination.exists())
                private_staging = list(self.root.glob(".incomplete-artifact-import-*"))
                self.assertTrue(private_staging)
                for staging in private_staging:
                    with self.assertRaises(checkpoint.CheckpointValidationError):
                        checkpoint.validate_checkpoint(staging)

    def test_unsafe_remote_record_paths_are_rejected_before_checkpoint_download(self):
        for unsafe in ("../outside", "/absolute/path"):
            with self.subTest(unsafe=unsafe):
                result = self.export(self.valid_checkpoint(), label="unsafe-" + hashlib.sha256(unsafe.encode()).hexdigest()[:8])
                record_path = result["remote_namespace"] + "/export-record.json"
                document = json.loads(self.hub.files[record_path])
                document["files"][0]["path"] = unsafe
                self.hub.files[record_path] = artifacts._canonical_bytes(document)
                self.hub.downloads.clear()
                with self.assertRaisesRegex(artifacts.ArtifactTransportError, "Unsafe"):
                    artifacts.import_checkpoint(repository=self.repository, export_id=result["export_id"],
                                                destination=self.root / "unsafe-import", hub=self.hub,
                                                environment=self.environment)
                self.assertEqual(self.hub.downloads, [record_path])

    def test_matching_import_is_reused_and_mismatched_destination_is_not_replaced(self):
        result = self.export(self.valid_checkpoint())
        destination = self.root / "imported"
        artifacts.import_checkpoint(repository=self.repository, export_id=result["export_id"], destination=destination,
                                    hub=self.hub, environment=self.environment)
        self.hub.downloads.clear()
        reused = artifacts.import_checkpoint(repository=self.repository, export_id=result["export_id"], destination=destination,
                                             hub=self.hub, environment=self.environment)
        self.assertTrue(reused["reused"])
        self.assertEqual(self.hub.downloads, [result["remote_namespace"] + "/export-record.json"])
        payload = destination / "rng.pt"
        modified = b"different-but-structurally-opaque"
        payload.write_bytes(modified)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["payloads"]:
            if entry["path"] == "rng.pt":
                entry["size_bytes"] = len(modified)
                entry["sha256"] = hashlib.sha256(modified).hexdigest()
        (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        checkpoint.validate_checkpoint(destination)
        with self.assertRaisesRegex(artifacts.ArtifactTransportError, "differs"):
            artifacts.import_checkpoint(repository=self.repository, export_id=result["export_id"], destination=destination,
                                        hub=self.hub, environment=self.environment)
        self.assertEqual(payload.read_bytes(), modified)

    def test_output_token_is_required_without_hub_access(self):
        with self.assertRaisesRegex(artifacts.ArtifactTransportError, "HF_OUTPUT_TOKEN"):
            artifacts.export_checkpoint(self.valid_checkpoint(), repository=self.repository, hub=self.hub, environment={})
        self.assertEqual(self.hub.uploads, [])


if __name__ == "__main__":
    unittest.main()
