"""CPU-only tests for explicit disposable-workspace asset hydration."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import dataset_transport as transport
from scripts import hydrate_runpod_assets as hydration


class RunpodAssetHydrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source_dataset = self.root / "source"
        self.source_dataset.mkdir()
        for name, content in (("one.png", b"image-one"), ("two.jpg", b"image-two")):
            (self.source_dataset / name).write_bytes(content)
            (self.source_dataset / (Path(name).stem + ".txt")).write_text("caption", encoding="utf-8")
        self.remote_package = self.root / "remote-package"
        transport.pack_dataset(self.source_dataset, self.remote_package, 1024)
        self.model_source = {"repository": "black-forest-labs/FLUX.2-klein-base-9B",
                             "requested_revision": "model-tag", "revision": "a" * 40,
                             "inventory": {"model_index.json": 2}, "hub": MagicMock(), "token": "fake-secret"}
        self.hub = MagicMock()
        siblings = [SimpleNamespace(rfilename=path.name, size=path.stat().st_size)
                    for path in self.remote_package.iterdir()]
        siblings.append(SimpleNamespace(rfilename="unrelated/notes.txt", size=1))
        self.hub.HfApi.return_value.dataset_info.return_value = SimpleNamespace(sha="b" * 40, siblings=siblings)

        def download(**kwargs):
            source = self.remote_package / kwargs["filename"]
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return str(target)

        self.hub.hf_hub_download.side_effect = download

    def config(self):
        return {"workspace_root": self.workspace,
                "model": {"repository": "black-forest-labs/FLUX.2-klein-base-9B", "revision": "model-tag",
                          "destination": self.workspace / "models" / "base-9b"},
                "dataset": {"repository": "owner/dataset-transport", "revision": "dataset-tag",
                            "package_destination": self.workspace / "cache" / "dataset-package",
                            "destination": self.workspace / "datasets" / "training"}}

    def model_mocks(self):
        def download(destination, source):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model_index.json").write_text("{}", encoding="utf-8")
            return {"repository": source["repository"], "revision": source["revision"], "downloaded_files": 1}
        return (patch.object(hydration.bootstrap, "resolve_model_source", return_value=self.model_source),
                patch.object(hydration.bootstrap, "download_model_source", side_effect=download),
                patch.object(hydration.bootstrap, "validate_model", return_value={"selected_files": 10, "selected_bytes": 1234}))

    def hydrate(self):
        with self.model_mocks()[0] as resolve, self.model_mocks()[1] as download, self.model_mocks()[2] as validate:
            result = hydration.hydrate(self.config(), hub=self.hub, environment={"HF_TOKEN": "fake-secret"})
        return result, resolve, download, validate

    def test_successful_hydration_uses_bootstrap_model_helpers_and_phase41_dataset_round_trip(self):
        result, resolve, download, validate = self.hydrate()
        self.assertEqual(result["status"], "complete")
        resolve.assert_called_once()
        download.assert_called_once()
        validate.assert_called_once()
        config = self.config()
        manifest = transport.validate_package(config["dataset"]["package_destination"])
        transport.validate_extracted_dataset(config["dataset"]["destination"], manifest)
        expected_files = {transport.MANIFEST_NAME, *(shard["path"] for shard in manifest["shards"])}
        self.assertEqual({call.kwargs["filename"] for call in self.hub.hf_hub_download.call_args_list}, expected_files)
        self.assertTrue(all(call.kwargs["repo_type"] == "dataset" and call.kwargs["revision"] == "b" * 40
                            for call in self.hub.hf_hub_download.call_args_list))
        self.assertEqual(manifest["dataset_identity"], result["dataset"]["dataset_identity"])
        self.assertEqual(result["dataset"]["pair_count"], 2)
        self.assertEqual(result["model"]["resolved_revision"], "a" * 40)
        self.assertEqual(result["dataset"]["resolved_revision"], "b" * 40)
        self.assertTrue(all((self.workspace / name).is_dir()
                            for name in ("models", "datasets", "runs", "evidence", "cache")))
        serialized = json.dumps(result)
        self.assertNotIn("fake-secret", serialized)
        for record in (self.workspace / ".klein-hydration").glob("*.json"):
            self.assertNotIn("fake-secret", record.read_text(encoding="utf-8"))

    def test_matching_completed_assets_are_verified_and_reused(self):
        first, _, _, _ = self.hydrate()
        self.hub.hf_hub_download.reset_mock()
        second, _, _, _ = self.hydrate()
        self.assertEqual(second, first)
        self.hub.hf_hub_download.assert_not_called()

    def test_existing_unrecorded_model_destination_fails_without_overwrite(self):
        config = self.config()
        destination = config["model"]["destination"]
        destination.mkdir(parents=True)
        marker = destination / "unproven"
        marker.write_text("preserve", encoding="utf-8")
        with self.model_mocks()[0] as resolve, self.model_mocks()[1] as download, self.model_mocks()[2]:
            with self.assertRaisesRegex(hydration.HydrationError, "partial hydration state"):
                hydration.hydrate_model(config["model"] | {"workspace_root": self.workspace}, "fake-secret", hub=self.hub)
        resolve.assert_called_once()
        download.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_existing_unrecorded_dataset_destination_fails_without_overwrite(self):
        config = self.config()
        destination = config["dataset"]["destination"]
        destination.mkdir(parents=True)
        marker = destination / "preserve"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(hydration.HydrationError, "cannot be proven"):
            hydration.hydrate_dataset(config["dataset"] | {"workspace_root": self.workspace}, "fake-secret", hub=self.hub)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.hub.hf_hub_download.assert_not_called()

    def test_corrupt_or_missing_shard_fails_without_completion_record(self):
        manifest = json.loads((self.remote_package / transport.MANIFEST_NAME).read_text(encoding="utf-8"))
        shard = self.remote_package / manifest["shards"][0]["path"]
        for action, message in ((lambda: shard.write_bytes(b"corrupt"), "failed Phase 4.1 validation"),
                                (lambda: shard.unlink(), "Dataset download failed")):
            with self.subTest(action=message):
                config = self.config()
                action()
                with self.assertRaisesRegex(hydration.HydrationError, message):
                    hydration.hydrate_dataset(config["dataset"] | {"workspace_root": self.workspace}, "fake-secret", hub=self.hub)
                self.assertFalse((self.workspace / ".klein-hydration" / "dataset.json").exists())
                self.assertFalse(config["dataset"]["package_destination"].exists())
                self.assertFalse(config["dataset"]["destination"].exists())
                if not shard.exists():
                    shutil.copyfile(self.source_dataset / "one.png", shard)

    def test_failed_extraction_does_not_expose_dataset_or_completion_record(self):
        config = self.config()
        with patch.object(hydration.dataset_transport, "extract_package", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(hydration.HydrationError, "extraction failed"):
                hydration.hydrate_dataset(config["dataset"] | {"workspace_root": self.workspace}, "fake-secret", hub=self.hub)
        self.assertFalse(config["dataset"]["destination"].exists())
        self.assertFalse((self.workspace / ".klein-hydration" / "dataset.json").exists())

    def test_missing_environment_token_stops_before_hub_access(self):
        with self.assertRaisesRegex(hydration.HydrationError, "HF_TOKEN"):
            hydration.hydrate(self.config(), hub=self.hub, environment={})
        self.hub.HfApi.assert_not_called()

    def test_load_config_requires_explicit_safe_destinations(self):
        config_file = self.root / "assets.json"
        config_file.write_text(json.dumps({"schema_version": 1, "workspace_root": str(self.workspace),
                                           "model": {"repository": "repo/model", "revision": "main",
                                                     "destination": str(self.workspace / "models/model")},
                                           "dataset": {"repository": "repo/data", "revision": "main",
                                                       "package_destination": str(self.workspace / "cache/package"),
                                                       "destination": str(self.workspace / "datasets/data")}}), encoding="utf-8")
        loaded = hydration.load_config(config_file)
        self.assertEqual(loaded["workspace_root"], self.workspace.resolve())
        document = json.loads(config_file.read_text(encoding="utf-8"))
        for unsafe in ("/outside", str(self.workspace / ".." / "outside")):
            document["dataset"]["destination"] = unsafe
            config_file.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(hydration.HydrationError, "under workspace_root"):
                hydration.load_config(config_file)


if __name__ == "__main__":
    unittest.main()
