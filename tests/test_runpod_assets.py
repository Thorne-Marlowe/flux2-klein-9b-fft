"""CPU-only tests for the provider-independent flat dataset transport."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import dataset_transport as transport


class DatasetTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self, names=("alpha.png", "bravo.jpg", "charlie.webp")):
        root = self.root / "input"
        root.mkdir(exist_ok=True)
        for index, image in enumerate(names):
            (root / image).write_bytes(("image-" + str(index)).encode() * 3)
            (root / (Path(image).stem + ".txt")).write_bytes(("caption-" + str(index)).encode())
        return root

    def pack(self, source=None, name="package", size=20):
        return transport.pack_dataset(source or self.dataset(), self.root / name, size)

    def manifest(self, package):
        return json.loads((package / transport.MANIFEST_NAME).read_text(encoding="utf-8"))

    def write_manifest(self, package, value):
        (package / transport.MANIFEST_NAME).write_bytes(transport._canonical_bytes(value))

    def refresh_shard(self, package, manifest, index=0):
        shard = package / manifest["shards"][index]["path"]
        manifest["shards"][index]["size_bytes"] = shard.stat().st_size
        manifest["shards"][index]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()

    def rebuild_shard(self, package, manifest, entries, index=0):
        shard = package / manifest["shards"][index]["path"]
        replacement = shard.with_suffix(".replacement")
        with tarfile.open(replacement, "w", format=tarfile.PAX_FORMAT) as archive:
            for name, data, kind in entries:
                info = tarfile.TarInfo(name)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                if kind == "regular":
                    info.size = len(data)
                    import io
                    archive.addfile(info, io.BytesIO(data))
                elif kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "outside"
                    archive.addfile(info)
                elif kind == "hardlink":
                    info.type = tarfile.LNKTYPE
                    info.linkname = "outside"
                    archive.addfile(info)
                else:
                    raise AssertionError(kind)
        replacement.replace(shard)
        self.refresh_shard(package, manifest, index)

    def first_shard_entries(self, package, manifest, index=0):
        shard = package / manifest["shards"][index]["path"]
        with tarfile.open(shard, "r") as archive:
            return [(member.name, archive.extractfile(member).read(), "regular")
                    for member in archive if member.isreg()]

    def test_round_trip_validate_and_extract(self):
        source = self.dataset()
        manifest = self.pack(source, size=20)
        package = self.root / "package"
        self.assertEqual(transport.validate_package(package), manifest)
        destination = self.root / "reconstructed"
        self.assertEqual(transport.extract_package(package, destination), manifest)
        self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()},
                         {path.name: path.read_bytes() for path in destination.iterdir()})
        self.assertFalse(list(self.root.glob(".reconstructed.extracting-*")))

    def test_rebuild_is_byte_deterministic(self):
        source = self.dataset()
        self.pack(source, "one", size=20)
        self.pack(source, "two", size=20)
        first, second = self.root / "one", self.root / "two"
        self.assertEqual((first / transport.MANIFEST_NAME).read_bytes(),
                         (second / transport.MANIFEST_NAME).read_bytes())
        self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first.glob("*.tar")},
                         {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in second.glob("*.tar")})

    def test_identity_describes_files_not_shard_target(self):
        source = self.dataset()
        one = self.pack(source, "one", size=10_000)
        two = self.pack(source, "two", size=10)
        self.assertNotEqual([shard["path"] for shard in one["shards"]], [shard["path"] for shard in two["shards"]])
        self.assertEqual(one["dataset_identity"], two["dataset_identity"])

    def test_missing_caption_and_orphan_caption_are_rejected(self):
        source = self.dataset(("alpha.png",))
        (source / "alpha.txt").unlink()
        with self.assertRaisesRegex(transport.DatasetTransportError, "without .txt"):
            self.pack(source)
        (source / "alpha.txt").write_text("caption", encoding="utf-8")
        (source / "orphan.txt").write_text("caption", encoding="utf-8")
        with self.assertRaisesRegex(transport.DatasetTransportError, "without images"):
            self.pack(source)

    def test_archive_corruption_and_member_hash_mismatch_are_rejected(self):
        self.pack(size=10_000)
        package = self.root / "package"
        manifest = self.manifest(package)
        shard = package / manifest["shards"][0]["path"]
        with shard.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(transport.DatasetTransportError, "Shard size or hash mismatch"):
            transport.validate_package(package)

        shutil.rmtree(package)
        self.pack(size=10_000)
        manifest = self.manifest(package)
        entries = self.first_shard_entries(package, manifest)
        entries[0] = (entries[0][0], b"X" * len(entries[0][1]), "regular")
        self.rebuild_shard(package, manifest, entries)
        self.write_manifest(package, manifest)
        with self.assertRaisesRegex(transport.DatasetTransportError, "member hash mismatch"):
            transport.validate_package(package)

    def test_traversal_and_absolute_manifest_paths_are_rejected(self):
        self.pack(size=10_000)
        package = self.root / "package"
        for unsafe in ("../outside.png", "/outside.png"):
            manifest = self.manifest(package)
            manifest["members"][0]["path"] = unsafe
            projection = [{key: member[key] for key in ("path", "size_bytes", "sha256")}
                          for member in manifest["members"]]
            manifest["dataset_identity"] = hashlib.sha256(transport._canonical_bytes(
                {"layout": transport.LAYOUT, "members": projection})).hexdigest()
            self.write_manifest(package, manifest)
            with self.assertRaisesRegex(transport.DatasetTransportError, "Unsafe"):
                transport.validate_package(package)
            self.pack(self.dataset(), "replacement", size=10_000)
            shutil.rmtree(package)
            (self.root / "replacement").replace(package)

    def test_traversal_and_absolute_archive_members_are_rejected_after_shard_hash_refresh(self):
        for unsafe in ("../outside.png", "/outside.png"):
            package_name = "package-" + hashlib.sha256(unsafe.encode()).hexdigest()[:8]
            self.pack(name=package_name, size=10_000)
            package = self.root / package_name
            manifest = self.manifest(package)
            entries = self.first_shard_entries(package, manifest)
            entries[0] = (unsafe, entries[0][1], "regular")
            self.rebuild_shard(package, manifest, entries)
            # The shard inventory is deliberately refreshed so validation must
            # inspect and reject the unsafe tar member path itself.
            self.write_manifest(package, manifest)
            with self.assertRaisesRegex(transport.DatasetTransportError, "Unsafe non-flat path in archive member"):
                transport.validate_package(package)

    def test_symlink_hardlink_duplicate_and_unexpected_members_are_rejected(self):
        for kind in ("symlink", "hardlink"):
            package_name = "package-" + kind
            self.pack(name=package_name, size=10_000)
            package = self.root / package_name
            manifest = self.manifest(package)
            entries = self.first_shard_entries(package, manifest)
            entries[0] = (entries[0][0], b"", kind)
            self.rebuild_shard(package, manifest, entries)
            self.write_manifest(package, manifest)
            with self.assertRaisesRegex(transport.DatasetTransportError, "not a regular"):
                transport.validate_package(package)

        self.pack(name="package-duplicate", size=10_000)
        package = self.root / "package-duplicate"
        manifest = self.manifest(package)
        entries = self.first_shard_entries(package, manifest)
        self.rebuild_shard(package, manifest, entries + [entries[0]])
        self.write_manifest(package, manifest)
        with self.assertRaisesRegex(transport.DatasetTransportError, "Duplicate archive member"):
            transport.validate_package(package)

        self.pack(name="package-unexpected", size=10_000)
        package = self.root / "package-unexpected"
        manifest = self.manifest(package)
        entries = self.first_shard_entries(package, manifest)
        self.rebuild_shard(package, manifest, entries + [("unexpected.txt", b"x", "regular")])
        self.write_manifest(package, manifest)
        with self.assertRaisesRegex(transport.DatasetTransportError, "Unexpected archive member"):
            transport.validate_package(package)

    def test_duplicate_member_across_shards_and_missing_member_are_rejected(self):
        self.pack(size=10)
        package = self.root / "package"
        manifest = self.manifest(package)
        self.assertGreaterEqual(len(manifest["shards"]), 2)
        first = self.first_shard_entries(package, manifest, 0)
        second = self.first_shard_entries(package, manifest, 1)
        self.rebuild_shard(package, manifest, second + [first[0]], 1)
        self.write_manifest(package, manifest)
        with self.assertRaisesRegex(transport.DatasetTransportError, "Unexpected archive member"):
            transport.validate_package(package)

        self.pack(name="missing", size=10_000)
        package = self.root / "missing"
        manifest = self.manifest(package)
        entries = self.first_shard_entries(package, manifest)
        self.rebuild_shard(package, manifest, entries[:-1])
        self.write_manifest(package, manifest)
        with self.assertRaisesRegex(transport.DatasetTransportError, "missing manifest members"):
            transport.validate_package(package)

    def test_failed_extraction_never_exposes_destination_and_existing_destination_is_preserved(self):
        self.pack(size=10_000)
        package = self.root / "package"
        destination = self.root / "dataset"
        with patch.object(transport, "_copy_member", side_effect=RuntimeError("injected extraction failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                transport.extract_package(package, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".dataset.extracting-*")))

        destination.mkdir()
        marker = destination / "preserve"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(transport.DatasetTransportError, "already exists"):
            transport.extract_package(package, destination)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_extracted_directory_rejects_unexpected_final_file(self):
        manifest = self.pack(size=10_000)
        destination = self.root / "dataset"
        transport.extract_package(self.root / "package", destination)
        (destination / "unexpected.bin").write_bytes(b"x")
        with self.assertRaisesRegex(transport.DatasetTransportError, "unexpected files"):
            transport._validate_extracted_directory(destination, manifest)


if __name__ == "__main__":
    unittest.main()
