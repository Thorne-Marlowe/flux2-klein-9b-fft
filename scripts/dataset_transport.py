"""Deterministic, provider-independent transport for flat Klein datasets.

The transport identity is SHA-256 of canonical JSON containing the fixed flat
layout contract and the ordered ``{path, size_bytes, sha256}`` records.  It is
therefore an identity of the reconstructed training files, independent of tar
shard boundaries or archive metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import BinaryIO


FORMAT = "klein-flat-dataset-transport"
SCHEMA_VERSION = 1
MANIFEST_NAME = "dataset-manifest.json"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp")
LAYOUT = {
    "caption_extension": ".txt",
    "image_extensions": list(IMAGE_EXTENSIONS),
    "kind": "flat-image-caption-pairs",
    "top_level_only": True,
}
_CHUNK = 1024 * 1024


class DatasetTransportError(ValueError):
    """A package is malformed, unsafe, incomplete, or cannot be published."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(_CHUNK)
        if not chunk:
            return size, digest.hexdigest()
        digest.update(chunk)
        size += len(chunk)


def _file_record(path: Path, relative: str | None = None) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DatasetTransportError(f"Expected a regular file: {path}")
    with path.open("rb") as handle:
        size, digest = _sha256_stream(handle)
    return {"path": relative or path.name, "size_bytes": size, "sha256": digest}


def _safe_flat_name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetTransportError(f"{context} must be a non-empty path")
    path = PurePosixPath(value)
    if (path.is_absolute() or len(path.parts) != 1 or path.name != value
            or value in (".", "..") or "/" in value or "\\" in value):
        raise DatasetTransportError(f"Unsafe non-flat path in {context}: {value!r}")
    return value


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DatasetTransportError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DatasetTransportError(f"{context} must be an integer >= {minimum}")
    return value


def _normalise_root(path: Path, context: str) -> Path:
    original = Path(path)
    if original.is_symlink():
        raise DatasetTransportError(f"{context} must not be a symbolic link: {path}")
    try:
        result = original.resolve(strict=True)
    except OSError as error:
        raise DatasetTransportError(f"{context} does not exist: {path}") from error
    if result.is_symlink() or not result.is_dir():
        raise DatasetTransportError(f"{context} must be a real directory: {path}")
    return result


def _ensure_parent(destination: Path) -> Path:
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise DatasetTransportError(f"Destination parent must be an existing non-linked directory: {parent}")
    return parent


def _reject_nested(source: Path, destination: Path, operation: str) -> None:
    try:
        destination.resolve().relative_to(source.resolve())
    except ValueError:
        return
    raise DatasetTransportError(f"{operation} destination must not be inside its source directory")


def _input_members(root: Path) -> list[tuple[Path, Path]]:
    images: dict[str, Path] = {}
    captions: dict[str, Path] = {}
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise DatasetTransportError(f"Dataset input must contain only regular top-level files: {item.name!r}")
        _safe_flat_name(item.name, "dataset input filename")
        suffix = item.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            if item.stem in images:
                raise DatasetTransportError(f"Duplicate image stem: {item.stem!r}")
            images[item.stem] = item
        elif item.suffix == ".txt":
            if item.stem in captions:
                raise DatasetTransportError(f"Duplicate caption stem: {item.stem!r}")
            captions[item.stem] = item
        else:
            raise DatasetTransportError(f"Unexpected input file: {item.name!r}")
    if not images:
        raise DatasetTransportError("Dataset input contains no supported images")
    missing = sorted(set(images) - set(captions))
    orphaned = sorted(set(captions) - set(images))
    if missing:
        raise DatasetTransportError("Images without .txt captions: " + ", ".join(missing))
    if orphaned:
        raise DatasetTransportError("Captions without images: " + ", ".join(orphaned))
    return [(images[stem], captions[stem]) for stem in sorted(images, key=lambda value: images[value].name)]


def _shard_sources(pairs: list[tuple[Path, Path]], target: int) -> list[list[Path]]:
    shards: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0
    for image, caption in pairs:
        pair = [image, caption]
        pair_size = sum(path.stat().st_size for path in pair)
        if current and current_size + pair_size > target:
            shards.append(current)
            current, current_size = [], 0
        current.extend(pair)
        current_size += pair_size
    if current:
        shards.append(current)
    return shards


class _HashingReader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.digest.update(chunk)
        self.size += len(chunk)
        return chunk


def _write_shard(path: Path, sources: list[Path]) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    payload_size = 0
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for source in sources:
            before = source.stat()
            info = tarfile.TarInfo(name=source.name)
            info.size = before.st_size
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.pax_headers = {}
            with source.open("rb") as raw:
                reader = _HashingReader(raw)
                archive.addfile(info, reader)
            after = source.stat()
            if (reader.size != before.st_size or after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns or after.st_ino != before.st_ino):
                raise DatasetTransportError(f"Input changed while packing: {source.name!r}")
            records.append({"path": source.name, "size_bytes": reader.size,
                            "sha256": reader.digest.hexdigest()})
            payload_size += reader.size
    return records, payload_size


def _identity(layout: dict[str, object], members: list[dict[str, object]]) -> str:
    projection = [{key: member[key] for key in ("path", "size_bytes", "sha256")} for member in members]
    return hashlib.sha256(_canonical_bytes({"layout": layout, "members": projection})).hexdigest()


def _publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise DatasetTransportError(f"Destination already exists and will not be overwritten: {destination}")
    try:
        os.rename(staging, destination)
    except OSError as error:
        if destination.exists() or destination.is_symlink():
            raise DatasetTransportError(f"Destination appeared during publication and was not overwritten: {destination}") from error
        raise DatasetTransportError(f"Could not publish staged directory: {error}") from error


def pack_dataset(input_dir: Path, package_dir: Path, shard_size_bytes: int) -> dict[str, object]:
    """Pack a strict flat image-caption directory into deterministic tar shards."""
    if type(shard_size_bytes) is not int or shard_size_bytes <= 0:
        raise DatasetTransportError("Shard size must be a positive integer number of bytes")
    source = _normalise_root(Path(input_dir), "Dataset input")
    destination = Path(package_dir).absolute()
    _ensure_parent(destination)
    _reject_nested(source, destination, "Package")
    if destination.exists() or destination.is_symlink():
        raise DatasetTransportError(f"Package destination already exists: {destination}")
    pairs = _input_members(source)
    shards = _shard_sources(pairs, shard_size_bytes)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.packing-", dir=destination.parent))
    try:
        all_members: list[dict[str, object]] = []
        shard_records: list[dict[str, object]] = []
        total = len(shards)
        for index, sources in enumerate(shards, start=1):
            name = f"data-{index:05d}-of-{total:05d}.tar"
            records, payload_size = _write_shard(staging / name, sources)
            for record in records:
                record["shard"] = name
            all_members.extend(records)
            archive = _file_record(staging / name, name)
            shard_records.append({**archive, "payload_size_bytes": payload_size, "member_count": len(records)})
        members = sorted(all_members, key=lambda item: item["path"])
        manifest = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "layout": LAYOUT,
            "pair_count": len(pairs),
            "shards": shard_records,
            "members": members,
            "dataset_identity": _identity(LAYOUT, members),
        }
        (staging / MANIFEST_NAME).write_bytes(_canonical_bytes(manifest))
        _publish_directory(staging, destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _require_keys(value: object, keys: set[str], context: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DatasetTransportError(f"Malformed {context}")
    return value


def _validate_member_layout(members: list[dict[str, object]]) -> None:
    images: dict[str, str] = {}
    captions: set[str] = set()
    for member in members:
        name = member["path"]
        suffix = Path(name).suffix.lower()
        stem = Path(name).stem
        if suffix in IMAGE_EXTENSIONS:
            if stem in images:
                raise DatasetTransportError(f"Duplicate image stem in manifest: {stem!r}")
            images[stem] = name
        elif Path(name).suffix == ".txt":
            captions.add(stem)
        else:
            raise DatasetTransportError(f"Unexpected manifest member type: {name!r}")
    if not images or set(images) != captions:
        raise DatasetTransportError("Manifest does not describe complete image-caption pairs")


def _validate_manifest(document: object) -> dict[str, object]:
    manifest = _require_keys(document, {"format", "schema_version", "layout", "pair_count", "shards", "members", "dataset_identity"}, "manifest")
    if manifest["format"] != FORMAT or manifest["schema_version"] != SCHEMA_VERSION:
        raise DatasetTransportError("Unsupported dataset transport format/schema version")
    if manifest["layout"] != LAYOUT:
        raise DatasetTransportError("Unsupported dataset layout contract")
    _integer(manifest["pair_count"], "pair_count", 1)
    _digest(manifest["dataset_identity"], "dataset_identity")
    if not isinstance(manifest["shards"], list) or not manifest["shards"]:
        raise DatasetTransportError("Manifest must contain one or more shards")
    if not isinstance(manifest["members"], list) or not manifest["members"]:
        raise DatasetTransportError("Manifest must contain members")
    members: list[dict[str, object]] = []
    paths: set[str] = set()
    for raw in manifest["members"]:
        member = _require_keys(raw, {"path", "size_bytes", "sha256", "shard"}, "member")
        name = _safe_flat_name(member["path"], "member path")
        if name in paths:
            raise DatasetTransportError(f"Duplicate manifest member: {name!r}")
        paths.add(name)
        _integer(member["size_bytes"], f"member size {name}")
        _digest(member["sha256"], f"member digest {name}")
        _safe_flat_name(member["shard"], f"member shard {name}")
        members.append(member)
    if [member["path"] for member in members] != sorted(paths):
        raise DatasetTransportError("Manifest members must be sorted by path")
    _validate_member_layout(members)
    if manifest["pair_count"] != len(members) // 2:
        raise DatasetTransportError("Manifest pair count does not match members")
    if manifest["dataset_identity"] != _identity(LAYOUT, members):
        raise DatasetTransportError("Manifest dataset identity does not match member records")
    shard_names: set[str] = set()
    shard_records: list[dict[str, object]] = []
    for index, raw in enumerate(manifest["shards"], start=1):
        shard = _require_keys(raw, {"path", "size_bytes", "sha256", "payload_size_bytes", "member_count"}, "shard")
        name = _safe_flat_name(shard["path"], "shard path")
        expected = f"data-{index:05d}-of-{len(manifest['shards']):05d}.tar"
        if name != expected or name in shard_names:
            raise DatasetTransportError(f"Invalid shard sequence: {name!r}")
        shard_names.add(name)
        _integer(shard["size_bytes"], f"shard size {name}")
        _integer(shard["payload_size_bytes"], f"shard payload size {name}")
        _integer(shard["member_count"], f"shard member count {name}", 1)
        _digest(shard["sha256"], f"shard digest {name}")
        shard_records.append(shard)
    if {member["shard"] for member in members} != shard_names:
        raise DatasetTransportError("Members do not map exactly to manifest shards")
    for shard in shard_records:
        expected_members = [member for member in members if member["shard"] == shard["path"]]
        if shard["member_count"] != len(expected_members) or shard["payload_size_bytes"] != sum(member["size_bytes"] for member in expected_members):
            raise DatasetTransportError(f"Shard inventory does not match members: {shard['path']}")
    return manifest


def read_manifest(package_dir: Path) -> dict[str, object]:
    root = _normalise_root(Path(package_dir), "Package")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DatasetTransportError("Package manifest is missing or not a regular file")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetTransportError("Package manifest is not valid UTF-8 JSON") from error
    return _validate_manifest(document)


def _validate_tar_members(shard_path: Path, expected: dict[str, dict[str, object]]) -> None:
    seen: set[str] = set()
    try:
        # ``r:"`` intentionally rejects compressed or auto-detected archive
        # formats: the contract is an uncompressed .tar transport shard.
        with tarfile.open(shard_path, mode="r:") as archive:
            for member in archive:
                name = _safe_flat_name(member.name, "archive member")
                if not member.isreg():
                    raise DatasetTransportError(f"Archive member is not a regular file: {name!r}")
                if name in seen:
                    raise DatasetTransportError(f"Duplicate archive member: {name!r}")
                seen.add(name)
                record = expected.get(name)
                if record is None:
                    raise DatasetTransportError(f"Unexpected archive member: {name!r}")
                if member.size != record["size_bytes"]:
                    raise DatasetTransportError(f"Archive member size mismatch: {name!r}")
                source = archive.extractfile(member)
                if source is None:
                    raise DatasetTransportError(f"Cannot read archive member: {name!r}")
                with source:
                    size, digest = _sha256_stream(source)
                if size != record["size_bytes"] or digest != record["sha256"]:
                    raise DatasetTransportError(f"Archive member hash mismatch: {name!r}")
    except (tarfile.TarError, OSError) as error:
        raise DatasetTransportError(f"Invalid tar shard {shard_path.name!r}") from error
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise DatasetTransportError("Archive is missing manifest members: " + ", ".join(missing))


def validate_package(package_dir: Path) -> dict[str, object]:
    """Validate every manifest, shard, and member without extracting files."""
    root = _normalise_root(Path(package_dir), "Package")
    manifest = read_manifest(root)
    expected_root = {MANIFEST_NAME, *(shard["path"] for shard in manifest["shards"])}
    actual_root = {item.name for item in root.iterdir()}
    if actual_root != expected_root:
        raise DatasetTransportError("Package contains missing or unexpected root entries")
    members_by_shard: dict[str, dict[str, dict[str, object]]] = {}
    for member in manifest["members"]:
        members_by_shard.setdefault(member["shard"], {})[member["path"]] = member
    seen_global: set[str] = set()
    for shard in manifest["shards"]:
        path = root / shard["path"]
        actual = _file_record(path, shard["path"])
        if actual["size_bytes"] != shard["size_bytes"] or actual["sha256"] != shard["sha256"]:
            raise DatasetTransportError(f"Shard size or hash mismatch: {shard['path']!r}")
        expected = members_by_shard[shard["path"]]
        overlap = seen_global & set(expected)
        if overlap:
            raise DatasetTransportError("Duplicate members across shards: " + ", ".join(sorted(overlap)))
        seen_global.update(expected)
        _validate_tar_members(path, expected)
    if seen_global != {member["path"] for member in manifest["members"]}:
        raise DatasetTransportError("Manifest member inventory is incomplete")
    return manifest


def _copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path,
                 record: dict[str, object]) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise DatasetTransportError(f"Cannot read archive member: {member.name!r}")
    digest = hashlib.sha256()
    size = 0
    with source, target.open("xb") as output:
        while True:
            chunk = source.read(_CHUNK)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    os.chmod(target, 0o644)
    if size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
        raise DatasetTransportError(f"Archive member hash mismatch during extraction: {member.name!r}")


def _validate_extracted_directory(root: Path, manifest: dict[str, object]) -> None:
    expected = {member["path"]: member for member in manifest["members"]}
    actual = {item.name for item in root.iterdir()}
    if actual != set(expected):
        raise DatasetTransportError("Extracted dataset contains missing or unexpected files")
    for name, record in expected.items():
        actual_record = _file_record(root / name, name)
        if actual_record["size_bytes"] != record["size_bytes"] or actual_record["sha256"] != record["sha256"]:
            raise DatasetTransportError(f"Extracted member mismatch: {name!r}")
    _validate_member_layout(list(expected.values()))


def extract_package(package_dir: Path, dataset_dir: Path) -> dict[str, object]:
    """Safely stage and atomically expose a verified flat training directory."""
    package = _normalise_root(Path(package_dir), "Package")
    destination = Path(dataset_dir).absolute()
    _ensure_parent(destination)
    _reject_nested(package, destination, "Dataset")
    if destination.exists() or destination.is_symlink():
        raise DatasetTransportError(f"Dataset destination already exists: {destination}")
    manifest = validate_package(package)
    members_by_shard: dict[str, dict[str, dict[str, object]]] = {}
    for member in manifest["members"]:
        members_by_shard.setdefault(member["shard"], {})[member["path"]] = member
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.extracting-", dir=destination.parent))
    try:
        for shard in manifest["shards"]:
            shard_path = package / shard["path"]
            actual = _file_record(shard_path, shard["path"])
            if actual["size_bytes"] != shard["size_bytes"] or actual["sha256"] != shard["sha256"]:
                raise DatasetTransportError(f"Shard changed before extraction: {shard['path']!r}")
            expected = members_by_shard[shard["path"]]
            with tarfile.open(shard_path, mode="r:") as archive:
                seen: set[str] = set()
                for member in archive:
                    name = _safe_flat_name(member.name, "archive member")
                    if not member.isreg() or name in seen or name not in expected:
                        raise DatasetTransportError(f"Unsafe or unexpected archive member during extraction: {name!r}")
                    seen.add(name)
                    _copy_member(archive, member, staging / name, expected[name])
                if seen != set(expected):
                    raise DatasetTransportError(f"Shard changed during extraction: {shard['path']!r}")
        _validate_extracted_directory(staging, manifest)
        _publish_directory(staging, destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
