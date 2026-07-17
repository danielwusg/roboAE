from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ARCHIVE_TARGETS = {
    "embodiments.zip": "embodiments",
    "objects.zip": "objects",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _archive_inventory(archive: Path, target: str) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    inventory: dict[str, tuple[int, int]] = {}
    prefix = f"{target}/"
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):
                continue
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != target:
                raise RuntimeError(f"unsafe RoboTwin 2 archive path: {item.filename}")
            key = relative.as_posix()
            if key in inventory:
                raise RuntimeError(f"duplicate RoboTwin 2 archive path: {key}")
            inventory[key] = (item.file_size, item.CRC)
    if not inventory:
        raise RuntimeError(f"RoboTwin 2 archive lacks target {target!r}")
    digest = hashlib.sha256()
    for key, (size, crc) in sorted(inventory.items()):
        digest.update(f"{key}\0{size}\0{crc:08x}\n".encode())
    summary = {
        "file_count": len(inventory),
        "uncompressed_bytes": sum(size for size, _ in inventory.values()),
        "inventory_sha256": digest.hexdigest(),
    }
    return inventory, summary


def _installed_inventory(assets: Path, target: str) -> dict[str, int]:
    root = assets / target
    if not root.is_dir():
        raise RuntimeError(f"RoboTwin 2 asset target is absent: {root}")
    inventory: dict[str, int] = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"RoboTwin 2 asset target contains a symlink: {path}")
        if stat.S_ISREG(mode):
            inventory[path.relative_to(assets).as_posix()] = path.stat().st_size
        elif not stat.S_ISDIR(mode):
            raise RuntimeError(f"RoboTwin 2 asset target contains a special file: {path}")
    return inventory


def validate_robotwin2_assets(
    *,
    assets: Path,
    manifest: Path,
    verify_archive_hashes: bool,
) -> dict[str, Any]:
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    downloads = assets / ".downloads"
    summaries: dict[str, Any] = {}
    for name, target in ARCHIVE_TARGETS.items():
        archive = downloads / name
        expected = manifest_data["archives"][name]
        if not archive.is_file() or archive.stat().st_size != expected["size_bytes"]:
            raise RuntimeError(f"RoboTwin 2 archive size differs: {archive}")
        if verify_archive_hashes and _sha256(archive) != expected["sha256"]:
            raise RuntimeError(f"RoboTwin 2 archive digest differs: {archive}")
        archived, summary = _archive_inventory(archive, target)
        installed = _installed_inventory(assets, target)
        expected_sizes = {key: value[0] for key, value in archived.items()}
        if installed != expected_sizes:
            missing = sorted(set(expected_sizes) - set(installed))
            extra = sorted(set(installed) - set(expected_sizes))
            changed = sorted(key for key in set(installed) & set(expected_sizes) if installed[key] != expected_sizes[key])
            detail = {"missing": missing[:3], "extra": extra[:3], "size_changed": changed[:3]}
            raise RuntimeError(f"RoboTwin 2 installed asset inventory differs: {detail}")
        summaries[name] = summary
    return {
        "schema_version": 2,
        "asset_manifest_sha256": _sha256(manifest),
        "dataset_revision": manifest_data["dataset_revision"],
        "archives": summaries,
    }


def read_and_validate_robotwin2_asset_record(
    *,
    assets: Path,
    manifest: Path,
    verify_archive_hashes: bool,
) -> dict[str, Any]:
    record_path = assets / ".robot_auto_evolve_assets.json"
    if not record_path.is_file():
        raise RuntimeError("RoboTwin 2 asset provenance record is absent")
    expected = validate_robotwin2_assets(
        assets=assets,
        manifest=manifest,
        verify_archive_hashes=verify_archive_hashes,
    )
    actual = json.loads(record_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("RoboTwin 2 asset provenance record differs from the installed inventory")
    return actual
