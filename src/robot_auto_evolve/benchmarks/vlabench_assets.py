from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ARCHIVE_TARGETS = {
    "obj.zip": "obj",
    "scene.zip": "scenes",
}
RECORD_RELATIVE_PATH = Path("obj") / ".robot_auto_evolve_assets.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inventory_digest(inventory: dict[str, tuple[int, int]], *, include_crc: bool) -> str:
    digest = hashlib.sha256()
    for key, (size, crc) in sorted(inventory.items()):
        suffix = f"\0{crc:08x}" if include_crc else ""
        digest.update(f"{key}\0{size}{suffix}\n".encode())
    return digest.hexdigest()


def archive_inventory(archive: Path, target: str) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    inventory: dict[str, tuple[int, int]] = {}
    casefolded: set[str] = set()
    prefix = f"{target}/"
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            if "\\" in item.filename:
                raise RuntimeError(f"unsafe VLABench archive path: {item.filename}")
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe VLABench archive path: {item.filename}")
            if item.is_dir():
                continue
            if not item.filename.startswith(prefix) or relative.parts[0] != target:
                raise RuntimeError(f"unexpected VLABench archive path: {item.filename}")
            mode = item.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG}:
                raise RuntimeError(f"unsupported VLABench archive entry: {item.filename}")
            key = relative.as_posix()
            folded = key.casefold()
            if key in inventory or folded in casefolded:
                raise RuntimeError(f"duplicate VLABench archive path: {key}")
            inventory[key] = (item.file_size, item.CRC)
            casefolded.add(folded)
    if not inventory:
        raise RuntimeError(f"VLABench archive lacks target {target!r}")
    total = sum(size for size, _ in inventory.values())
    return inventory, {
        "target": target,
        "file_count": len(inventory),
        "uncompressed_bytes": total,
        "archive_inventory_sha256": _inventory_digest(inventory, include_crc=True),
        "installed_inventory_sha256": _inventory_digest(inventory, include_crc=False),
    }


def inspect_archives(archive_root: Path) -> dict[str, Any]:
    root = Path(archive_root).resolve()
    result: dict[str, Any] = {}
    for name, target in ARCHIVE_TARGETS.items():
        archive = root / name
        if not archive.is_file():
            raise RuntimeError(f"VLABench archive is absent: {archive}")
        _, summary = archive_inventory(archive, target)
        result[name] = {
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            **summary,
        }
    return result


def _installed_inventory(assets: Path, target: str) -> dict[str, tuple[int, int]]:
    root = assets / target
    if not root.is_dir():
        raise RuntimeError(f"VLABench asset target is absent: {root}")
    inventory: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path == assets / RECORD_RELATIVE_PATH:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"VLABench asset target contains a symlink: {path}")
        if stat.S_ISREG(mode):
            inventory[path.relative_to(assets).as_posix()] = (path.stat().st_size, 0)
        elif not stat.S_ISDIR(mode):
            raise RuntimeError(f"VLABench asset target contains a special file: {path}")
    return inventory


def _expected_record(manifest: Path, manifest_data: dict[str, Any]) -> dict[str, Any]:
    summaries = {
        name: {
            key: entry[key]
            for key in ("target", "file_count", "uncompressed_bytes", "installed_inventory_sha256")
        }
        for name, entry in manifest_data["archives"].items()
    }
    return {
        "schema_version": 2,
        "asset_manifest_sha256": sha256_file(manifest),
        "source_commit": manifest_data["source_commit"],
        "archives": summaries,
    }


def validate_vlabench_assets(
    *,
    assets: Path,
    manifest: Path,
    archive_root: Path | None,
) -> dict[str, Any]:
    assets = Path(assets).resolve()
    manifest = Path(manifest).resolve()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_data.get("schema_version") != 1 or manifest_data.get("status") != "frozen":
        raise RuntimeError("VLABench asset manifest is not frozen")
    if set(manifest_data.get("archives", {})) != set(ARCHIVE_TARGETS):
        raise RuntimeError("VLABench asset manifest archive set differs")
    if archive_root is not None:
        observed = inspect_archives(archive_root)
        for name, entry in observed.items():
            expected = manifest_data["archives"][name]
            for key, value in entry.items():
                if expected.get(key) != value:
                    raise RuntimeError(f"VLABench archive {name} differs at {key}")
    for name, target in ARCHIVE_TARGETS.items():
        installed = _installed_inventory(assets, target)
        expected = manifest_data["archives"][name]
        if len(installed) != expected["file_count"]:
            raise RuntimeError(f"VLABench installed file count differs: {target}")
        if sum(size for size, _ in installed.values()) != expected["uncompressed_bytes"]:
            raise RuntimeError(f"VLABench installed byte count differs: {target}")
        if _inventory_digest(installed, include_crc=False) != expected["installed_inventory_sha256"]:
            raise RuntimeError(f"VLABench installed inventory differs: {target}")
    return _expected_record(manifest, manifest_data)


def read_and_validate_vlabench_asset_record(*, assets: Path, manifest: Path) -> dict[str, Any]:
    assets = Path(assets).resolve()
    record_path = assets / RECORD_RELATIVE_PATH
    if not record_path.is_file():
        raise RuntimeError("VLABench asset provenance record is absent")
    expected = validate_vlabench_assets(assets=assets, manifest=manifest, archive_root=None)
    actual = json.loads(record_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("VLABench asset provenance record differs from the installed inventory")
    return actual
