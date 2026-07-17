from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from robot_auto_evolve.protocol import StrictSchemaError


EDITABLE_FILES = ("scaffold.py",)
GENERATED_SOURCE_DIRECTORIES = frozenset({"__pycache__"})
GENERATED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})
TREE_MANIFEST_NAME = "run_manifest.json"
TREE_MANIFEST_SCHEMA_VERSION = 2
TREE_PERMISSION_POLICY = {"files": "0444", "directories": "0555"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_source_path(path: Path) -> bool:
    return (
        any(part in GENERATED_SOURCE_DIRECTORIES or part.endswith(".egg-info") for part in path.parts)
        or path.suffix in GENERATED_SOURCE_SUFFIXES
    )


def tree_hashes(root: Path, *, ignore_generated: bool = False) -> dict[str, str]:
    root = root.resolve()
    if not root.is_dir():
        raise StrictSchemaError(f"hash root is not a directory: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StrictSchemaError(f"symlink is forbidden: {path}")
        if path.is_dir():
            continue
        if path.is_file():
            relative = path.relative_to(root)
            if not ignore_generated or not _generated_source_path(relative):
                result[relative.as_posix()] = file_sha256(path)
            continue
        raise StrictSchemaError(f"special filesystem entry is forbidden: {path}")
    return result


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_name(name: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise StrictSchemaError("tree manifest name must be a file name")


def _tree_directories(root: Path) -> list[str]:
    result: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StrictSchemaError(f"symlink is forbidden: {path}")
        if path.is_dir():
            result.append(path.relative_to(root).as_posix())
        elif not path.is_file():
            raise StrictSchemaError(f"special filesystem entry is forbidden: {path}")
    return result


def _tree_paths(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StrictSchemaError(f"symlink is forbidden: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            files.append(path)
        else:
            raise StrictSchemaError(f"special filesystem entry is forbidden: {path}")
    return files, directories


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_tree_read_only(root: Path) -> None:
    files, directories = _tree_paths(root)
    for path in files:
        path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _verify_tree_read_only(root: Path) -> None:
    files, directories = _tree_paths(root)
    differences: list[str] = []
    for path in files:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o444:
            differences.append(f"{path.relative_to(root).as_posix()}={mode:04o}")
    for path in [root, *directories]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o555:
            relative = "." if path == root else path.relative_to(root).as_posix()
            differences.append(f"{relative}={mode:04o}")
    if differences:
        raise RuntimeError(f"tree manifest: read-only permissions differ: {', '.join(differences)}")


def write_tree_manifest(root: Path, name: str = TREE_MANIFEST_NAME) -> dict[str, Any]:
    root = root.resolve()
    _validate_manifest_name(name)
    path = root / name
    if path.exists():
        raise RuntimeError(f"tree manifest already exists: {path}")
    hashes = tree_hashes(root)
    directories = _tree_directories(root)
    inventory = {"files": hashes, "directories": directories}
    stable = {
        "schema_version": TREE_MANIFEST_SCHEMA_VERSION,
        "files": hashes,
        "directories": directories,
        "content_sha256": mapping_sha256(inventory),
        "permissions": dict(TREE_PERMISSION_POLICY),
    }
    manifest = {**stable, "manifest_sha256": mapping_sha256(stable)}
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root.parent,
        prefix=f".{root.name}.{name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_symlink() and path.is_file() and file_sha256(path) == file_sha256(temporary):
                temporary.unlink()
                _fsync_directory(root)
                _make_tree_read_only(root)
                _fsync_directory(root)
                _verify_tree_read_only(root)
            raise
        temporary.unlink()
        _fsync_directory(root)
        _make_tree_read_only(root)
        _fsync_directory(root)
        _verify_tree_read_only(root)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def verify_tree_manifest(root: Path, name: str = TREE_MANIFEST_NAME) -> dict[str, Any]:
    root = root.resolve()
    _validate_manifest_name(name)
    value = json.loads((root / name).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "files",
        "directories",
        "content_sha256",
        "permissions",
        "manifest_sha256",
    }:
        raise StrictSchemaError("tree manifest: invalid fields")
    stable = dict(value)
    manifest_sha256 = stable.pop("manifest_sha256")
    if stable["schema_version"] != TREE_MANIFEST_SCHEMA_VERSION or mapping_sha256(stable) != manifest_sha256:
        raise StrictSchemaError("tree manifest: checksum mismatch")
    hashes = stable["files"]
    directories = stable["directories"]
    inventory = {"files": hashes, "directories": directories}
    if (
        not isinstance(hashes, Mapping)
        or not isinstance(directories, list)
        or mapping_sha256(inventory) != stable["content_sha256"]
    ):
        raise StrictSchemaError("tree manifest: content checksum mismatch")
    if stable["permissions"] != TREE_PERMISSION_POLICY:
        raise StrictSchemaError("tree manifest: permission policy mismatch")
    actual = tree_hashes(root)
    actual.pop(name, None)
    actual_directories = _tree_directories(root)
    if dict(hashes) != actual or directories != actual_directories:
        raise RuntimeError("tree manifest: directory contents changed")
    _verify_tree_read_only(root)
    return dict(value)


@dataclass(frozen=True)
class EditablePolicy:
    allowed: tuple[str, ...] = EDITABLE_FILES

    def __post_init__(self) -> None:
        allowed = tuple(self.allowed)
        if allowed != EDITABLE_FILES:
            raise StrictSchemaError(f"editable allowlist must equal {EDITABLE_FILES}")
        object.__setattr__(self, "allowed", allowed)

    def validate_tree(self, root: Path) -> dict[str, str]:
        hashes = tree_hashes(root)
        missing = set(self.allowed) - set(hashes)
        unknown = set(hashes) - set(self.allowed)
        if missing or unknown:
            raise StrictSchemaError(f"editable tree differs from allowlist: missing={sorted(missing)}, unknown={sorted(unknown)}")
        compile((root / "scaffold.py").read_text(), str(root / "scaffold.py"), "exec")
        return hashes

    def validate_revision(self, incumbent: Path, candidate: Path) -> dict[str, str]:
        before = self.validate_tree(incumbent)
        after = self.validate_tree(candidate)
        if before == after:
            raise StrictSchemaError("candidate did not change an editable file")
        return after


class FrozenHashGuard:
    def __init__(self, paths: Iterable[Path]) -> None:
        resolved = tuple(sorted((Path(path).resolve() for path in paths), key=str))
        if not resolved:
            raise StrictSchemaError("frozen hash guard requires at least one path")
        self.paths = resolved
        self.expected = self.capture()

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "FrozenHashGuard":
        if set(value) != {"paths", "hashes", "manifest_sha256"}:
            raise StrictSchemaError("frozen hash manifest: invalid fields")
        stable = {"paths": value["paths"], "hashes": value["hashes"]}
        if mapping_sha256(stable) != value["manifest_sha256"]:
            raise StrictSchemaError("frozen hash manifest: checksum mismatch")
        guard = object.__new__(cls)
        guard.paths = tuple(Path(path) for path in value["paths"])
        guard.expected = dict(value["hashes"])
        guard.verify()
        return guard

    def capture(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in self.paths:
            if path.is_symlink():
                raise StrictSchemaError(f"frozen path is a symlink: {path}")
            if path.is_file():
                result[str(path)] = file_sha256(path)
            elif path.is_dir():
                for relative, digest in tree_hashes(path, ignore_generated=True).items():
                    result[f"{path}/{relative}"] = digest
            else:
                raise StrictSchemaError(f"frozen path does not exist: {path}")
        return result

    def verify(self) -> None:
        actual = self.capture()
        if actual != self.expected:
            missing = sorted(set(self.expected) - set(actual))
            added = sorted(set(actual) - set(self.expected))
            changed = sorted(key for key in set(actual) & set(self.expected) if actual[key] != self.expected[key])
            raise RuntimeError(f"frozen hash guard failed: missing={missing}, added={added}, changed={changed}")

    def to_mapping(self) -> dict[str, Any]:
        stable = {"paths": [str(path) for path in self.paths], "hashes": dict(sorted(self.expected.items()))}
        return {**stable, "manifest_sha256": mapping_sha256(stable)}
