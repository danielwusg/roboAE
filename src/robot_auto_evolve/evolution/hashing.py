from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robot_auto_evolve.protocol import StrictSchemaError


EDITABLE_FILES = ("scaffold.py",)
GENERATED_SOURCE_DIRECTORIES = frozenset({"__pycache__"})
GENERATED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})


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


@dataclass(frozen=True)
class EditablePolicy:
    allowed: tuple[str, ...] = EDITABLE_FILES

    def __post_init__(self) -> None:
        allowed = tuple(self.allowed)
        if allowed != EDITABLE_FILES:
            raise StrictSchemaError(f"editable allowlist must equal {EDITABLE_FILES}")
        object.__setattr__(self, "allowed", allowed)

    def validate_tree(self, root: Path) -> dict[str, str]:
        hashes = tree_hashes(root, ignore_generated=True)
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
