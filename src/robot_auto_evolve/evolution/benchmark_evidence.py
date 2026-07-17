from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from robot_auto_evolve.evaluation.scalars import BenchmarkOutcome, BenchmarkScalar, compute_benchmark_scalar
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import canonical_json_bytes

from .benchmark_models import MAX_BENCHMARK_OUTCOMES, PublicDiagnostic


MAX_OUTCOME_BYTES = 32 * 1024 * 1024
MAX_DIAGNOSTICS = 96
MAX_DIAGNOSTIC_INDEX_BYTES = 2 * 1024 * 1024
MAX_TEXT_BYTES = 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = MAX_OUTCOME_BYTES + MAX_DIAGNOSTIC_INDEX_BYTES + MAX_IMAGE_TOTAL_BYTES + 64 * 1024


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _descriptor(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _sha(payload), "size_bytes": len(payload)}


def _bounded_text(payload: bytes) -> tuple[str, bool]:
    text = payload.decode("utf-8", errors="replace")
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_TEXT_BYTES:
        return text, False
    return encoded[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore"), True


def _select_diagnostics(values: Sequence[PublicDiagnostic]) -> tuple[PublicDiagnostic, ...]:
    groups: dict[tuple[str, bool], list[PublicDiagnostic]] = {}
    for value in values:
        groups.setdefault(
            (value.outcome.key.task_id, bool(value.outcome.metrics["success"])),
            [],
        ).append(value)
    for rows in groups.values():
        rows.sort(key=lambda item: item.rank())
    selected: list[PublicDiagnostic] = []
    offset = 0
    while len(selected) < MAX_DIAGNOSTICS:
        added = False
        for key in sorted(groups):
            rows = groups[key]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == MAX_DIAGNOSTICS:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


class BenchmarkPublicEvidence:
    def __init__(
        self,
        root: Path,
        outcomes: tuple[BenchmarkOutcome, ...],
        scalar: BenchmarkScalar,
        manifest: Mapping[str, Any],
    ) -> None:
        self.root = Path(root).resolve()
        self.outcomes = outcomes
        self.scalar = scalar
        self.manifest = dict(manifest)

    @property
    def bundle_sha256(self) -> str:
        return self.manifest["bundle_sha256"]

    @classmethod
    def create(
        cls,
        root: Path,
        outcomes: Sequence[BenchmarkOutcome],
        scalar: BenchmarkScalar,
        diagnostics: Sequence[PublicDiagnostic] = (),
    ) -> "BenchmarkPublicEvidence":
        rows = tuple(sorted(outcomes, key=lambda item: item.key))
        if not 1 <= len(rows) <= MAX_BENCHMARK_OUTCOMES or len({item.key for item in rows}) != len(rows):
            raise StrictSchemaError("benchmark evidence outcome set differs")
        if not isinstance(scalar, BenchmarkScalar):
            raise StrictSchemaError("benchmark evidence scalar differs")
        if compute_benchmark_scalar(scalar.metric, rows).to_mapping() != scalar.to_mapping():
            raise StrictSchemaError("benchmark evidence scalar does not recompute")
        checked_diagnostics = tuple(diagnostics)
        if any(not isinstance(item, PublicDiagnostic) for item in checked_diagnostics):
            raise StrictSchemaError("benchmark evidence diagnostics differ")
        outcome_by_key = {item.key: item for item in rows}
        if any(outcome_by_key.get(item.outcome.key) != item.outcome for item in checked_diagnostics):
            raise StrictSchemaError("benchmark evidence diagnostic outcome differs")
        target = Path(root).resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
        staging.mkdir()
        (staging / "images").mkdir()
        try:
            outcomes_value = {
                "schema_version": 1,
                "kind": "full_benchmark_outcomes",
                "n_episodes": len(rows),
                "scalar": scalar.to_mapping(),
                "outcomes": [item.to_mapping() for item in rows],
            }
            outcomes_payload = canonical_json_bytes(outcomes_value)
            if len(outcomes_payload) > MAX_OUTCOME_BYTES:
                raise StrictSchemaError("benchmark evidence outcomes exceed byte limit")
            (staging / "outcomes.json").write_bytes(outcomes_payload)

            diagnostic_rows = []
            image_descriptors = []
            image_total = 0
            for item in _select_diagnostics(checked_diagnostics):
                common = {
                    "episode_id": item.outcome.key.artifact_id(),
                    "task_id": item.outcome.key.task_id,
                    "success": item.outcome.metrics["success"],
                    "label": item.label,
                    "media_type": item.media_type,
                    "payload_sha256": _sha(item.payload),
                    "payload_bytes": len(item.payload),
                }
                if item.media_type == "text/plain":
                    text, truncated = _bounded_text(item.payload)
                    diagnostic_rows.append({**common, "text": text, "truncated": truncated, "image": None})
                    continue
                if len(item.payload) > MAX_IMAGE_BYTES or image_total + len(item.payload) > MAX_IMAGE_TOTAL_BYTES:
                    continue
                if not item.payload.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise StrictSchemaError("benchmark evidence image is not PNG")
                relative = f"images/{item.rank()[:32]}.png"
                if (staging / relative).exists():
                    raise RuntimeError("benchmark evidence image path collision")
                (staging / relative).write_bytes(item.payload)
                descriptor = _descriptor(relative, item.payload)
                image_descriptors.append(descriptor)
                diagnostic_rows.append({**common, "text": None, "truncated": False, "image": descriptor})
                image_total += len(item.payload)
            diagnostics_value = {
                "schema_version": 1,
                "kind": "bounded_benchmark_diagnostics",
                "limits": {
                    "max_diagnostics": MAX_DIAGNOSTICS,
                    "max_text_bytes": MAX_TEXT_BYTES,
                    "max_image_bytes": MAX_IMAGE_BYTES,
                    "max_image_total_bytes": MAX_IMAGE_TOTAL_BYTES,
                    "max_index_bytes": MAX_DIAGNOSTIC_INDEX_BYTES,
                },
                "diagnostics": diagnostic_rows,
            }
            diagnostics_payload = canonical_json_bytes(diagnostics_value)
            if len(diagnostics_payload) > MAX_DIAGNOSTIC_INDEX_BYTES:
                raise StrictSchemaError("benchmark evidence diagnostics exceed byte limit")
            (staging / "diagnostics.json").write_bytes(diagnostics_payload)
            stable_manifest = {
                "schema_version": 1,
                "kind": "full_benchmark_public_evidence",
                "outcome_count": len(rows),
                "scalar": scalar.to_mapping(),
                "outcomes": _descriptor("outcomes.json", outcomes_payload),
                "diagnostics": _descriptor("diagnostics.json", diagnostics_payload),
                "images": sorted(image_descriptors, key=lambda item: item["path"]),
                "payload_bytes": len(outcomes_payload) + len(diagnostics_payload) + image_total,
            }
            manifest = {**stable_manifest, "bundle_sha256": _sha(canonical_json_bytes(stable_manifest))}
            manifest_payload = canonical_json_bytes(manifest)
            if manifest["payload_bytes"] + len(manifest_payload) > MAX_BUNDLE_BYTES:
                raise StrictSchemaError("benchmark evidence bundle exceeds byte limit")
            (staging / "manifest.json").write_bytes(manifest_payload)
            for path in staging.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)
            (staging / "images").chmod(0o555)
            staging.chmod(0o555)
            os.rename(staging, target)
        except BaseException:
            if staging.exists():
                staging.chmod(0o700)
                images = staging / "images"
                if images.exists():
                    images.chmod(0o700)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls.load(target)

    @classmethod
    def load(cls, root: Path) -> "BenchmarkPublicEvidence":
        path = Path(root).resolve()
        if not path.is_dir() or path.is_symlink():
            raise StrictSchemaError("benchmark evidence root differs")
        manifest_path = path / "manifest.json"
        try:
            manifest_payload = manifest_path.read_bytes()
            manifest = json.loads(manifest_payload)
        except Exception as exc:
            raise StrictSchemaError(f"benchmark evidence manifest is invalid: {exc}") from exc
        expected = {
            "schema_version",
            "kind",
            "outcome_count",
            "scalar",
            "outcomes",
            "diagnostics",
            "images",
            "payload_bytes",
            "bundle_sha256",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != expected or canonical_json_bytes(manifest) != manifest_payload:
            raise StrictSchemaError("benchmark evidence manifest fields differ")
        if manifest["schema_version"] != 1 or manifest["kind"] != "full_benchmark_public_evidence":
            raise StrictSchemaError("benchmark evidence manifest identity differs")
        stable = dict(manifest)
        digest = stable.pop("bundle_sha256")
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None or _sha(canonical_json_bytes(stable)) != digest:
            raise StrictSchemaError("benchmark evidence bundle hash differs")
        descriptors = [manifest["outcomes"], manifest["diagnostics"], *manifest["images"]]
        expected_files = {"manifest.json"}
        payload_total = 0
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256", "size_bytes"}:
                raise StrictSchemaError("benchmark evidence descriptor differs")
            relative = descriptor["path"]
            if type(relative) is not str or relative.startswith("/") or ".." in Path(relative).parts:
                raise StrictSchemaError("benchmark evidence descriptor path differs")
            source = path / relative
            if not source.is_file() or source.is_symlink():
                raise StrictSchemaError("benchmark evidence payload is absent")
            payload = source.read_bytes()
            if len(payload) != descriptor["size_bytes"] or _sha(payload) != descriptor["sha256"]:
                raise StrictSchemaError("benchmark evidence payload descriptor differs")
            expected_files.add(relative)
            payload_total += len(payload)
        actual_files = set()
        actual_directories = set()
        for item in path.rglob("*"):
            if item.is_symlink():
                raise StrictSchemaError("benchmark evidence contains a symlink")
            if item.is_file():
                actual_files.add(item.relative_to(path).as_posix())
            elif item.is_dir():
                actual_directories.add(item.relative_to(path).as_posix())
            else:
                raise StrictSchemaError("benchmark evidence contains a special entry")
        if actual_files != expected_files or actual_directories != {"images"}:
            raise StrictSchemaError("benchmark evidence inventory differs")
        if payload_total != manifest["payload_bytes"] or payload_total + len(manifest_payload) > MAX_BUNDLE_BYTES:
            raise StrictSchemaError("benchmark evidence payload byte count differs")
        outcomes_payload = (path / "outcomes.json").read_bytes()
        if len(outcomes_payload) > MAX_OUTCOME_BYTES:
            raise StrictSchemaError("benchmark evidence outcome payload is too large")
        outcomes_value = json.loads(outcomes_payload)
        if canonical_json_bytes(outcomes_value) != outcomes_payload or set(outcomes_value) != {
            "schema_version",
            "kind",
            "n_episodes",
            "scalar",
            "outcomes",
        }:
            raise StrictSchemaError("benchmark evidence outcomes differ")
        if outcomes_value["schema_version"] != 1 or outcomes_value["kind"] != "full_benchmark_outcomes":
            raise StrictSchemaError("benchmark evidence outcomes identity differs")
        rows = tuple(BenchmarkOutcome.from_mapping(item) for item in outcomes_value["outcomes"])
        if rows != tuple(sorted(rows, key=lambda item: item.key)) or len({item.key for item in rows}) != len(rows):
            raise StrictSchemaError("benchmark evidence outcomes are not canonical")
        if not 1 <= len(rows) <= MAX_BENCHMARK_OUTCOMES or outcomes_value["n_episodes"] != len(rows) or manifest["outcome_count"] != len(rows):
            raise StrictSchemaError("benchmark evidence outcome count differs")
        scalar = BenchmarkScalar.from_mapping(outcomes_value["scalar"])
        if scalar.to_mapping() != manifest["scalar"] or compute_benchmark_scalar(scalar.metric, rows).to_mapping() != scalar.to_mapping():
            raise StrictSchemaError("benchmark evidence scalar differs")
        diagnostics_payload = (path / "diagnostics.json").read_bytes()
        if len(diagnostics_payload) > MAX_DIAGNOSTIC_INDEX_BYTES:
            raise StrictSchemaError("benchmark evidence diagnostic payload is too large")
        diagnostics_value = json.loads(diagnostics_payload)
        if canonical_json_bytes(diagnostics_value) != diagnostics_payload or not isinstance(diagnostics_value.get("diagnostics"), list):
            raise StrictSchemaError("benchmark evidence diagnostics differ")
        if len(diagnostics_value["diagnostics"]) > MAX_DIAGNOSTICS:
            raise StrictSchemaError("benchmark evidence diagnostic count differs")
        return cls(path, rows, scalar, manifest)
