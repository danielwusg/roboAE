from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from robot_auto_evolve.protocol.schema import (
    StrictSchemaError,
    enum,
    fields,
    integer,
    json_object,
    reject_json_constant,
    sequence,
    sha256,
    string,
)
from robot_auto_evolve.services.identity import ServiceIdentity

from .manifest import (
    EPISODE_STATES,
    ArtifactDescriptor,
    EpisodeKey,
    EpisodeManifest,
    EpisodePlan,
    canonical_json_bytes,
    mapping_sha256,
)


RUN_STATES = frozenset({"complete", "partial", "error"})


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=json_object, parse_constant=reject_json_constant)
    except Exception as exc:
        raise StrictSchemaError(f"artifact: failed to load {path}: {exc}") from exc


class ArtifactRun:
    def __init__(self, path: Path, header: Mapping[str, Any], plan: EpisodePlan) -> None:
        self.path = path
        self.header = dict(header)
        self.plan = plan
        self.scope_split = self.header["scope_split"]
        self.expected_keys = (
            plan.episodes if self.scope_split is None else plan.for_split(self.scope_split)
        )
        self._key_by_id = {item.artifact_id(): item for item in self.expected_keys}

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        run_id: str,
        profile_hash: str,
        plan: EpisodePlan,
        code_hash: str,
        service_identities: Sequence[ServiceIdentity],
        split: str | None = None,
        created_ns: int | None = None,
    ) -> "ArtifactRun":
        root = Path(path)
        if not isinstance(plan, EpisodePlan):
            raise StrictSchemaError("artifact.plan: expected EpisodePlan")
        run_id = string(run_id, "artifact.run_id")
        profile_hash = sha256(profile_hash, "artifact.profile_hash")
        code_hash = sha256(code_hash, "artifact.code_hash")
        identities = tuple(service_identities)
        if not identities or any(not isinstance(item, ServiceIdentity) for item in identities):
            raise StrictSchemaError("artifact.service_identities: expected nonempty ServiceIdentity sequence")
        identity_keys = [(item.service_name, item.replica_id) for item in identities]
        if len(set(identity_keys)) != len(identity_keys):
            raise StrictSchemaError("artifact.service_identities: duplicate service replica")
        identities = tuple(sorted(identities, key=lambda item: (item.service_name, item.replica_id)))
        created = time.time_ns() if created_ns is None else integer(created_ns, "artifact.created_ns", minimum=0)
        scope_split = None if split is None else enum(split, {"evolve", "selection", "transfer"}, "artifact.split")
        header = {
            "schema_version": 1,
            "run_id": run_id,
            "profile_sha256": profile_hash,
            "episode_plan_sha256": plan.resolved_hash(),
            "code_sha256": code_hash,
            "created_ns": created,
            "service_identities": [item.to_mapping() for item in identities],
            "episode_plan": plan.to_mapping(),
            "scope_split": scope_split,
        }
        root.mkdir(parents=True, exist_ok=False)
        try:
            (root / "episodes").mkdir()
            (root / ".lock").touch(mode=0o600, exist_ok=False)
            _atomic_write(root / "run.json", canonical_json_bytes(header))
            _fsync_directory(root)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return cls(root, header, plan)

    @classmethod
    def resume(
        cls,
        path: str | Path,
        *,
        profile_hash: str | None = None,
        plan: EpisodePlan | None = None,
        code_hash: str | None = None,
        service_identities: Sequence[ServiceIdentity] | None = None,
        split: str | None = None,
    ) -> "ArtifactRun":
        root = Path(path)
        if (root / "final.json").exists():
            raise RuntimeError("artifact run is finalized")
        header = cls._parse_header(_load_json(root / "run.json"))
        stored_plan = EpisodePlan.from_mapping(header["episode_plan"])
        if stored_plan.resolved_hash() != header["episode_plan_sha256"]:
            raise StrictSchemaError("artifact: episode plan hash mismatch")
        if profile_hash is not None and sha256(profile_hash, "artifact.profile_hash") != header["profile_sha256"]:
            raise StrictSchemaError("artifact: profile hash mismatch")
        if code_hash is not None and sha256(code_hash, "artifact.code_hash") != header["code_sha256"]:
            raise StrictSchemaError("artifact: code hash mismatch")
        if plan is not None and plan.resolved_hash() != header["episode_plan_sha256"]:
            raise StrictSchemaError("artifact: supplied episode plan differs")
        if service_identities is not None:
            expected = [
                item.to_mapping()
                for item in sorted(service_identities, key=lambda item: (item.service_name, item.replica_id))
            ]
            if expected != header["service_identities"]:
                raise StrictSchemaError("artifact: service identities differ")
        if split is not None and enum(split, {"evolve", "selection", "transfer"}, "artifact.split") != header["scope_split"]:
            raise StrictSchemaError("artifact: split scope differs")
        return cls(root, header, stored_plan)

    @classmethod
    def verify(cls, path: str | Path) -> dict[str, Any]:
        root = Path(path)
        header = cls._parse_header(_load_json(root / "run.json"))
        plan = EpisodePlan.from_mapping(header["episode_plan"])
        if plan.resolved_hash() != header["episode_plan_sha256"]:
            raise StrictSchemaError("artifact: episode plan hash mismatch")
        final_value = fields(
            _load_json(root / "final.json"),
            {
                "schema_version",
                "run_id",
                "state",
                "profile_sha256",
                "episode_plan_sha256",
                "scope_split",
                "code_sha256",
                "run_header_sha256",
                "episode_manifest_sha256",
                "n_expected",
                "n_recorded",
                "n_complete",
                "missing_episode_ids",
                "finalized_ns",
                "manifest_sha256",
            },
            path="final_manifest",
        )
        final = dict(final_value)
        if integer(final["schema_version"], "final_manifest.schema_version") != 1:
            raise StrictSchemaError("final_manifest.schema_version: expected 1")
        if final["run_id"] != header["run_id"]:
            raise StrictSchemaError("artifact: run id mismatch")
        if final["state"] not in RUN_STATES:
            raise StrictSchemaError("artifact: invalid final state")
        for key in ("profile_sha256", "episode_plan_sha256", "code_sha256"):
            if final[key] != header[key]:
                raise StrictSchemaError(f"artifact: {key} mismatch")
        if final["scope_split"] != header["scope_split"]:
            raise StrictSchemaError("artifact: split scope mismatch")
        expected_header_hash = hashlib.sha256((root / "run.json").read_bytes()).hexdigest()
        if final["run_header_sha256"] != expected_header_hash:
            raise StrictSchemaError("artifact: run header hash mismatch")
        manifest_hash = sha256(final.pop("manifest_sha256"), "final_manifest.manifest_sha256")
        if mapping_sha256(final) != manifest_hash:
            raise StrictSchemaError("artifact: final manifest hash mismatch")
        final["manifest_sha256"] = manifest_hash
        run = cls(root, header, plan)
        manifests = run.episode_manifests()
        stored_hashes = final["episode_manifest_sha256"]
        if not isinstance(stored_hashes, Mapping) or any(type(key) is not str for key in stored_hashes):
            raise StrictSchemaError("artifact: invalid episode hash mapping")
        actual_hashes = {item.key.artifact_id(): run._verify_episode(item) for item in manifests}
        if actual_hashes != dict(stored_hashes):
            raise StrictSchemaError("artifact: episode manifest hashes differ")
        actual_episode_dirs = {item.name for item in (root / "episodes").iterdir()}
        if actual_episode_dirs != set(actual_hashes):
            raise StrictSchemaError("artifact: unexpected episode directories")
        if {item.name for item in root.iterdir()} != {".lock", "episodes", "final.json", "run.json"}:
            raise StrictSchemaError("artifact: unexpected run files")
        expected_ids = {key.artifact_id() for key in run.expected_keys}
        recorded_ids = set(actual_hashes)
        actual_missing = sorted(expected_ids - recorded_ids)
        if actual_missing != final["missing_episode_ids"]:
            raise StrictSchemaError("artifact: missing episode ids differ")
        if integer(final["n_expected"], "final_manifest.n_expected") != len(run.expected_keys):
            raise StrictSchemaError("artifact: expected count mismatch")
        if integer(final["n_recorded"], "final_manifest.n_recorded") != len(manifests):
            raise StrictSchemaError("artifact: recorded count mismatch")
        complete_count = sum(item.state == "complete" for item in manifests)
        if integer(final["n_complete"], "final_manifest.n_complete") != complete_count:
            raise StrictSchemaError("artifact: complete count mismatch")
        if any(item.state == "error" for item in manifests):
            expected_state = "error"
        elif not actual_missing and complete_count == len(manifests):
            expected_state = "complete"
        else:
            expected_state = "partial"
        if final["state"] != expected_state:
            raise StrictSchemaError("artifact: final state differs from episode state")
        integer(final["finalized_ns"], "final_manifest.finalized_ns", minimum=header["created_ns"])
        return final

    @staticmethod
    def _parse_header(value: Any) -> dict[str, Any]:
        obj = fields(
            value,
            {
                "schema_version",
                "run_id",
                "profile_sha256",
                "episode_plan_sha256",
                "code_sha256",
                "created_ns",
                "service_identities",
                "episode_plan",
                "scope_split",
            },
            path="run_header",
        )
        if integer(obj["schema_version"], "run_header.schema_version") != 1:
            raise StrictSchemaError("run_header.schema_version: expected 1")
        string(obj["run_id"], "run_header.run_id")
        sha256(obj["profile_sha256"], "run_header.profile_sha256")
        sha256(obj["episode_plan_sha256"], "run_header.episode_plan_sha256")
        sha256(obj["code_sha256"], "run_header.code_sha256")
        integer(obj["created_ns"], "run_header.created_ns", minimum=0)
        identities = [
            ServiceIdentity.from_mapping(item)
            for item in sequence(obj["service_identities"], "run_header.service_identities")
        ]
        if not identities:
            raise StrictSchemaError("run_header.service_identities: empty")
        if obj["scope_split"] is not None:
            enum(obj["scope_split"], {"evolve", "selection", "transfer"}, "run_header.scope_split")
        EpisodePlan.from_mapping(obj["episode_plan"])
        return dict(obj)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with (self.path / ".lock").open("r+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _assert_mutable(self) -> None:
        if (self.path / "final.json").exists():
            raise RuntimeError("artifact run is finalized")

    def record_episode(
        self,
        key: EpisodeKey,
        *,
        state: str,
        success: bool | None,
        steps: int,
        artifacts: Mapping[str, bytes],
        error: str | None = None,
        started_ns: int | None = None,
        finished_ns: int | None = None,
    ) -> EpisodeManifest:
        if not isinstance(key, EpisodeKey) or key not in self.expected_keys:
            raise StrictSchemaError("artifact.key: episode not in run scope")
        if state not in EPISODE_STATES:
            raise StrictSchemaError("artifact.state: invalid episode state")
        if not isinstance(artifacts, Mapping):
            raise StrictSchemaError("artifact.artifacts: expected mapping")
        descriptors: list[ArtifactDescriptor] = []
        checked_payloads: dict[str, bytes] = {}
        for name, payload in sorted(artifacts.items()):
            if type(payload) is not bytes:
                raise StrictSchemaError(f"artifact.{name}: expected bytes")
            descriptor = ArtifactDescriptor(
                name=name,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
            descriptors.append(descriptor)
            checked_payloads[descriptor.name] = payload
        started = time.time_ns() if started_ns is None else started_ns
        finished = time.time_ns() if finished_ns is None else finished_ns
        manifest = EpisodeManifest(
            key=key,
            state=state,
            success=success,
            steps=steps,
            started_ns=started,
            finished_ns=finished,
            artifacts=tuple(descriptors),
            error=error,
        )
        target = self.path / "episodes" / key.artifact_id()
        self._assert_mutable()
        with self._locked():
            self._assert_mutable()
            if target.exists():
                existing = EpisodeManifest.from_mapping(_load_json(target / "episode.json"))
                stable_existing = existing.to_mapping()
                stable_new = manifest.to_mapping()
                for item in (stable_existing, stable_new):
                    item.pop("started_ns")
                    item.pop("finished_ns")
                if stable_existing != stable_new:
                    raise RuntimeError("episode artifact already exists with different content")
                return existing
            temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
            temporary.mkdir()
            try:
                for descriptor in descriptors:
                    artifact_path = temporary / descriptor.name
                    descriptor_fd = os.open(artifact_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor_fd, "wb") as stream:
                        stream.write(checked_payloads[descriptor.name])
                        stream.flush()
                        os.fsync(stream.fileno())
                _atomic_write(temporary / "episode.json", canonical_json_bytes(manifest.to_mapping()))
                _fsync_directory(temporary)
                os.rename(temporary, target)
                _fsync_directory(target.parent)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return manifest

    def episode_manifests(self) -> tuple[EpisodeManifest, ...]:
        manifests: list[EpisodeManifest] = []
        for key in self.expected_keys:
            path = self.path / "episodes" / key.artifact_id() / "episode.json"
            if path.exists():
                manifest = EpisodeManifest.from_mapping(_load_json(path))
                if manifest.key != key:
                    raise StrictSchemaError(f"artifact: episode key mismatch at {path}")
                manifests.append(manifest)
        return tuple(manifests)

    def _verify_episode(self, manifest: EpisodeManifest) -> str:
        root = self.path / "episodes" / manifest.key.artifact_id()
        expected_names = {"episode.json", *(item.name for item in manifest.artifacts)}
        actual_names = {item.name for item in root.iterdir()}
        if actual_names != expected_names:
            raise StrictSchemaError(f"artifact: unexpected episode files at {root}")
        for descriptor in manifest.artifacts:
            path = root / descriptor.name
            if not path.is_file() or path.stat().st_size != descriptor.size_bytes:
                raise StrictSchemaError(f"artifact: missing or wrong size {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != descriptor.sha256:
                raise StrictSchemaError(f"artifact: hash mismatch {path}")
        return hashlib.sha256((root / "episode.json").read_bytes()).hexdigest()

    def finalize(self, *, state: str | None = None, finalized_ns: int | None = None) -> dict[str, Any]:
        with self._locked():
            self._assert_mutable()
            manifests = self.episode_manifests()
            by_key = {item.key: item for item in manifests}
            missing = [key for key in self.expected_keys if key not in by_key]
            if any(item.state == "error" for item in manifests):
                actual_state = "error"
            elif not missing and all(item.state == "complete" for item in manifests):
                actual_state = "complete"
            else:
                actual_state = "partial"
            if state is not None:
                if state not in RUN_STATES:
                    raise StrictSchemaError("artifact.final_state: invalid")
                if state != actual_state:
                    raise StrictSchemaError(f"artifact.final_state: expected {actual_state}")
            episode_hashes = {
                item.key.artifact_id(): self._verify_episode(item)
                for item in manifests
            }
            finalized = time.time_ns() if finalized_ns is None else integer(
                finalized_ns, "artifact.finalized_ns", minimum=self.header["created_ns"]
            )
            final = {
                "schema_version": 1,
                "run_id": self.header["run_id"],
                "state": actual_state,
                "profile_sha256": self.header["profile_sha256"],
                "episode_plan_sha256": self.header["episode_plan_sha256"],
                "scope_split": self.scope_split,
                "code_sha256": self.header["code_sha256"],
                "run_header_sha256": hashlib.sha256((self.path / "run.json").read_bytes()).hexdigest(),
                "episode_manifest_sha256": episode_hashes,
                "n_expected": len(self.expected_keys),
                "n_recorded": len(manifests),
                "n_complete": sum(item.state == "complete" for item in manifests),
                "missing_episode_ids": [key.artifact_id() for key in missing],
                "finalized_ns": finalized,
            }
            final["manifest_sha256"] = mapping_sha256(final)
            _atomic_write(self.path / "final.json", canonical_json_bytes(final))
            for manifest in manifests:
                episode_dir = self.path / "episodes" / manifest.key.artifact_id()
                for child in episode_dir.iterdir():
                    if child.is_file():
                        child.chmod(0o444)
                episode_dir.chmod(0o555)
            (self.path / "run.json").chmod(0o444)
            (self.path / "final.json").chmod(0o444)
            (self.path / ".lock").chmod(0o444)
            (self.path / "episodes").chmod(0o555)
            self.path.chmod(0o555)
            return final
