from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from robot_auto_evolve.protocol.schema import (
    StrictSchemaError,
    boolean,
    enum,
    fields,
    integer,
    json_object,
    reject_json_constant,
    sequence,
    sha256,
    string,
)


SPLITS = frozenset({"evolve", "selection", "transfer"})
EPISODE_SPLITS = SPLITS | {"benchmark"}
EPISODE_STATES = frozenset({"complete", "partial", "error"})


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, order=True)
class EpisodeKey:
    split: str
    task_id: str
    scenario_id: str
    environment_seed: int
    policy_seed: int
    replicate_id: str
    horizon: int
    protocol: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", enum(self.split, EPISODE_SPLITS, "episode.split"))
        object.__setattr__(self, "task_id", string(self.task_id, "episode.task_id"))
        object.__setattr__(self, "scenario_id", string(self.scenario_id, "episode.scenario_id"))
        object.__setattr__(
            self, "environment_seed", integer(self.environment_seed, "episode.environment_seed", minimum=0)
        )
        object.__setattr__(self, "policy_seed", integer(self.policy_seed, "episode.policy_seed", minimum=0))
        object.__setattr__(self, "replicate_id", string(self.replicate_id, "episode.replicate_id"))
        object.__setattr__(self, "horizon", integer(self.horizon, "episode.horizon", minimum=1))
        object.__setattr__(self, "protocol", string(self.protocol, "episode.protocol"))

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodeKey":
        obj = fields(
            value,
            {
                "split",
                "task_id",
                "scenario_id",
                "environment_seed",
                "policy_seed",
                "replicate_id",
                "horizon",
                "protocol",
            },
            path="episode",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "environment_seed": self.environment_seed,
            "policy_seed": self.policy_seed,
            "replicate_id": self.replicate_id,
            "horizon": self.horizon,
            "protocol": self.protocol,
        }

    def sampling_key(self) -> tuple[Any, ...]:
        return (
            self.task_id,
            self.scenario_id,
            self.environment_seed,
            self.policy_seed,
            self.replicate_id,
            self.protocol,
        )

    def artifact_id(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.task_id).strip("-.")[:48] or "task"
        digest = mapping_sha256(self.to_mapping())[:16]
        return f"{self.split}-{slug}-{digest}"


def validate_disjoint_splits(episodes: Iterable[EpisodeKey]) -> None:
    items = tuple(episodes)
    by_split = {split: [item for item in items if item.split == split] for split in SPLITS}
    missing = sorted(split for split, rows in by_split.items() if not rows)
    if missing:
        raise StrictSchemaError(f"episode_plan: empty splits {missing}")
    evolve_tasks = {item.task_id for item in by_split["evolve"]}
    selection_tasks = {item.task_id for item in by_split["selection"]}
    transfer_tasks = {item.task_id for item in by_split["transfer"]}
    if evolve_tasks != selection_tasks:
        raise StrictSchemaError("episode_plan: evolve and selection task sets differ")
    if transfer_tasks & evolve_tasks:
        raise StrictSchemaError("episode_plan: transfer tasks overlap evolve tasks")
    seen: dict[tuple[Any, ...], str] = {}
    for item in items:
        sampling = item.sampling_key()
        previous = seen.get(sampling)
        if previous is not None and previous != item.split:
            raise StrictSchemaError(
                f"episode_plan: sampling key appears in both {previous} and {item.split}"
            )
        seen[sampling] = item.split


@dataclass(frozen=True)
class EpisodePlan:
    plan_id: str
    episodes: tuple[EpisodeKey, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("episode_plan.schema_version: expected 1")
        object.__setattr__(self, "plan_id", string(self.plan_id, "episode_plan.plan_id"))
        episodes = tuple(self.episodes)
        if not episodes or any(not isinstance(item, EpisodeKey) for item in episodes):
            raise StrictSchemaError("episode_plan.episodes: expected nonempty EpisodeKey sequence")
        if any(item.split not in SPLITS for item in episodes):
            raise StrictSchemaError("episode_plan.episodes: benchmark rows require BenchmarkPlan")
        if tuple(sorted(episodes)) != episodes:
            raise StrictSchemaError("episode_plan.episodes: expected canonical sorted order")
        if len(set(episodes)) != len(episodes):
            raise StrictSchemaError("episode_plan.episodes: duplicate episode keys")
        ids = [item.artifact_id() for item in episodes]
        if len(set(ids)) != len(ids):
            raise StrictSchemaError("episode_plan.episodes: artifact id collision")
        validate_disjoint_splits(episodes)
        object.__setattr__(self, "episodes", episodes)

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodePlan":
        obj = fields(value, {"schema_version", "plan_id", "episodes"}, path="episode_plan")
        episodes = tuple(
            sorted(EpisodeKey.from_mapping(item) for item in sequence(obj["episodes"], "episode_plan.episodes"))
        )
        return cls(
            schema_version=integer(obj["schema_version"], "episode_plan.schema_version"),
            plan_id=obj["plan_id"],
            episodes=episodes,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EpisodePlan":
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise StrictSchemaError("episode_plan: expected .json file")
        try:
            with source.open("r", encoding="utf-8") as stream:
                return cls.from_mapping(
                    json.load(stream, object_pairs_hook=json_object, parse_constant=reject_json_constant)
                )
        except StrictSchemaError:
            raise
        except Exception as exc:
            raise StrictSchemaError(f"episode_plan: failed to load {source}: {exc}") from exc

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "episodes": [item.to_mapping() for item in self.episodes],
        }

    def resolved_hash(self) -> str:
        return mapping_sha256(self.to_mapping())

    def for_split(self, split: str) -> tuple[EpisodeKey, ...]:
        target = enum(split, SPLITS, "split")
        return tuple(item for item in self.episodes if item.split == target)

@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        name = string(self.name, "artifact.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise StrictSchemaError("artifact.name: invalid filename")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sha256", sha256(self.sha256, "artifact.sha256"))
        object.__setattr__(self, "size_bytes", integer(self.size_bytes, "artifact.size_bytes", minimum=0))

    @classmethod
    def from_mapping(cls, value: Any) -> "ArtifactDescriptor":
        obj = fields(value, {"name", "sha256", "size_bytes"}, path="artifact")
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class EpisodeManifest:
    key: EpisodeKey
    state: str
    success: bool | None
    steps: int
    started_ns: int
    finished_ns: int
    artifacts: tuple[ArtifactDescriptor, ...]
    error: str | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("episode_manifest.schema_version: expected 1")
        if not isinstance(self.key, EpisodeKey):
            raise StrictSchemaError("episode_manifest.key: expected EpisodeKey")
        state = enum(self.state, EPISODE_STATES, "episode_manifest.state")
        if self.success is not None:
            success = boolean(self.success, "episode_manifest.success")
        else:
            success = None
        steps = integer(self.steps, "episode_manifest.steps", minimum=0, maximum=self.key.horizon)
        started = integer(self.started_ns, "episode_manifest.started_ns", minimum=0)
        finished = integer(self.finished_ns, "episode_manifest.finished_ns", minimum=started)
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactDescriptor) for item in artifacts):
            raise StrictSchemaError("episode_manifest.artifacts: expected ArtifactDescriptor entries")
        if tuple(sorted(item.name for item in artifacts)) != tuple(item.name for item in artifacts):
            raise StrictSchemaError("episode_manifest.artifacts: expected sorted names")
        if len({item.name for item in artifacts}) != len(artifacts):
            raise StrictSchemaError("episode_manifest.artifacts: duplicate names")
        if state == "complete" and (success is None or self.error is not None):
            raise StrictSchemaError("episode_manifest: complete requires success and null error")
        if state == "complete" and "trace.jsonl" not in {item.name for item in artifacts}:
            raise StrictSchemaError("episode_manifest: complete requires trace.jsonl")
        if state == "partial" and (success is not None or self.error is not None):
            raise StrictSchemaError("episode_manifest: partial requires null success and error")
        if state == "error" and (success is not None or self.error is None):
            raise StrictSchemaError("episode_manifest: error requires null success and non-null error")
        error = None if self.error is None else string(self.error, "episode_manifest.error")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "started_ns", started)
        object.__setattr__(self, "finished_ns", finished)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "error", error)

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodeManifest":
        obj = fields(
            value,
            {
                "schema_version",
                "key",
                "state",
                "success",
                "steps",
                "started_ns",
                "finished_ns",
                "artifacts",
                "error",
            },
            path="episode_manifest",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "episode_manifest.schema_version"),
            key=EpisodeKey.from_mapping(obj["key"]),
            state=obj["state"],
            success=obj["success"],
            steps=obj["steps"],
            started_ns=obj["started_ns"],
            finished_ns=obj["finished_ns"],
            artifacts=tuple(
                ArtifactDescriptor.from_mapping(item)
                for item in sequence(obj["artifacts"], "episode_manifest.artifacts")
            ),
            error=obj["error"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key.to_mapping(),
            "state": self.state,
            "success": self.success,
            "steps": self.steps,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "artifacts": [item.to_mapping() for item in self.artifacts],
            "error": self.error,
        }
