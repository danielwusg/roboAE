from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from robot_auto_evolve.protocol.schema import StrictSchemaError, enum, fields, integer, sequence, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeKey, EpisodePlan, mapping_sha256
from robot_auto_evolve.provenance.manifest import SPLITS


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, order=True)
class RoboLabBatchKey:
    task_id: str
    scenario_id: str
    environment_seed: int
    protocol: str
    horizon: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", string(self.task_id, "robolab_batch_key.task_id"))
        object.__setattr__(self, "scenario_id", string(self.scenario_id, "robolab_batch_key.scenario_id"))
        object.__setattr__(
            self,
            "environment_seed",
            integer(self.environment_seed, "robolab_batch_key.environment_seed", minimum=0),
        )
        object.__setattr__(self, "protocol", string(self.protocol, "robolab_batch_key.protocol"))
        object.__setattr__(self, "horizon", integer(self.horizon, "robolab_batch_key.horizon", minimum=1))

    @classmethod
    def from_episode(cls, episode: EpisodeKey) -> "RoboLabBatchKey":
        if not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("robolab_batch_key: expected EpisodeKey")
        return cls(
            task_id=episode.task_id,
            scenario_id=episode.scenario_id,
            environment_seed=episode.environment_seed,
            protocol=episode.protocol,
            horizon=episode.horizon,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabBatchKey":
        obj = fields(
            value,
            {"task_id", "scenario_id", "environment_seed", "protocol", "horizon"},
            path="robolab_batch_key",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "environment_seed": self.environment_seed,
            "protocol": self.protocol,
            "horizon": self.horizon,
        }


def _batch_identity(
    key: RoboLabBatchKey,
    episodes: tuple[EpisodeKey, ...],
    vector_batch_size: int,
) -> str:
    return mapping_sha256(
        {
            "schema_version": 1,
            "key": key.to_mapping(),
            "vector_batch_size": vector_batch_size,
            "episodes": [episode.to_mapping() for episode in episodes],
        }
    )


@dataclass(frozen=True)
class RoboLabBatch:
    batch_id: str
    key: RoboLabBatchKey
    vector_batch_size: int
    episodes: tuple[EpisodeKey, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab_batch.schema_version: expected 1")
        if not isinstance(self.key, RoboLabBatchKey):
            raise StrictSchemaError("robolab_batch.key: expected RoboLabBatchKey")
        size = integer(self.vector_batch_size, "robolab_batch.vector_batch_size", minimum=1)
        episodes = tuple(self.episodes)
        if not episodes or any(not isinstance(episode, EpisodeKey) for episode in episodes):
            raise StrictSchemaError("robolab_batch.episodes: expected nonempty EpisodeKey sequence")
        if len(episodes) > size:
            raise StrictSchemaError("robolab_batch.episodes: exceeds vector batch size")
        if tuple(sorted(episodes)) != episodes:
            raise StrictSchemaError("robolab_batch.episodes: expected canonical plan order")
        if len(set(episodes)) != len(episodes):
            raise StrictSchemaError("robolab_batch.episodes: duplicate episode keys")
        if len({episode.split for episode in episodes}) != 1:
            raise StrictSchemaError("robolab_batch.episodes: split differs")
        if any(RoboLabBatchKey.from_episode(episode) != self.key for episode in episodes):
            raise StrictSchemaError("robolab_batch.episodes: grouping key differs")
        batch_id = sha256(self.batch_id, "robolab_batch.batch_id")
        if batch_id != _batch_identity(self.key, episodes, size):
            raise StrictSchemaError("robolab_batch.batch_id: identity mismatch")
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "vector_batch_size", size)
        object.__setattr__(self, "episodes", episodes)

    @classmethod
    def create(
        cls,
        key: RoboLabBatchKey,
        episodes: tuple[EpisodeKey, ...],
        vector_batch_size: int,
    ) -> "RoboLabBatch":
        if not isinstance(key, RoboLabBatchKey):
            raise StrictSchemaError("robolab_batch.key: expected RoboLabBatchKey")
        rows = tuple(episodes)
        size = integer(vector_batch_size, "robolab_batch.vector_batch_size", minimum=1)
        return cls(
            batch_id=_batch_identity(key, rows, size),
            key=key,
            vector_batch_size=size,
            episodes=rows,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabBatch":
        obj = fields(
            value,
            {"schema_version", "batch_id", "key", "vector_batch_size", "episodes"},
            path="robolab_batch",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "robolab_batch.schema_version"),
            batch_id=obj["batch_id"],
            key=RoboLabBatchKey.from_mapping(obj["key"]),
            vector_batch_size=obj["vector_batch_size"],
            episodes=tuple(
                EpisodeKey.from_mapping(episode)
                for episode in sequence(obj["episodes"], "robolab_batch.episodes")
            ),
        )

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(episode.artifact_id() for episode in self.episodes)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "key": self.key.to_mapping(),
            "vector_batch_size": self.vector_batch_size,
            "episodes": [episode.to_mapping() for episode in self.episodes],
        }


def _group_batches(
    episodes: tuple[EpisodeKey, ...],
    vector_batch_size: int,
) -> tuple[RoboLabBatch, ...]:
    groups: dict[RoboLabBatchKey, list[EpisodeKey]] = {}
    for episode in episodes:
        groups.setdefault(RoboLabBatchKey.from_episode(episode), []).append(episode)
    batches: list[RoboLabBatch] = []
    for key, rows in groups.items():
        for start in range(0, len(rows), vector_batch_size):
            batches.append(RoboLabBatch.create(key, tuple(rows[start : start + vector_batch_size]), vector_batch_size))
    return tuple(batches)


@dataclass(frozen=True)
class RoboLabBatchSchedule:
    plan_id: str
    plan_sha256: str
    split: str
    vector_batch_size: int
    episode_order: tuple[EpisodeKey, ...]
    batches: tuple[RoboLabBatch, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab_batch_schedule.schema_version: expected 1")
        plan_id = string(self.plan_id, "robolab_batch_schedule.plan_id")
        plan_hash = sha256(self.plan_sha256, "robolab_batch_schedule.plan_sha256")
        split = enum(self.split, SPLITS, "robolab_batch_schedule.split")
        size = integer(self.vector_batch_size, "robolab_batch_schedule.vector_batch_size", minimum=1)
        order = tuple(self.episode_order)
        if not order or any(not isinstance(episode, EpisodeKey) for episode in order):
            raise StrictSchemaError("robolab_batch_schedule.episode_order: expected nonempty EpisodeKey sequence")
        if tuple(sorted(order)) != order:
            raise StrictSchemaError("robolab_batch_schedule.episode_order: expected canonical plan order")
        if any(episode.split != split for episode in order):
            raise StrictSchemaError("robolab_batch_schedule.episode_order: split differs")
        if len(set(order)) != len(order):
            raise StrictSchemaError("robolab_batch_schedule.episode_order: duplicate episode keys")
        batches = tuple(self.batches)
        if not batches or any(not isinstance(batch, RoboLabBatch) for batch in batches):
            raise StrictSchemaError("robolab_batch_schedule.batches: expected nonempty RoboLabBatch sequence")
        if any(batch.vector_batch_size != size for batch in batches):
            raise StrictSchemaError("robolab_batch_schedule.batches: vector batch size differs")
        if batches != _group_batches(order, size):
            raise StrictSchemaError("robolab_batch_schedule.batches: deterministic grouping differs")
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "plan_sha256", plan_hash)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "vector_batch_size", size)
        object.__setattr__(self, "episode_order", order)
        object.__setattr__(self, "batches", batches)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabBatchSchedule":
        obj = fields(
            value,
            {
                "schema_version",
                "plan_id",
                "plan_sha256",
                "split",
                "vector_batch_size",
                "episode_order",
                "batches",
            },
            path="robolab_batch_schedule",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "robolab_batch_schedule.schema_version"),
            plan_id=obj["plan_id"],
            plan_sha256=obj["plan_sha256"],
            split=obj["split"],
            vector_batch_size=obj["vector_batch_size"],
            episode_order=tuple(
                EpisodeKey.from_mapping(episode)
                for episode in sequence(obj["episode_order"], "robolab_batch_schedule.episode_order")
            ),
            batches=tuple(
                RoboLabBatch.from_mapping(batch)
                for batch in sequence(obj["batches"], "robolab_batch_schedule.batches")
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "split": self.split,
            "vector_batch_size": self.vector_batch_size,
            "episode_order": [episode.to_mapping() for episode in self.episode_order],
            "batches": [batch.to_mapping() for batch in self.batches],
        }

    def validate_plan(self, plan: EpisodePlan) -> None:
        if not isinstance(plan, EpisodePlan):
            raise StrictSchemaError("robolab_batch_schedule.plan: expected EpisodePlan")
        if plan.plan_id != self.plan_id or plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("robolab_batch_schedule.plan: identity mismatch")
        if plan.for_split(self.split) != self.episode_order:
            raise StrictSchemaError("robolab_batch_schedule.plan: episode order differs")

    def order_results(self, results: Mapping[str, ResultT]) -> tuple[tuple[EpisodeKey, ResultT], ...]:
        if not isinstance(results, Mapping) or any(type(key) is not str for key in results):
            raise StrictSchemaError("robolab_batch_schedule.results: expected string-keyed mapping")
        expected = {episode.artifact_id() for episode in self.episode_order}
        if set(results) != expected:
            raise StrictSchemaError("robolab_batch_schedule.results: episode coverage differs")
        return tuple((episode, results[episode.artifact_id()]) for episode in self.episode_order)


def build_robolab_batch_schedule(
    plan: EpisodePlan,
    *,
    split: str,
    vector_batch_size: int,
) -> RoboLabBatchSchedule:
    if not isinstance(plan, EpisodePlan):
        raise StrictSchemaError("robolab_batch_schedule.plan: expected EpisodePlan")
    target = enum(split, SPLITS, "robolab_batch_schedule.split")
    size = integer(vector_batch_size, "robolab_batch_schedule.vector_batch_size", minimum=1)
    episodes = plan.for_split(target)
    return RoboLabBatchSchedule(
        plan_id=plan.plan_id,
        plan_sha256=plan.resolved_hash(),
        split=target,
        vector_batch_size=size,
        episode_order=episodes,
        batches=_group_batches(episodes, size),
    )


def build_robolab_benchmark_batches(
    plan: BenchmarkPlan,
    *,
    vector_batch_size: int,
) -> tuple[RoboLabBatch, ...]:
    if not isinstance(plan, BenchmarkPlan):
        raise StrictSchemaError("robolab_benchmark_batches.plan: expected BenchmarkPlan")
    size = integer(vector_batch_size, "robolab_benchmark_batches.vector_batch_size", minimum=1)
    return _group_batches(plan.episodes, size)
