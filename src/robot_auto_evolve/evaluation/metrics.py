from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from robot_auto_evolve.protocol.schema import StrictSchemaError, fields
from robot_auto_evolve.provenance import EpisodeKey, EpisodeManifest


@dataclass(frozen=True)
class EpisodeOutcome:
    key: EpisodeKey
    success: bool

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise StrictSchemaError("outcome.key: expected EpisodeKey")
        if type(self.success) is not bool:
            raise StrictSchemaError("outcome.success: expected bool")

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodeOutcome":
        obj = fields(value, {"key", "success"}, path="outcome")
        return cls(key=EpisodeKey.from_mapping(obj["key"]), success=obj["success"])

    @classmethod
    def from_manifest(cls, manifest: EpisodeManifest) -> "EpisodeOutcome":
        if manifest.state != "complete" or manifest.success is None:
            raise StrictSchemaError("outcome: episode manifest is not complete")
        return cls(key=manifest.key, success=manifest.success)

    def to_mapping(self) -> dict[str, Any]:
        return {"key": self.key.to_mapping(), "success": self.success}


@dataclass(frozen=True)
class TaskMacroMetrics:
    split: str
    task_success: Mapping[str, float]
    task_counts: Mapping[str, int]
    macro_success: float
    pooled_success: float
    n_tasks: int
    n_episodes: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "task_success": dict(self.task_success),
            "task_counts": dict(self.task_counts),
            "macro_success": self.macro_success,
            "pooled_success": self.pooled_success,
            "n_tasks": self.n_tasks,
            "n_episodes": self.n_episodes,
        }


def _outcomes(values: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]]) -> tuple[EpisodeOutcome, ...]:
    result: list[EpisodeOutcome] = []
    for value in values:
        if isinstance(value, EpisodeOutcome):
            outcome = value
        elif isinstance(value, EpisodeManifest):
            outcome = EpisodeOutcome.from_manifest(value)
        elif isinstance(value, Mapping):
            outcome = EpisodeOutcome.from_mapping(value)
        else:
            raise StrictSchemaError("outcomes: unsupported entry")
        result.append(outcome)
    if not result:
        raise StrictSchemaError("outcomes: empty")
    keys = [item.key for item in result]
    if len(set(keys)) != len(keys):
        raise StrictSchemaError("outcomes: duplicate episode keys")
    return tuple(sorted(result, key=lambda item: item.key))


def compute_task_macro_metrics(
    values: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
) -> TaskMacroMetrics:
    outcomes = _outcomes(values)
    splits = {item.key.split for item in outcomes}
    if len(splits) != 1:
        raise StrictSchemaError("outcomes: expected one split")
    by_task: dict[str, list[bool]] = {}
    for outcome in outcomes:
        by_task.setdefault(outcome.key.task_id, []).append(outcome.success)
    task_success = {task: float(np.mean(rows)) for task, rows in sorted(by_task.items())}
    task_counts = {task: len(rows) for task, rows in sorted(by_task.items())}
    return TaskMacroMetrics(
        split=next(iter(splits)),
        task_success=task_success,
        task_counts=task_counts,
        macro_success=float(np.mean(list(task_success.values()))),
        pooled_success=float(np.mean([item.success for item in outcomes])),
        n_tasks=len(by_task),
        n_episodes=len(outcomes),
    )


def task_macro_success(values: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]]) -> float:
    return compute_task_macro_metrics(values).macro_success
