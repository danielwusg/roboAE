from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_auto_evolve.protocol.schema import (
    StrictSchemaError,
    fields,
    integer,
    json_object,
    reject_json_constant,
    sequence,
    string,
)

from .manifest import EpisodeKey, mapping_sha256


@dataclass(frozen=True)
class BenchmarkPlan:
    plan_id: str
    model_route: str
    episodes: tuple[EpisodeKey, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("benchmark_plan.schema_version: expected 1")
        object.__setattr__(self, "plan_id", string(self.plan_id, "benchmark_plan.plan_id"))
        object.__setattr__(self, "model_route", string(self.model_route, "benchmark_plan.model_route"))
        episodes = tuple(self.episodes)
        if not episodes or any(not isinstance(item, EpisodeKey) for item in episodes):
            raise StrictSchemaError("benchmark_plan.episodes: expected nonempty EpisodeKey sequence")
        if any(item.split != "benchmark" for item in episodes):
            raise StrictSchemaError("benchmark_plan.episodes: expected only benchmark rows")
        if tuple(sorted(episodes)) != episodes:
            raise StrictSchemaError("benchmark_plan.episodes: expected canonical sorted order")
        if len(set(episodes)) != len(episodes):
            raise StrictSchemaError("benchmark_plan.episodes: duplicate episode keys")
        artifact_ids = [item.artifact_id() for item in episodes]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise StrictSchemaError("benchmark_plan.episodes: artifact id collision")
        object.__setattr__(self, "episodes", episodes)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkPlan":
        obj = fields(
            value,
            {"schema_version", "plan_id", "model_route", "episodes"},
            path="benchmark_plan",
        )
        episodes = tuple(
            sorted(EpisodeKey.from_mapping(item) for item in sequence(obj["episodes"], "benchmark_plan.episodes"))
        )
        return cls(
            schema_version=integer(obj["schema_version"], "benchmark_plan.schema_version"),
            plan_id=obj["plan_id"],
            model_route=obj["model_route"],
            episodes=episodes,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkPlan":
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise StrictSchemaError("benchmark_plan: expected .json file")
        try:
            with source.open("r", encoding="utf-8") as stream:
                return cls.from_mapping(
                    json.load(stream, object_pairs_hook=json_object, parse_constant=reject_json_constant)
                )
        except StrictSchemaError:
            raise
        except Exception as exc:
            raise StrictSchemaError(f"benchmark_plan: failed to load {source}: {exc}") from exc

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "model_route": self.model_route,
            "episodes": [item.to_mapping() for item in self.episodes],
        }

    def resolved_hash(self) -> str:
        return mapping_sha256(self.to_mapping())
