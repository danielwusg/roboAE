from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from robot_auto_evolve.benchmarks.robocasa365 import TARGET_TASK_GROUPS, TARGET_TASK_HORIZONS
from robot_auto_evolve.benchmarks.robocasa365_worker import (
    PUBLIC_BENCHMARK_PROTOCOL,
    PUBLIC_ENVIRONMENTS,
    PUBLIC_EPISODES_PER_ENVIRONMENT,
    public_episode_coordinates,
)
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.benchmark import CanonicalBenchmarkEvaluator, verify_benchmark_output
from robot_auto_evolve.evaluation.metrics import compute_task_macro_metrics
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import fields, integer, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest


MODEL_ROUTE = "rldx_robocasa365"
METRIC = "equal_group_task_macro_success"
EPISODES_PER_TASK = PUBLIC_ENVIRONMENTS * PUBLIC_EPISODES_PER_ENVIRONMENT


def _relative_json(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
        raise StrictSchemaError(f"{name}: expected safe relative JSON path")
    return path.as_posix()


def robocasa365_metrics(manifests: tuple[EpisodeManifest, ...]) -> dict[str, Any]:
    task_metrics = {
        task: compute_task_macro_metrics(tuple(item for item in manifests if item.key.task_id == task)).to_mapping()
        for task in sorted(TARGET_TASK_HORIZONS)
    }
    group_metrics = {}
    for group, tasks in TARGET_TASK_GROUPS.items():
        rows = tuple(item for item in manifests if item.key.task_id in tasks)
        group_metrics[group] = compute_task_macro_metrics(rows).to_mapping()
    score = sum(value["macro_success"] for value in group_metrics.values()) / len(group_metrics)
    return {
        "metric": METRIC,
        "score": score,
        "group_metrics": group_metrics,
        "task_metrics": task_metrics,
        "all_task_macro": compute_task_macro_metrics(manifests).to_mapping(),
    }


@dataclass(frozen=True)
class RoboCasa365BenchmarkConfig:
    benchmark_id: str
    protocol: str
    episodes_per_task: int
    plan_path: str
    plan_sha256: str
    profile_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robocasa365_benchmark.schema_version: expected 1")
        object.__setattr__(self, "benchmark_id", string(self.benchmark_id, "robocasa365_benchmark.benchmark_id"))
        if self.protocol != PUBLIC_BENCHMARK_PROTOCOL:
            raise StrictSchemaError("robocasa365_benchmark.protocol differs")
        count = integer(self.episodes_per_task, "robocasa365_benchmark.episodes_per_task", minimum=1)
        if count != EPISODES_PER_TASK:
            raise StrictSchemaError("robocasa365_benchmark.episodes_per_task differs")
        object.__setattr__(self, "episodes_per_task", count)
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "robocasa365_benchmark.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "robocasa365_benchmark.plan.sha256"))
        object.__setattr__(self, "profile_path", _relative_json(self.profile_path, "robocasa365_benchmark.profile"))

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboCasa365BenchmarkConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "model_route",
                "metric",
                "protocol",
                "episodes_per_task",
                "plan",
                "profile",
            },
            path="robocasa365_benchmark",
        )
        if obj["model_route"] != MODEL_ROUTE or obj["metric"] != METRIC:
            raise StrictSchemaError("robocasa365_benchmark route or metric differs")
        plan = fields(obj["plan"], {"path", "sha256"}, path="robocasa365_benchmark.plan")
        return cls(
            schema_version=integer(obj["schema_version"], "robocasa365_benchmark.schema_version"),
            benchmark_id=obj["benchmark_id"],
            protocol=obj["protocol"],
            episodes_per_task=obj["episodes_per_task"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profile_path=obj["profile"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "RoboCasa365BenchmarkConfig":
        import json

        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        plan = BenchmarkPlan.load(Path(project_root).resolve() / self.plan_path)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("robocasa365_benchmark.plan: hash mismatch")
        self.validate_plan(plan)
        return plan

    def load_profile(self, project_root: str | Path) -> Profile:
        root = Path(project_root).resolve()
        profile = Profile.load(root / self.profile_path, project_root=root)
        if profile.environment.suite != "robocasa365_target":
            raise StrictSchemaError("robocasa365_benchmark.profile suite differs")
        if {item.identity.service_name for item in profile.policy.replicas} != {MODEL_ROUTE}:
            raise StrictSchemaError("robocasa365_benchmark.profile policy differs")
        return profile

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if plan.plan_id != self.benchmark_id or plan.model_route != MODEL_ROUTE:
            raise StrictSchemaError("robocasa365_benchmark.plan identity differs")
        counts = {task: 0 for task in TARGET_TASK_HORIZONS}
        scenarios = {task: set() for task in TARGET_TASK_HORIZONS}
        for episode in plan.episodes:
            if episode.task_id not in counts or episode.protocol != self.protocol:
                raise StrictSchemaError("robocasa365_benchmark.plan task or protocol differs")
            if episode.horizon != TARGET_TASK_HORIZONS[episode.task_id]:
                raise StrictSchemaError("robocasa365_benchmark.plan horizon differs")
            public_episode_coordinates(episode)
            counts[episode.task_id] += 1
            scenarios[episode.task_id].add(episode.scenario_id)
        if set(counts.values()) != {self.episodes_per_task}:
            raise StrictSchemaError("robocasa365_benchmark.plan task counts differ")
        if any(len(value) != self.episodes_per_task for value in scenarios.values()):
            raise StrictSchemaError("robocasa365_benchmark.plan scenario grid differs")

    def evaluator(self, profile: Profile, plan: BenchmarkPlan, **kwargs: Any) -> CanonicalBenchmarkEvaluator:
        suite = profile.environment.suite
        return CanonicalBenchmarkEvaluator(
            {suite: profile},
            plan,
            task_suites={task: suite for task in TARGET_TASK_HORIZONS},
            metric_function=robocasa365_metrics,
            **kwargs,
        )


def verify_robocasa365_benchmark_output(
    path: str | Path,
    config: RoboCasa365BenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    return verify_benchmark_output(
        path,
        plan_validator=config.validate_plan,
        metric_function=robocasa365_metrics,
        profile_suites=(config.load_profile(project_root).environment.suite,),
    )
