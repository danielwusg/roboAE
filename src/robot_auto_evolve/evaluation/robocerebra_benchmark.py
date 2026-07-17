from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from robot_auto_evolve.benchmarks.robocerebra import (
    CONDITIONS,
    PAPER_TRIALS_PER_CASE,
    PUBLIC_PROTOCOL,
    load_case_catalog,
    parse_task_id,
)
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.benchmark import CanonicalBenchmarkEvaluator, verify_benchmark_output
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import boolean, fields, integer, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest


MODEL_ROUTE = "smolvla_robocerebra"
METRIC = "equal_condition_case_macro_success"
SUITE = "robocerebra_public60"


def _relative_json(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
        raise StrictSchemaError(f"{name}: expected safe relative JSON path")
    return path.as_posix()


def robocerebra_metrics(manifests: tuple[EpisodeManifest, ...], episode_root: Path) -> dict[str, Any]:
    case_metrics = {}
    condition_metrics = {}
    subtask_values = {}
    for condition in CONDITIONS:
        condition_scores = []
        condition_subtasks = []
        for case_number in range(1, 11):
            case_id = f"case{case_number}"
            task = f"robocerebra_public60::{condition}::{case_id}"
            rows = [item for item in manifests if item.key.task_id == task]
            if len(rows) != PAPER_TRIALS_PER_CASE:
                raise StrictSchemaError("RoboCerebra benchmark case episode count differs")
            success = sum(bool(item.success) for item in rows) / len(rows)
            subtask = []
            for item in rows:
                path = episode_root / item.key.artifact_id() / "private_metrics.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("kind") != "private_evaluator_metrics":
                    raise StrictSchemaError("RoboCerebra private metric artifact differs")
                subtask.append(float(value["metrics"]["subtask_completion"]))
            case_metrics[task] = {"episode_success": success, "subtask_completion": sum(subtask) / len(subtask)}
            condition_scores.append(success)
            condition_subtasks.extend(subtask)
        condition_metrics[condition] = {
            "case_macro_success": sum(condition_scores) / len(condition_scores),
            "episode_subtask_completion": sum(condition_subtasks) / len(condition_subtasks),
        }
        subtask_values[condition] = condition_metrics[condition]["episode_subtask_completion"]
    score = sum(value["case_macro_success"] for value in condition_metrics.values()) / len(condition_metrics)
    return {
        "metric": METRIC,
        "score": score,
        "condition_metrics": condition_metrics,
        "case_metrics": case_metrics,
        "secondary_equal_condition_subtask_completion": sum(subtask_values.values()) / len(subtask_values),
    }


@dataclass(frozen=True)
class RoboCerebraBenchmarkConfig:
    benchmark_id: str
    protocol: str
    trials_per_case: int
    plan_path: str
    plan_sha256: str
    profile_path: str
    public_substitute: bool
    paper_checkpoint_available: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.protocol != PUBLIC_PROTOCOL:
            raise StrictSchemaError("RoboCerebra benchmark version or protocol differs")
        object.__setattr__(self, "benchmark_id", string(self.benchmark_id, "robocerebra_benchmark.benchmark_id"))
        if integer(self.trials_per_case, "robocerebra_benchmark.trials_per_case", minimum=1) != 10:
            raise StrictSchemaError("RoboCerebra benchmark requires 10 trials per case")
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "robocerebra_benchmark.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "robocerebra_benchmark.plan.sha256"))
        object.__setattr__(self, "profile_path", _relative_json(self.profile_path, "robocerebra_benchmark.profile"))
        if not self.public_substitute or self.paper_checkpoint_available:
            raise StrictSchemaError("RoboCerebra benchmark must identify the public policy substitution")

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboCerebraBenchmarkConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "model_route",
                "metric",
                "protocol",
                "trials_per_case",
                "plan",
                "profile",
                "public_substitute",
                "paper_checkpoint_available",
            },
            path="robocerebra_benchmark",
        )
        if obj["model_route"] != MODEL_ROUTE or obj["metric"] != METRIC:
            raise StrictSchemaError("RoboCerebra benchmark route or metric differs")
        plan = fields(obj["plan"], {"path", "sha256"}, path="robocerebra_benchmark.plan")
        return cls(
            schema_version=integer(obj["schema_version"], "robocerebra_benchmark.schema_version"),
            benchmark_id=obj["benchmark_id"],
            protocol=obj["protocol"],
            trials_per_case=obj["trials_per_case"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profile_path=obj["profile"],
            public_substitute=boolean(obj["public_substitute"], "robocerebra_benchmark.public_substitute"),
            paper_checkpoint_available=boolean(
                obj["paper_checkpoint_available"], "robocerebra_benchmark.paper_checkpoint_available"
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RoboCerebraBenchmarkConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        plan = BenchmarkPlan.load(Path(project_root).resolve() / self.plan_path)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("RoboCerebra benchmark plan hash differs")
        self.validate_plan(plan, project_root)
        return plan

    def load_profile(self, project_root: str | Path) -> Profile:
        root = Path(project_root).resolve()
        profile = Profile.load(root / self.profile_path, project_root=root)
        if profile.environment.suite != SUITE:
            raise StrictSchemaError("RoboCerebra benchmark profile suite differs")
        if {item.identity.service_name for item in profile.policy.replicas} != {MODEL_ROUTE}:
            raise StrictSchemaError("RoboCerebra benchmark profile policy differs")
        return profile

    def validate_plan(self, plan: BenchmarkPlan, project_root: str | Path | None = None) -> None:
        if plan.plan_id != self.benchmark_id or plan.model_route != MODEL_ROUTE or len(plan.episodes) != 600:
            raise StrictSchemaError("RoboCerebra benchmark plan identity or count differs")
        if project_root is None:
            catalog_path = Path(__file__).resolve().parents[3] / "manifests" / "robocerebra_cases.json"
        else:
            catalog_path = Path(project_root).resolve() / "manifests" / "robocerebra_cases.json"
        catalog = {item.task_id: item for item in load_case_catalog(catalog_path)}
        counts = {task: 0 for task in catalog}
        scenarios = {task: set() for task in catalog}
        for episode in plan.episodes:
            parse_task_id(episode.task_id)
            if episode.task_id not in catalog or episode.protocol != PUBLIC_PROTOCOL:
                raise StrictSchemaError("RoboCerebra benchmark task or protocol differs")
            if episode.horizon != catalog[episode.task_id].horizon:
                raise StrictSchemaError("RoboCerebra benchmark horizon differs")
            counts[episode.task_id] += 1
            scenarios[episode.task_id].add(episode.scenario_id)
        if set(counts.values()) != {10} or any(len(value) != 10 for value in scenarios.values()):
            raise StrictSchemaError("RoboCerebra benchmark trial grid differs")

    def evaluator(self, profile: Profile, plan: BenchmarkPlan, **kwargs: Any) -> CanonicalBenchmarkEvaluator:
        return CanonicalBenchmarkEvaluator(
            {SUITE: profile},
            plan,
            task_suites={item.task_id: SUITE for item in plan.episodes},
            artifact_metric_function=robocerebra_metrics,
            **kwargs,
        )


def verify_robocerebra_benchmark_output(
    path: str | Path,
    config: RoboCerebraBenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    return verify_benchmark_output(
        path,
        plan_validator=lambda plan: config.validate_plan(plan, project_root),
        artifact_metric_function=robocerebra_metrics,
        profile_suites=(SUITE,),
    )
