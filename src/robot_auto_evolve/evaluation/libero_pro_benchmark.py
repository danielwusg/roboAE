from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from robot_auto_evolve.benchmarks.libero_pro import (
    HARNESS_PROTOCOLS,
    HARNESS_SUITES,
    HARNESS_TASK_SUITE,
    base_tasks,
    split_suite,
    split_task_id,
    task_id,
)
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.metrics import compute_task_macro_metrics
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import fields, integer, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest

from .benchmark import verify_benchmark_output


def _relative_json(value: Any, path: str) -> str:
    result = PurePosixPath(string(value, path))
    if result.is_absolute() or ".." in result.parts or result.suffix.lower() != ".json":
        raise StrictSchemaError(f"{path}: expected safe relative JSON path")
    return result.as_posix()


def libero_pro_metrics(manifests: tuple[EpisodeManifest, ...]) -> dict[str, Any]:
    cell_metrics = {}
    for suite in HARNESS_SUITES:
        rows = tuple(item for item in manifests if HARNESS_TASK_SUITE[item.key.task_id] == suite)
        cell_metrics[suite] = compute_task_macro_metrics(rows).to_mapping()
    all_tasks = compute_task_macro_metrics(manifests).to_mapping()
    score = sum(value["macro_success"] for value in cell_metrics.values()) / len(cell_metrics)
    return {
        "metric": "equal_cell_task_macro_success",
        "score": score,
        "cell_metrics": cell_metrics,
        "all_task_macro": all_tasks,
    }


@dataclass(frozen=True)
class LiberoProBenchmarkConfig:
    benchmark_id: str
    model_route: str
    metric: str
    trials_per_task: int
    plan_path: str
    plan_sha256: str
    profiles: Mapping[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("benchmark_config.schema_version: expected 1")
        object.__setattr__(self, "benchmark_id", string(self.benchmark_id, "benchmark_config.benchmark_id"))
        if self.model_route != "rlinf_pi05_libero_pro":
            raise StrictSchemaError("benchmark_config.model_route: expected exact RLinf pi0.5 route")
        if self.metric != "equal_cell_task_macro_success":
            raise StrictSchemaError("benchmark_config.metric: expected equal_cell_task_macro_success")
        if integer(self.trials_per_task, "benchmark_config.trials_per_task", minimum=1) != 10:
            raise StrictSchemaError("benchmark_config.trials_per_task: Harness paper-v3 requires 10")
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "benchmark_config.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "benchmark_config.plan.sha256"))
        profiles = dict(self.profiles)
        if set(profiles) != set(HARNESS_SUITES):
            raise StrictSchemaError("benchmark_config.profiles: expected all eight Harness paper-v3 cells")
        object.__setattr__(
            self,
            "profiles",
            {suite: _relative_json(profiles[suite], f"benchmark_config.profiles.{suite}") for suite in HARNESS_SUITES},
        )

    @classmethod
    def load(cls, path: str | Path) -> "LiberoProBenchmarkConfig":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        obj = fields(
            value,
            {"schema_version", "benchmark_id", "model_route", "metric", "trials_per_task", "plan", "profiles"},
            path="benchmark_config",
        )
        plan = fields(obj["plan"], {"path", "sha256"}, path="benchmark_config.plan")
        profiles = fields(obj["profiles"], set(HARNESS_SUITES), path="benchmark_config.profiles")
        return cls(
            schema_version=obj["schema_version"],
            benchmark_id=obj["benchmark_id"],
            model_route=obj["model_route"],
            metric=obj["metric"],
            trials_per_task=obj["trials_per_task"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profiles=profiles,
        )

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        root = Path(project_root).resolve()
        plan = BenchmarkPlan.load(root / self.plan_path)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("benchmark_config.plan: hash mismatch")
        if plan.plan_id != self.benchmark_id or plan.model_route != self.model_route:
            raise StrictSchemaError("benchmark_config.plan: identity mismatch")
        return plan

    def load_profiles(self, project_root: str | Path) -> dict[str, Profile]:
        root = Path(project_root).resolve()
        return {suite: Profile.load(root / self.profiles[suite], project_root=root) for suite in HARNESS_SUITES}

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if not isinstance(plan, BenchmarkPlan) or plan.model_route != self.model_route or len(plan.episodes) != 800:
            raise StrictSchemaError("benchmark_config: expected exact 800-row RLinf pi0.5 plan")
        expected_tasks = set(HARNESS_TASK_SUITE)
        if {row.task_id for row in plan.episodes} != expected_tasks:
            raise StrictSchemaError("benchmark_config: expected 80 namespaced LIBERO-Pro tasks")
        by_task = {task: [] for task in expected_tasks}
        for row in plan.episodes:
            by_task[row.task_id].append(row)
            suite, task_slug = split_task_id(row.task_id)
            base_suite, _ = split_suite(suite)
            if task_id(suite, task_slug) != row.task_id or task_slug not in base_tasks(base_suite):
                raise StrictSchemaError("benchmark_config: invalid namespaced task")
            if row.protocol not in HARNESS_PROTOCOLS[suite] or row.horizon != HARNESS_PROTOCOLS[suite][row.protocol]:
                raise StrictSchemaError("benchmark_config: protocol or horizon differs")
            if row.environment_seed != 7 or row.policy_seed != 7:
                raise StrictSchemaError("benchmark_config: fixed paper baseline seed differs")
        for rows in by_task.values():
            if len(rows) != 10 or {row.scenario_id for row in rows} != {f"init_state_{index:02d}" for index in range(1, 11)}:
                raise StrictSchemaError("benchmark_config: held-out states must be s1 through s10")


def verify_libero_pro_benchmark_output(
    path: str | Path,
    config: LiberoProBenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    profiles = config.load_profiles(project_root)
    return verify_benchmark_output(
        path,
        plan_validator=config.validate_plan,
        metric_function=libero_pro_metrics,
        profile_suites=tuple(profiles),
    )
