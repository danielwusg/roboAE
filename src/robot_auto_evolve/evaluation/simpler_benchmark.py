from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.benchmark import CanonicalBenchmarkEvaluator, verify_benchmark_output
from robot_auto_evolve.evaluation.metrics import compute_task_macro_metrics
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import boolean, fields, integer, mapping, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest


def _relative_json(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
        raise StrictSchemaError(f"{name}: expected safe relative JSON path")
    return path.as_posix()


def simpler_metrics(manifests: tuple[EpisodeManifest, ...]) -> dict[str, Any]:
    tasks = sorted({item.key.task_id for item in manifests})
    task_metrics = {
        task: compute_task_macro_metrics(tuple(item for item in manifests if item.key.task_id == task)).to_mapping()
        for task in tasks
    }
    overall = compute_task_macro_metrics(manifests).to_mapping()
    return {
        "metric": "task_macro_success",
        "score": overall["macro_success"],
        "task_metrics": task_metrics,
        "all_task_macro": overall,
    }


@dataclass(frozen=True)
class SimplerBenchmarkConfig:
    benchmark_id: str
    model_route: str
    protocol: str
    paper_comparable_scope: bool
    task_episode_counts: Mapping[str, int]
    plan_path: str
    plan_sha256: str
    profile_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("simpler_benchmark.schema_version: expected 1")
        object.__setattr__(self, "benchmark_id", string(self.benchmark_id, "simpler_benchmark.benchmark_id"))
        object.__setattr__(self, "model_route", string(self.model_route, "simpler_benchmark.model_route"))
        object.__setattr__(self, "protocol", string(self.protocol, "simpler_benchmark.protocol"))
        counts = {
            string(task, "simpler_benchmark.task_episode_counts.task"): integer(
                count, f"simpler_benchmark.task_episode_counts.{task}", minimum=1
            )
            for task, count in mapping(self.task_episode_counts, "simpler_benchmark.task_episode_counts").items()
        }
        if not counts:
            raise StrictSchemaError("simpler_benchmark.task_episode_counts: expected tasks")
        object.__setattr__(self, "task_episode_counts", counts)
        object.__setattr__(self, "paper_comparable_scope", boolean(self.paper_comparable_scope, "simpler_benchmark.paper_comparable_scope"))
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "simpler_benchmark.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "simpler_benchmark.plan.sha256"))
        object.__setattr__(self, "profile_path", _relative_json(self.profile_path, "simpler_benchmark.profile"))

    @classmethod
    def from_mapping(cls, value: Any) -> "SimplerBenchmarkConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "model_route",
                "metric",
                "protocol",
                "paper_comparable_scope",
                "task_episode_counts",
                "plan",
                "profile",
            },
            path="simpler_benchmark",
        )
        if obj["metric"] != "task_macro_success":
            raise StrictSchemaError("simpler_benchmark.metric: expected task_macro_success")
        plan = fields(obj["plan"], {"path", "sha256"}, path="simpler_benchmark.plan")
        return cls(
            schema_version=integer(obj["schema_version"], "simpler_benchmark.schema_version"),
            benchmark_id=obj["benchmark_id"],
            model_route=obj["model_route"],
            protocol=obj["protocol"],
            paper_comparable_scope=obj["paper_comparable_scope"],
            task_episode_counts=obj["task_episode_counts"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profile_path=obj["profile"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimplerBenchmarkConfig":
        import json

        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        plan = BenchmarkPlan.load(Path(project_root).resolve() / self.plan_path)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("simpler_benchmark.plan: hash mismatch")
        self.validate_plan(plan)
        return plan

    def load_profile(self, project_root: str | Path) -> Profile:
        root = Path(project_root).resolve()
        profile = Profile.load(root / self.profile_path, project_root=root)
        expected_suite = {
            "openvla_simpler_google_va": "simpler_google_va",
            "openvla_simpler_google_vm": "simpler_google_vm",
            "xvla_simpler_google_va": "simpler_google_va",
            "xvla_simpler_google_vm": "simpler_google_vm",
            "xvla_simpler_widowx": "simpler_widowx_vm",
        }.get(self.model_route)
        if profile.environment.suite != expected_suite:
            raise StrictSchemaError("simpler_benchmark.profile: route suite differs")
        if {item.identity.service_name for item in profile.policy.replicas} != {self.model_route}:
            raise StrictSchemaError("simpler_benchmark.profile: policy service differs")
        return profile

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if plan.plan_id != self.benchmark_id or plan.model_route != self.model_route:
            raise StrictSchemaError("simpler_benchmark.plan: identity differs")
        counts: dict[str, int] = {}
        for key in plan.episodes:
            if key.protocol != self.protocol:
                raise StrictSchemaError("simpler_benchmark.plan: protocol differs")
            counts[key.task_id] = counts.get(key.task_id, 0) + 1
        if counts != dict(self.task_episode_counts):
            raise StrictSchemaError("simpler_benchmark.plan: task episode counts differ")

    def validate_episode_manifest(self, manifest: EpisodeManifest) -> None:
        if self.model_route.startswith("openvla_simpler_") and manifest.steps != manifest.key.horizon:
            raise StrictSchemaError("OpenVLA SimplerEnv benchmark episode did not execute its full horizon")

    def evaluator(
        self,
        profile: Profile,
        plan: BenchmarkPlan,
        **kwargs: Any,
    ) -> CanonicalBenchmarkEvaluator:
        suite = profile.environment.suite
        return CanonicalBenchmarkEvaluator(
            {suite: profile},
            plan,
            task_suites={task: suite for task in self.task_episode_counts},
            metric_function=simpler_metrics,
            episode_manifest_validator=self.validate_episode_manifest,
            **kwargs,
        )


def verify_simpler_benchmark_output(
    path: str | Path,
    config: SimplerBenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    return verify_benchmark_output(
        path,
        plan_validator=config.validate_plan,
        metric_function=simpler_metrics,
        episode_manifest_validator=config.validate_episode_manifest,
        profile_suites=(config.load_profile(project_root).environment.suite,),
    )
