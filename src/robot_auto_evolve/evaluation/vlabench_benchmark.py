from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from robot_auto_evolve.benchmarks.vlabench_worker import (
    TASK_HORIZONS,
    VLABENCH_BENCHMARK_PROTOCOL,
    parse_vlabench_scenario,
)
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.benchmark import CanonicalBenchmarkEvaluator, verify_benchmark_output
from robot_auto_evolve.evaluation.private_metrics import validate_private_metrics
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import fields, integer, reject_json_constant, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest, canonical_json_bytes


MODEL_ROUTE = "xvla_vlabench"
METRIC = "equal_track_task_macro_progress_score"
EPISODES_PER_TASK = 10
TRACK_TASKS = {
    "track_1": (
        "add_condiment",
        "insert_flower",
        "select_book",
        "select_chemistry_tube",
        "select_drink",
        "select_fruit",
        "select_mahjong",
        "select_painting",
        "select_poker",
        "select_toy",
    ),
    "track_2": (
        "add_condiment",
        "insert_flower",
        "select_book",
        "select_chemistry_tube",
        "select_drink",
        "select_fruit",
        "select_mahjong",
        "select_painting",
        "select_poker",
        "select_toy",
    ),
    "track_3": (
        "add_condiment",
        "insert_flower",
        "select_book",
        "select_chemistry_tube",
        "select_drink",
        "select_fruit",
        "select_nth_largest_poker",
        "select_painting",
        "select_toy",
        "select_unique_type_mahjong",
    ),
    "track_4": (
        "add_condiment",
        "insert_flower",
        "select_book",
        "select_chemistry_tube",
        "select_drink",
        "select_fruit",
        "select_mahjong",
        "select_painting",
        "select_poker",
        "select_toy",
    ),
}
TRACK_NAMES = {
    "track_1": "track_1_in_distribution",
    "track_2": "track_2_cross_category",
    "track_3": "track_3_common_sense",
    "track_4": "track_4_semantic_instruction",
}
PRIVATE_METRIC_NAMES = {"success", "intention_score", "progress_score"}


def _relative_json(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
        raise StrictSchemaError(f"{name}: expected safe relative JSON path")
    return path.as_posix()


def _episode_metrics(path: Path, manifest: EpisodeManifest) -> dict[str, bool | float]:
    source = path / manifest.key.artifact_id() / "private_metrics.json"
    if not source.is_file() or source.is_symlink():
        raise StrictSchemaError("vlabench_benchmark.private_metrics: artifact is absent")
    payload = source.read_bytes()
    try:
        value = json.loads(payload, parse_constant=reject_json_constant)
    except Exception as exc:
        raise StrictSchemaError(f"vlabench_benchmark.private_metrics: invalid JSON: {exc}") from exc
    if canonical_json_bytes(value) != payload:
        raise StrictSchemaError("vlabench_benchmark.private_metrics: JSON is not canonical")
    obj = fields(
        value,
        {"schema_version", "kind", "metrics"},
        path="vlabench_benchmark.private_metrics",
    )
    if obj["schema_version"] != 1 or obj["kind"] != "private_evaluator_metrics":
        raise StrictSchemaError("vlabench_benchmark.private_metrics: identity differs")
    metrics = validate_private_metrics(obj["metrics"], "vlabench_benchmark.private_metrics.metrics")
    if set(metrics) != PRIVATE_METRIC_NAMES or type(metrics["success"]) is not bool:
        raise StrictSchemaError("vlabench_benchmark.private_metrics: metric set differs")
    if metrics["success"] is not manifest.success:
        raise StrictSchemaError("vlabench_benchmark.private_metrics: success differs")
    if any(not 0.0 <= float(metrics[name]) <= 1.0 for name in ("intention_score", "progress_score")):
        raise StrictSchemaError("vlabench_benchmark.private_metrics: score is outside [0, 1]")
    return metrics


def _mean_metrics(rows: list[dict[str, bool | float]]) -> dict[str, float]:
    if not rows:
        raise StrictSchemaError("vlabench_benchmark.metrics: empty group")
    return {
        "success_rate": sum(float(item["success"]) for item in rows) / len(rows),
        "intention_score": sum(float(item["intention_score"]) for item in rows) / len(rows),
        "progress_score": sum(float(item["progress_score"]) for item in rows) / len(rows),
    }


def vlabench_metrics(manifests: tuple[EpisodeManifest, ...], episode_root: Path) -> dict[str, Any]:
    rows: dict[tuple[str, str], list[dict[str, bool | float]]] = {
        (track, task): [] for track, tasks in TRACK_TASKS.items() for task in tasks
    }
    for manifest in manifests:
        track, _ = parse_vlabench_scenario(manifest.key.scenario_id)
        key = (track, manifest.key.task_id)
        if key not in rows:
            raise StrictSchemaError("vlabench_benchmark.metrics: unexpected track or task")
        rows[key].append(_episode_metrics(Path(episode_root), manifest))
    if any(len(value) != EPISODES_PER_TASK for value in rows.values()):
        raise StrictSchemaError("vlabench_benchmark.metrics: episode count differs")
    track_metrics = {}
    for track, tasks in TRACK_TASKS.items():
        task_metrics = {task: _mean_metrics(rows[(track, task)]) for task in tasks}
        averages = {
            name: sum(value[name] for value in task_metrics.values()) / len(task_metrics)
            for name in ("success_rate", "intention_score", "progress_score")
        }
        track_metrics[TRACK_NAMES[track]] = {"tasks": task_metrics, "averages": averages}
    averages = {
        name: sum(value["averages"][name] for value in track_metrics.values()) / len(track_metrics)
        for name in ("success_rate", "intention_score", "progress_score")
    }
    return {
        "metric": METRIC,
        "score": averages["progress_score"],
        "averages": averages,
        "track_metrics": track_metrics,
    }


@dataclass(frozen=True)
class VLABenchBenchmarkConfig:
    benchmark_id: str
    protocol: str
    episodes_per_task: int
    plan_path: str
    plan_sha256: str
    profile_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("vlabench_benchmark.schema_version: expected 1")
        object.__setattr__(self, "benchmark_id", string(self.benchmark_id, "vlabench_benchmark.benchmark_id"))
        if self.protocol != VLABENCH_BENCHMARK_PROTOCOL:
            raise StrictSchemaError("vlabench_benchmark.protocol differs")
        count = integer(self.episodes_per_task, "vlabench_benchmark.episodes_per_task", minimum=1)
        if count != EPISODES_PER_TASK:
            raise StrictSchemaError("vlabench_benchmark.episodes_per_task differs")
        object.__setattr__(self, "episodes_per_task", count)
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "vlabench_benchmark.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "vlabench_benchmark.plan.sha256"))
        object.__setattr__(self, "profile_path", _relative_json(self.profile_path, "vlabench_benchmark.profile"))

    @classmethod
    def from_mapping(cls, value: Any) -> "VLABenchBenchmarkConfig":
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
            path="vlabench_benchmark",
        )
        if obj["model_route"] != MODEL_ROUTE or obj["metric"] != METRIC:
            raise StrictSchemaError("vlabench_benchmark route or metric differs")
        plan = fields(obj["plan"], {"path", "sha256"}, path="vlabench_benchmark.plan")
        return cls(
            schema_version=integer(obj["schema_version"], "vlabench_benchmark.schema_version"),
            benchmark_id=obj["benchmark_id"],
            protocol=obj["protocol"],
            episodes_per_task=obj["episodes_per_task"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profile_path=obj["profile"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "VLABenchBenchmarkConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_json_constant))

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        plan = BenchmarkPlan.load(Path(project_root).resolve() / self.plan_path)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("vlabench_benchmark.plan: hash mismatch")
        self.validate_plan(plan)
        return plan

    def load_profile(self, project_root: str | Path) -> Profile:
        root = Path(project_root).resolve()
        profile = Profile.load(root / self.profile_path, project_root=root)
        if profile.environment.suite != "vlabench_xvla_tracks_1_4":
            raise StrictSchemaError("vlabench_benchmark.profile suite differs")
        if {item.identity.service_name for item in profile.policy.replicas} != {MODEL_ROUTE}:
            raise StrictSchemaError("vlabench_benchmark.profile policy differs")
        return profile

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if plan.plan_id != self.benchmark_id or plan.model_route != MODEL_ROUTE:
            raise StrictSchemaError("vlabench_benchmark.plan identity differs")
        expected = {}
        policy_seed = 731000
        for track, tasks in TRACK_TASKS.items():
            for task in sorted(tasks):
                for index in range(EPISODES_PER_TASK):
                    expected[(track, task, index)] = policy_seed
                    policy_seed += 1
        found = set()
        for episode in plan.episodes:
            track, index = parse_vlabench_scenario(episode.scenario_id)
            coordinate = (track, episode.task_id, index)
            if (
                episode.split != "benchmark"
                or episode.protocol != self.protocol
                or coordinate not in expected
                or episode.environment_seed != index
                or episode.policy_seed != expected[coordinate]
                or episode.replicate_id != f"b-{track}-{episode.task_id}-{index:03d}"
                or episode.horizon != TASK_HORIZONS[episode.task_id]
            ):
                raise StrictSchemaError("vlabench_benchmark.plan episode differs")
            found.add(coordinate)
        if found != set(expected) or len(plan.episodes) != len(expected):
            raise StrictSchemaError("vlabench_benchmark.plan grid differs")

    def evaluator(self, profile: Profile, plan: BenchmarkPlan, **kwargs: Any) -> CanonicalBenchmarkEvaluator:
        suite = profile.environment.suite
        return CanonicalBenchmarkEvaluator(
            {suite: profile},
            plan,
            task_suites={task: suite for tasks in TRACK_TASKS.values() for task in tasks},
            artifact_metric_function=vlabench_metrics,
            **kwargs,
        )


def verify_vlabench_benchmark_output(
    path: str | Path,
    config: VLABenchBenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    return verify_benchmark_output(
        path,
        plan_validator=config.validate_plan,
        artifact_metric_function=vlabench_metrics,
        profile_suites=(config.load_profile(project_root).environment.suite,),
    )
