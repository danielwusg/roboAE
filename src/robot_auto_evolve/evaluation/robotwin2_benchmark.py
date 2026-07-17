from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from robot_auto_evolve.benchmarks.robotwin2_protocol import (
    ROBOTWIN2_BENCHMARK_PROTOCOL,
    ROBOTWIN2_BENCHMARK_SMOKE_PROTOCOL,
    ROBOTWIN2_MODEL_ROUTE,
    ROBOTWIN2_OFFICIAL_SEED_BASE,
    ROBOTWIN2_OFFICIAL_TRIALS_PER_TASK,
    ROBOTWIN2_SCENARIO,
    ROBOTWIN2_SEED_PROTOCOL,
    ROBOTWIN2_SOURCE_COMMIT,
    ROBOTWIN2_STEP_LIMITS,
    expected_horizon,
)
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.benchmark import CanonicalBenchmarkEvaluator, verify_benchmark_output
from robot_auto_evolve.evaluation.metrics import compute_task_macro_metrics
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import (
    fields,
    integer,
    json_object,
    reject_json_constant,
    sha256,
    string,
)
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest, canonical_json_bytes


METRIC = "equal_task_macro_success"
OFFICIAL_ID = "xvla_robotwin2_demo_clean_official_50x100_v1"
SMOKE_ID = "xvla_robotwin2_demo_clean_official_smoke_v1"


def _relative_json(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
        raise StrictSchemaError(f"{name}: expected safe relative JSON path")
    return path.as_posix()


def robotwin2_metrics(manifests: tuple[EpisodeManifest, ...]) -> dict[str, Any]:
    task_metrics = {
        task: compute_task_macro_metrics(tuple(item for item in manifests if item.key.task_id == task)).to_mapping()
        for task in sorted(ROBOTWIN2_STEP_LIMITS)
    }
    overall = compute_task_macro_metrics(manifests).to_mapping()
    return {
        "metric": METRIC,
        "score": overall["macro_success"],
        "task_metrics": task_metrics,
        "all_task_macro": overall,
    }


@dataclass(frozen=True)
class RoboTwin2BenchmarkConfig:
    benchmark_id: str
    protocol: str
    trials_per_task: int
    plan_path: str
    plan_sha256: str
    preparation_path: str
    preparation_sha256: str
    profile_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robotwin2_benchmark.schema_version: expected 1")
        identity = string(self.benchmark_id, "robotwin2_benchmark.benchmark_id")
        expected = {
            OFFICIAL_ID: (ROBOTWIN2_BENCHMARK_PROTOCOL, ROBOTWIN2_OFFICIAL_TRIALS_PER_TASK),
            SMOKE_ID: (ROBOTWIN2_BENCHMARK_SMOKE_PROTOCOL, 1),
        }.get(identity)
        if expected is None or (self.protocol, self.trials_per_task) != expected:
            raise StrictSchemaError("robotwin2_benchmark identity, protocol, or trial count differs")
        object.__setattr__(self, "benchmark_id", identity)
        object.__setattr__(
            self,
            "trials_per_task",
            integer(self.trials_per_task, "robotwin2_benchmark.trials_per_task", minimum=1),
        )
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "robotwin2_benchmark.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "robotwin2_benchmark.plan.sha256"))
        object.__setattr__(
            self,
            "preparation_path",
            _relative_json(self.preparation_path, "robotwin2_benchmark.preparation.path"),
        )
        object.__setattr__(
            self,
            "preparation_sha256",
            sha256(self.preparation_sha256, "robotwin2_benchmark.preparation.sha256"),
        )
        if self.plan_sha256 == "0" * 64 or self.preparation_sha256 == "0" * 64:
            raise StrictSchemaError("robotwin2_benchmark contains a placeholder hash")
        object.__setattr__(self, "profile_path", _relative_json(self.profile_path, "robotwin2_benchmark.profile"))

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboTwin2BenchmarkConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "model_route",
                "metric",
                "protocol",
                "trials_per_task",
                "plan",
                "preparation",
                "profile",
            },
            path="robotwin2_benchmark",
        )
        if obj["model_route"] != ROBOTWIN2_MODEL_ROUTE or obj["metric"] != METRIC:
            raise StrictSchemaError("robotwin2_benchmark route or metric differs")
        if obj["plan"] is None or obj["preparation"] is None:
            raise StrictSchemaError(
                "robotwin2_benchmark requires an expert-success-filtered 5,000-row plan prepared and pinned in the development workspace before porting"
            )
        plan = fields(obj["plan"], {"path", "sha256"}, path="robotwin2_benchmark.plan")
        preparation = fields(
            obj["preparation"],
            {"path", "sha256"},
            path="robotwin2_benchmark.preparation",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "robotwin2_benchmark.schema_version"),
            benchmark_id=obj["benchmark_id"],
            protocol=obj["protocol"],
            trials_per_task=obj["trials_per_task"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            preparation_path=preparation["path"],
            preparation_sha256=preparation["sha256"],
            profile_path=obj["profile"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "RoboTwin2BenchmarkConfig":
        return cls.from_mapping(
            json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_json_constant)
        )

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        plan = BenchmarkPlan.load(Path(project_root).resolve() / self.plan_path)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("robotwin2_benchmark.plan hash differs")
        self.validate_plan(plan)
        return plan

    def load_profile(self, project_root: str | Path) -> Profile:
        root = Path(project_root).resolve()
        profile = Profile.load(root / self.profile_path, project_root=root)
        if profile.environment.suite != "robotwin2_demo_clean":
            raise StrictSchemaError("robotwin2_benchmark.profile suite differs")
        if {item.identity.service_name for item in profile.policy.replicas} != {ROBOTWIN2_MODEL_ROUTE}:
            raise StrictSchemaError("robotwin2_benchmark.profile policy differs")
        return profile

    def validate_preparation(self, project_root: str | Path, plan: BenchmarkPlan) -> None:
        self.validate_plan(plan)
        root = Path(project_root).resolve()
        path = root / self.preparation_path
        if not path.is_file() or path.is_symlink():
            raise StrictSchemaError("robotwin2_benchmark.preparation is absent")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.preparation_sha256:
            raise StrictSchemaError("robotwin2_benchmark.preparation hash differs")
        try:
            value = json.loads(payload, parse_constant=reject_json_constant)
        except Exception as exc:
            raise StrictSchemaError(f"robotwin2_benchmark.preparation is invalid: {exc}") from exc
        if canonical_json_bytes(value) != payload or set(value) != {
            "schema_version",
            "source_commit",
            "seed_protocol",
            "evaluation_protocol",
            "trials_per_task",
            "task_count",
            "episode_count",
            "plan_sha256",
            "seed_records",
        }:
            raise StrictSchemaError("robotwin2_benchmark.preparation fields differ")
        if (
            value["schema_version"] != 1
            or value["source_commit"] != ROBOTWIN2_SOURCE_COMMIT
            or value["seed_protocol"] != ROBOTWIN2_SEED_PROTOCOL
            or value["evaluation_protocol"] != self.protocol
            or value["trials_per_task"] != self.trials_per_task
            or value["task_count"] != len(ROBOTWIN2_STEP_LIMITS)
            or value["episode_count"] != len(plan.episodes)
            or value["plan_sha256"] != plan.resolved_hash()
            or type(value["seed_records"]) is not list
            or not value["seed_records"]
        ):
            raise StrictSchemaError("robotwin2_benchmark.preparation provenance differs")
        accepted_by_task = {}
        paths = set()
        for item_index, item in enumerate(value["seed_records"]):
            if type(item) is not dict or set(item) != {
                "record_path",
                "record_sha256",
                "journal_path",
                "journal_sha256",
            }:
                raise StrictSchemaError("robotwin2_benchmark.seed record provenance differs")
            record_value = string(
                item["record_path"],
                f"robotwin2_benchmark.seed_records.{item_index}.record_path",
            )
            journal_value = string(
                item["journal_path"],
                f"robotwin2_benchmark.seed_records.{item_index}.journal_path",
            )
            record_hash = sha256(
                item["record_sha256"],
                f"robotwin2_benchmark.seed_records.{item_index}.record_sha256",
            )
            journal_hash = sha256(
                item["journal_sha256"],
                f"robotwin2_benchmark.seed_records.{item_index}.journal_sha256",
            )
            record_input = Path(record_value)
            journal_input = Path(journal_value)
            record_path = record_input.resolve()
            journal_path = journal_input.resolve()
            try:
                record_path.relative_to(root)
                journal_path.relative_to(root)
            except ValueError as exc:
                raise StrictSchemaError("robotwin2_benchmark.seed record escapes project root") from exc
            if (
                not record_input.is_absolute()
                or not journal_input.is_absolute()
                or str(record_path) != record_value
                or str(journal_path) != journal_value
                or journal_path != record_path.with_name(record_path.name + ".attempts.jsonl")
            ):
                raise StrictSchemaError("robotwin2_benchmark.seed record path differs")
            if record_path in paths or journal_path in paths:
                raise StrictSchemaError("robotwin2_benchmark.seed record path is duplicated")
            paths.update((record_path, journal_path))
            if (
                not record_path.is_file()
                or record_path.is_symlink()
                or not journal_path.is_file()
                or journal_path.is_symlink()
                or hashlib.sha256(record_path.read_bytes()).hexdigest() != record_hash
                or hashlib.sha256(journal_path.read_bytes()).hexdigest() != journal_hash
            ):
                raise StrictSchemaError("robotwin2_benchmark.seed record hash differs")
            record_payload = record_path.read_bytes()
            record = json.loads(
                record_payload,
                object_pairs_hook=json_object,
                parse_constant=reject_json_constant,
            )
            if (
                type(record) is not dict
                or set(record) != {
                    "schema_version",
                    "source_commit",
                    "protocol",
                    "scenario",
                    "split",
                    "seed_base",
                    "requested_per_task",
                    "render_gpu_id",
                    "tasks",
                }
                or record.get("schema_version") != 2
                or record.get("source_commit") != ROBOTWIN2_SOURCE_COMMIT
                or record.get("protocol") != ROBOTWIN2_SEED_PROTOCOL
                or record.get("scenario") != ROBOTWIN2_SCENARIO
                or record.get("split") != "benchmark"
                or record.get("seed_base") != ROBOTWIN2_OFFICIAL_SEED_BASE
                or type(record.get("requested_per_task")) is not int
                or record["requested_per_task"] != self.trials_per_task
                or type(record.get("render_gpu_id")) is not int
                or record["render_gpu_id"] < 0
                or type(record.get("tasks")) is not dict
                or record_payload
                != (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            ):
                raise StrictSchemaError("robotwin2_benchmark.seed record differs")
            grouped = {task: [] for task in record["tasks"]}
            active = None
            closed = set()
            journal_payload = journal_path.read_bytes()
            if journal_payload and not journal_payload.endswith(b"\n"):
                raise StrictSchemaError("robotwin2_benchmark.seed journal is incomplete")
            for row_index, line in enumerate(journal_payload.splitlines(keepends=True)):
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=json_object,
                        parse_constant=reject_json_constant,
                    )
                except Exception as exc:
                    raise StrictSchemaError(
                        f"robotwin2_benchmark.seed journal row {row_index} is invalid"
                    ) from exc
                canonical = (
                    json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode("utf-8")
                if (
                    type(row) is not dict
                    or canonical != line
                    or set(row)
                    != {
                        "schema_version",
                        "source_commit",
                        "protocol",
                        "scenario",
                        "split",
                        "task_id",
                        "seed",
                        "accepted",
                        "error",
                        "fatal",
                    }
                    or row["schema_version"] != 2
                    or row["source_commit"] != ROBOTWIN2_SOURCE_COMMIT
                    or row["protocol"] != ROBOTWIN2_SEED_PROTOCOL
                    or row["scenario"] != ROBOTWIN2_SCENARIO
                    or row["split"] != "benchmark"
                    or row["task_id"] not in grouped
                    or row["fatal"] is not False
                ):
                    raise StrictSchemaError("robotwin2_benchmark.seed journal differs")
                task_id = row["task_id"]
                if active != task_id:
                    if task_id in closed:
                        raise StrictSchemaError("robotwin2_benchmark.seed journal tasks are interleaved")
                    if active is not None:
                        closed.add(active)
                    active = task_id
                grouped[task_id].append({key: row[key] for key in ("seed", "accepted", "error")})
            for task_id, task_record in record["tasks"].items():
                if task_id in accepted_by_task or task_id not in ROBOTWIN2_STEP_LIMITS:
                    raise StrictSchemaError("robotwin2_benchmark.seed task differs")
                seeds = task_record.get("accepted_seeds") if isinstance(task_record, dict) else None
                attempts = task_record.get("attempts") if isinstance(task_record, dict) else None
                if (
                    type(task_record) is not dict
                    or set(task_record) != {"accepted_seeds", "attempts"}
                    or type(seeds) is not list
                    or len(seeds) != record["requested_per_task"]
                    or any(type(seed) is not int or seed < ROBOTWIN2_OFFICIAL_SEED_BASE for seed in seeds)
                    or len(seeds) != len(set(seeds))
                    or type(attempts) is not list
                    or any(
                        type(attempt) is not dict
                        or set(attempt) != {"seed", "accepted", "error"}
                        or type(attempt["seed"]) is not int
                        or type(attempt["accepted"]) is not bool
                        or (attempt["error"] is not None and type(attempt["error"]) is not str)
                        or (attempt["accepted"] and attempt["error"] is not None)
                        for attempt in attempts
                    )
                    or [attempt["seed"] for attempt in attempts if attempt["accepted"]] != seeds
                    or [attempt["seed"] for attempt in attempts]
                    != list(range(ROBOTWIN2_OFFICIAL_SEED_BASE, ROBOTWIN2_OFFICIAL_SEED_BASE + len(attempts)))
                    or grouped[task_id] != attempts
                ):
                    raise StrictSchemaError("robotwin2_benchmark.seed attempts differ")
                accepted_by_task[task_id] = tuple(seeds)
        if set(accepted_by_task) != set(ROBOTWIN2_STEP_LIMITS):
            raise StrictSchemaError("robotwin2_benchmark.seed tasks do not cover the benchmark")
        plan_seeds = {task: [] for task in ROBOTWIN2_STEP_LIMITS}
        for episode in plan.episodes:
            plan_seeds[episode.task_id].append(episode.environment_seed)
        if any(tuple(plan_seeds[task]) != accepted_by_task[task] for task in plan_seeds):
            raise StrictSchemaError("robotwin2_benchmark.plan seeds differ from preparation")

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if plan.plan_id != self.benchmark_id or plan.model_route != ROBOTWIN2_MODEL_ROUTE:
            raise StrictSchemaError("robotwin2_benchmark.plan identity differs")
        tasks = sorted(ROBOTWIN2_STEP_LIMITS)
        if len(plan.episodes) != len(tasks) * self.trials_per_task:
            raise StrictSchemaError("robotwin2_benchmark.plan task counts differ")
        seeds = {task: set() for task in ROBOTWIN2_STEP_LIMITS}
        for episode_index, episode in enumerate(plan.episodes):
            task_index, trial = divmod(episode_index, self.trials_per_task)
            task_id = tasks[task_index]
            if (
                episode.task_id != task_id
                or episode.split != "benchmark"
                or episode.scenario_id != ROBOTWIN2_SCENARIO
                or episode.protocol != self.protocol
                or episode.environment_seed < ROBOTWIN2_OFFICIAL_SEED_BASE
                or episode.policy_seed != 500000 + task_index * self.trials_per_task + trial
                or episode.replicate_id != f"b-{task_id}-{trial:03d}"
                or episode.horizon != expected_horizon(self.protocol, task_id)
            ):
                raise StrictSchemaError("robotwin2_benchmark.plan episode differs")
            seeds[task_id].add(episode.environment_seed)
        if any(len(value) != self.trials_per_task for value in seeds.values()):
            raise StrictSchemaError("robotwin2_benchmark.plan environment seeds differ")

    def evaluator(self, profile: Profile, plan: BenchmarkPlan, **kwargs: Any) -> CanonicalBenchmarkEvaluator:
        suite = profile.environment.suite
        return CanonicalBenchmarkEvaluator(
            {suite: profile},
            plan,
            task_suites={task: suite for task in ROBOTWIN2_STEP_LIMITS},
            metric_function=robotwin2_metrics,
            **kwargs,
        )


def verify_robotwin2_benchmark_output(
    path: str | Path,
    config: RoboTwin2BenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    return verify_benchmark_output(
        path,
        plan_validator=config.validate_plan,
        metric_function=robotwin2_metrics,
        profile_suites=(config.load_profile(project_root).environment.suite,),
    )
