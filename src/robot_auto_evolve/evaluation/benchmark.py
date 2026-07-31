from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from robot_auto_evolve.agent import ToolEndpoint
from robot_auto_evolve.benchmarks.libero_suites import LIBERO_SUITE_TASKS, LIBERO_TASK_SUITE
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.metrics import compute_task_macro_metrics
from robot_auto_evolve.evolution.hashing import mapping_sha256, tree_hashes
from robot_auto_evolve.evolution.profile_evaluator import resolve_render_gpu_ids
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import (
    boolean,
    fields,
    integer,
    json_object,
    reject_json_constant,
    sequence,
    sha256,
    string,
)
from robot_auto_evolve.provenance import (
    ArtifactDescriptor,
    BenchmarkPlan,
    EpisodeKey,
    EpisodeManifest,
    canonical_json_bytes,
)
from robot_auto_evolve.services import MsgpackServiceClient, ReplicaScheduler, ServiceReplica
from robot_auto_evolve.services.identity import ServiceIdentity

MAX_ERRORED_EPISODE_FRACTION = 0.5


SUITES = tuple(LIBERO_SUITE_TASKS)
TOOL_CALL_TIMEOUT_S = 300.0


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=json_object, parse_constant=reject_json_constant)
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"benchmark: failed to load {path}: {exc}") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _preserve_scaffold(output: Path, scaffold: Path, code_hash: str) -> None:
    source = scaffold / "scaffold.py"
    payload = source.read_bytes()
    actual_hash = mapping_sha256({"scaffold.py": hashlib.sha256(payload).hexdigest()})
    if actual_hash != code_hash:
        raise RuntimeError("benchmark scaffold changed while preparing the run")
    target = output / "scaffold.py"
    if not target.exists():
        _atomic_write(target, payload)
        target.chmod(0o444)
    if (
        not target.is_file()
        or target.is_symlink()
        or stat.S_IMODE(target.stat().st_mode) != 0o444
        or mapping_sha256({"scaffold.py": hashlib.sha256(target.read_bytes()).hexdigest()}) != code_hash
    ):
        raise StrictSchemaError("benchmark preserved scaffold differs")


def _relative_json(value: Any, path: str) -> str:
    result = PurePosixPath(string(value, path))
    if result.is_absolute() or ".." in result.parts or result.suffix.lower() != ".json":
        raise StrictSchemaError(f"{path}: expected safe relative JSON path")
    return result.as_posix()


def _validate_standard_plan(plan: BenchmarkPlan, trials_per_task: int | None = None) -> int:
    by_task: dict[str, list[EpisodeKey]] = {}
    for key in plan.episodes:
        if key.task_id not in LIBERO_TASK_SUITE:
            raise StrictSchemaError("benchmark plan contains a nonstandard LIBERO task")
        by_task.setdefault(key.task_id, []).append(key)
    if set(by_task) != set(LIBERO_TASK_SUITE):
        raise StrictSchemaError("benchmark plan must contain all 40 standard LIBERO tasks")
    counts = {len(rows) for rows in by_task.values()}
    if len(counts) != 1:
        raise StrictSchemaError("benchmark plan trials per task differ")
    trials = next(iter(counts))
    if trials_per_task is not None and trials != trials_per_task:
        raise StrictSchemaError("benchmark plan trials per task differ")
    return trials


def _benchmark_metrics(manifests: tuple[EpisodeManifest, ...]) -> dict[str, Any]:
    suite_metrics = {}
    for suite, tasks in LIBERO_SUITE_TASKS.items():
        rows = [item for item in manifests if item.key.task_id in tasks]
        suite_metrics[suite] = compute_task_macro_metrics(rows).to_mapping()
    overall = compute_task_macro_metrics(manifests).to_mapping()
    suite_mean = sum(item["macro_success"] for item in suite_metrics.values()) / len(suite_metrics)
    return {
        "metric": "equal_suite_task_macro_success",
        "score": suite_mean,
        "suite_metrics": suite_metrics,
        "all_task_macro": overall,
    }


def _verify_episode_directory(path: Path, key: EpisodeKey) -> EpisodeManifest:
    if not path.is_dir() or path.is_symlink():
        raise StrictSchemaError("benchmark episode directory differs")
    manifest_path = path / "episode.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise StrictSchemaError("benchmark episode manifest is absent")
    manifest = EpisodeManifest.from_mapping(_load_json(manifest_path))
    if manifest.key != key or manifest.state not in {"complete", "error"}:
        raise StrictSchemaError("benchmark episode key or state differs")
    expected = {"episode.json", *(item.name for item in manifest.artifacts)}
    if {item.name for item in path.iterdir()} != expected:
        raise StrictSchemaError("benchmark episode files differ")
    for descriptor in manifest.artifacts:
        source = path / descriptor.name
        if not source.is_file() or source.is_symlink() or source.stat().st_size != descriptor.size_bytes:
            raise StrictSchemaError("benchmark episode artifact size differs")
        if hashlib.sha256(source.read_bytes()).hexdigest() != descriptor.sha256:
            raise StrictSchemaError("benchmark episode artifact hash differs")
    return manifest


@dataclass(frozen=True)
class LiberoBenchmarkConfig:
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
        object.__setattr__(self, "model_route", string(self.model_route, "benchmark_config.model_route"))
        if self.metric != "equal_suite_task_macro_success":
            raise StrictSchemaError("benchmark_config.metric: expected equal_suite_task_macro_success")
        object.__setattr__(
            self,
            "trials_per_task",
            integer(self.trials_per_task, "benchmark_config.trials_per_task", minimum=1),
        )
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "benchmark_config.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "benchmark_config.plan.sha256"))
        profiles = dict(self.profiles)
        if set(profiles) != set(SUITES):
            raise StrictSchemaError("benchmark_config.profiles: expected the four standard LIBERO suites")
        object.__setattr__(
            self,
            "profiles",
            {suite: _relative_json(profiles[suite], f"benchmark_config.profiles.{suite}") for suite in SUITES},
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "LiberoBenchmarkConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "model_route",
                "metric",
                "trials_per_task",
                "plan",
                "profiles",
            },
            path="benchmark_config",
        )
        plan = fields(obj["plan"], {"path", "sha256"}, path="benchmark_config.plan")
        profiles = fields(obj["profiles"], set(SUITES), path="benchmark_config.profiles")
        return cls(
            schema_version=integer(obj["schema_version"], "benchmark_config.schema_version"),
            benchmark_id=obj["benchmark_id"],
            model_route=obj["model_route"],
            metric=obj["metric"],
            trials_per_task=obj["trials_per_task"],
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profiles=profiles,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LiberoBenchmarkConfig":
        return cls.from_mapping(_load_json(Path(path)))

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
        return {
            suite: Profile.load(root / path, project_root=root)
            for suite, path in self.profiles.items()
        }

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if not isinstance(plan, BenchmarkPlan) or plan.model_route != self.model_route:
            raise StrictSchemaError("benchmark_config: plan route differs")
        _validate_standard_plan(plan, self.trials_per_task)


class CanonicalBenchmarkEvaluator:
    def __init__(
        self,
        profiles: Mapping[str, Profile],
        plan: BenchmarkPlan,
        *,
        agent_python: Path,
        simulator_python: Path,
        simulator_source: Path,
        live_clients: Mapping[tuple[str, str], MsgpackServiceClient],
        task_suites: Mapping[str, str] | None = None,
        metric_function: Callable[[tuple[EpisodeManifest, ...]], dict[str, Any]] | None = None,
        artifact_metric_function: Callable[[tuple[EpisodeManifest, ...], Path], dict[str, Any]] | None = None,
        episode_manifest_validator: Callable[[EpisodeManifest], Any] | None = None,
        render_gpu_ids: tuple[int, ...] | list[int] | None = None,
        reuse_agent: bool = False,
        reuse_sim: bool = False,
    ) -> None:
        self.reuse_agent = bool(reuse_agent)
        self.reuse_sim = bool(reuse_sim)
        self.profiles = dict(profiles)
        if not self.profiles:
            raise StrictSchemaError("benchmark evaluator requires at least one suite profile")
        if not isinstance(plan, BenchmarkPlan):
            raise StrictSchemaError("benchmark evaluator requires BenchmarkPlan")
        self.suites = tuple(self.profiles)
        primary = self.profiles["libero_spatial"] if "libero_spatial" in self.profiles else next(iter(self.profiles.values()))
        for suite, profile in self.profiles.items():
            if not isinstance(profile, Profile) or profile.environment.suite != suite:
                raise StrictSchemaError("benchmark evaluator profile suite differs")
            profile.validate()
            if profile.policy.to_mapping() != primary.policy.to_mapping():
                raise StrictSchemaError("benchmark evaluator policy profiles differ")
            if [tool.to_mapping() for tool in profile.tools] != [tool.to_mapping() for tool in primary.tools]:
                raise StrictSchemaError("benchmark evaluator tool profiles differ")
            if profile.resources != primary.resources:
                raise StrictSchemaError("benchmark evaluator resource profiles differ")
        if primary.policy.deployment_mode != "replicated":
            raise StrictSchemaError("benchmark evaluator requires replicated policy services")
        self.task_suites = dict(LIBERO_TASK_SUITE if task_suites is None else task_suites)
        if {item.task_id for item in plan.episodes} - set(self.task_suites):
            raise StrictSchemaError("benchmark evaluator task-to-suite map is incomplete")
        if set(self.task_suites.values()) - set(self.profiles):
            raise StrictSchemaError("benchmark evaluator task-to-suite map references an absent profile")
        if metric_function is not None and artifact_metric_function is not None:
            raise StrictSchemaError("benchmark evaluator accepts one metric function")
        self.metric_function = _benchmark_metrics if metric_function is None and artifact_metric_function is None else metric_function
        self.artifact_metric_function = artifact_metric_function
        self.episode_manifest_validator = episode_manifest_validator
        clients = dict(live_clients)
        replicas = []
        identities = []
        for item in primary.policy.replicas:
            key = (item.identity.service_name, item.identity.replica_id)
            if key not in clients:
                raise StrictSchemaError("benchmark evaluator policy client is absent")
            actual = clients[key].validate_identity()
            identities.append(actual)
            replicas.append(ServiceReplica(item.endpoint, actual, clients[key]))
        tool_endpoints = {}
        for tool in primary.tools:
            if not tool.enabled or tool.service is None:
                continue
            item = tool.service
            key = (item.identity.service_name, item.identity.replica_id)
            if key not in clients:
                raise StrictSchemaError("benchmark evaluator tool client is absent")
            identities.append(clients[key].validate_identity())
            tool_endpoints[tool.capability] = ToolEndpoint(
                item.endpoint,
                item.identity,
                tool.required,
                timeout_s=TOOL_CALL_TIMEOUT_S,
            )
        primary.validate_service_identities(identities)
        replica_count = len(replicas)
        sessions_per_replica = (primary.resources.workers + replica_count - 1) // replica_count
        self.plan = plan
        self.primary = primary
        self.agent_python = Path(agent_python).resolve()
        self.simulator_python = Path(simulator_python).resolve()
        self.simulator_source = Path(simulator_source).resolve()
        self.scheduler = ReplicaScheduler(replicas, max_sessions_per_replica=sessions_per_replica)
        self.tool_endpoints = tool_endpoints
        self.identities = tuple(sorted(identities, key=lambda item: (item.service_name, item.replica_id)))
        self.sessions_per_replica = sessions_per_replica
        self.render_gpu_ids = resolve_render_gpu_ids(primary, render_gpu_ids)
        self.render_gpu_by_replica = {
            replica.identity.replica_id: gpu_id
            for replica, gpu_id in zip(self.scheduler.replicas, self.render_gpu_ids, strict=True)
        }

    def _header(self, code_hash: str, created_ns: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark_plan": self.plan.to_mapping(),
            "benchmark_plan_sha256": self.plan.resolved_hash(),
            "profile_sha256": {suite: self.profiles[suite].resolved_hash() for suite in self.suites},
            "code_sha256": code_hash,
            "service_identities": [item.to_mapping() for item in self.identities],
            "created_ns": created_ns,
        }

    def _prepare(self, output: Path, code_hash: str) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        (output / "episodes").mkdir(exist_ok=True)
        path = output / "run.json"
        if not path.exists():
            value = self._header(code_hash, time.time_ns())
            _atomic_write(path, canonical_json_bytes(value))
            return value
        value = fields(
            _load_json(path),
            {
                "schema_version",
                "benchmark_plan",
                "benchmark_plan_sha256",
                "profile_sha256",
                "code_sha256",
                "service_identities",
                "created_ns",
            },
            path="benchmark_run",
        )
        expected = self._header(
            code_hash,
            integer(value["created_ns"], "benchmark_run.created_ns", minimum=0),
        )
        if value != expected:
            raise StrictSchemaError("benchmark run invariant differs")
        return dict(value)

    @staticmethod
    def _verify_episode(path: Path, key: EpisodeKey) -> EpisodeManifest:
        return _verify_episode_directory(path, key)

    def _load_manifests(self, output: Path) -> tuple[EpisodeManifest, ...]:
        key_by_id = {key.artifact_id(): key for key in self.plan.episodes}
        directories = {path.name: path for path in (output / "episodes").iterdir()}
        unknown = set(directories) - set(key_by_id)
        if unknown:
            raise StrictSchemaError(f"benchmark output contains unknown episode {sorted(unknown)[0]}")
        manifests = tuple(
            self._verify_episode(directories[artifact_id], key)
            for artifact_id, key in key_by_id.items()
            if artifact_id in directories
        )
        if self.episode_manifest_validator is not None:
            for manifest in manifests:
                self.episode_manifest_validator(manifest)
        return manifests

    def _record_episode(
        self,
        output: Path,
        staging_root: Path,
        key: EpisodeKey,
        execution: Any,
        started_ns: int,
    ) -> EpisodeManifest:
        if execution.state != "complete" or execution.success is None or execution.error is not None:
            raise RuntimeError("benchmark episode runner did not complete")
        descriptors = tuple(
            ArtifactDescriptor(name, hashlib.sha256(payload).hexdigest(), len(payload))
            for name, payload in sorted(execution.artifacts.items())
        )
        manifest = EpisodeManifest(
            key=key,
            state="complete",
            success=execution.success,
            steps=execution.steps,
            started_ns=started_ns,
            finished_ns=time.time_ns(),
            artifacts=descriptors,
            error=None,
        )
        if self.episode_manifest_validator is not None:
            self.episode_manifest_validator(manifest)
        target = output / "episodes" / key.artifact_id()
        staging = staging_root / key.artifact_id()
        staging.mkdir(parents=True, exist_ok=False)
        for name, payload in sorted(execution.artifacts.items()):
            (staging / name).write_bytes(payload)
        (staging / "episode.json").write_bytes(canonical_json_bytes(manifest.to_mapping()))
        os.rename(staging, target)
        return manifest

    def _record_error_episode(
        self,
        output: Path,
        staging_root: Path,
        key: EpisodeKey,
        exc: Exception,
        started_ns: int,
    ) -> EpisodeManifest:
        manifest = EpisodeManifest(
            key=key,
            state="error",
            success=None,
            steps=0,
            started_ns=started_ns,
            finished_ns=time.time_ns(),
            artifacts=(),
            error=f"{type(exc).__name__}: {exc}",
        )
        target = output / "episodes" / key.artifact_id()
        staging = staging_root / key.artifact_id()
        staging.mkdir(parents=True, exist_ok=False)
        (staging / "episode.json").write_bytes(canonical_json_bytes(manifest.to_mapping()))
        os.rename(staging, target)
        return manifest

    @staticmethod
    def _write_error(path: Path, key: EpisodeKey, exc: Exception) -> None:
        _atomic_write(
            path / f"{key.artifact_id()}.json",
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "key": key.to_mapping(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finished_ns": time.time_ns(),
                }
            ),
        )

    def _report(self, manifests: tuple[EpisodeManifest, ...], errors: int, output: Path) -> dict[str, Any]:
        errored = sum(1 for item in manifests if item.state == "error")
        complete = len(manifests) == len(self.plan.episodes) and errors == 0
        metrics = None
        if complete:
            metrics = (
                self.artifact_metric_function(manifests, output / "episodes")
                if self.artifact_metric_function is not None
                else self.metric_function(manifests)
            )
        return {
            "schema_version": 1,
            "benchmark_plan_sha256": self.plan.resolved_hash(),
            "n_expected": len(self.plan.episodes),
            "n_complete": len(manifests),
            "n_pending": len(self.plan.episodes) - len(manifests),
            "errors_this_invocation": errors,
            "n_errored": errored,
            "complete": complete,
            "metrics": metrics,
            "updated_ns": time.time_ns(),
        }

    def evaluate(self, scaffold_dir: Path, output_dir: Path, invocation_dir: Path) -> dict[str, Any]:
        from robot_auto_evolve.evolution.profile_evaluator import (
            AgentGatewayPool,
            ProfileEpisodeRunner,
            SimulatorProcessPool,
        )

        scaffold = Path(scaffold_dir).resolve()
        output = Path(output_dir).resolve()
        invocation = Path(invocation_dir).resolve()
        invocation.mkdir(parents=True, exist_ok=False)
        code_hash = mapping_sha256(tree_hashes(scaffold))
        header = self._prepare(output, code_hash)
        _preserve_scaffold(output, scaffold, code_hash)
        if (output / "final.json").is_file():
            def validate_resumed_plan(plan: BenchmarkPlan) -> None:
                if plan.to_mapping() != self.plan.to_mapping():
                    raise StrictSchemaError("benchmark resumed plan differs")

            verify_benchmark_output(
                output,
                plan_validator=validate_resumed_plan,
                metric_function=self.metric_function,
                artifact_metric_function=self.artifact_metric_function,
                episode_manifest_validator=self.episode_manifest_validator,
                profile_suites=self.suites,
            )
            return dict(_load_json(output / "report.json"))
        existing = self._load_manifests(output)
        retry = [item.key for item in existing if item.state == "error"]
        for key in retry:
            shutil.rmtree(output / "episodes" / key.artifact_id(), ignore_errors=True)
        existing_keys = {item.key for item in existing if item.state == "complete"}
        pending = [key for key in self.plan.episodes if key not in existing_keys]
        assignments = {
            key.artifact_id(): self.scheduler.replicas[index % len(self.scheduler.replicas)].identity.replica_id
            for index, key in enumerate(self.plan.episodes)
        }
        agent_pool = AgentGatewayPool(invocation / "agent_pool") if self.reuse_agent else None
        sim_pool = SimulatorProcessPool(invocation / "simulator_pool") if self.reuse_sim else None
        runners = {
            suite: ProfileEpisodeRunner(
                profile,
                scaffold,
                self.scheduler,
                self.tool_endpoints,
                agent_python=self.agent_python,
                simulator_python=self.simulator_python,
                simulator_source=self.simulator_source,
                runtime_root=invocation / "runtime",
                replica_assignments=assignments,
                render_gpu_assignments={
                    session_id: self.render_gpu_by_replica[replica_id]
                    for session_id, replica_id in assignments.items()
                },
                gateway_pool=agent_pool,
                simulator_pool=sim_pool,
            )
            for suite, profile in self.profiles.items()
        }
        staging = invocation / "episode_staging"
        error_root = invocation / "errors"

        def execute(key: EpisodeKey) -> int:
            started = time.time_ns()
            try:
                execution = runners[self.task_suites[key.task_id]](key)
                self._record_episode(output, staging, key, execution, started)
                return 0
            except Exception as exc:
                try:
                    self._write_error(error_root, key, exc)
                except Exception:
                    pass
                try:
                    self._record_error_episode(output, staging, key, exc, started)
                except Exception:
                    return 2
                return 1

        errored = 0
        errors = 0
        try:
            with ThreadPoolExecutor(max_workers=self.primary.resources.workers) as executor:
                futures = [executor.submit(execute, key) for key in pending]
                for future in as_completed(futures):
                    code = future.result()
                    errored += int(code == 1)
                    errors += int(code == 2)
        finally:
            if agent_pool is not None:
                agent_pool.close_all()
            if sim_pool is not None:
                sim_pool.close_all()
        manifests = self._load_manifests(output)
        report = self._report(manifests, errors, output)
        _atomic_write(output / "report.json", canonical_json_bytes(report))
        if report["complete"]:
            manifest_hashes = {
                item.key.artifact_id(): hashlib.sha256(
                    (output / "episodes" / item.key.artifact_id() / "episode.json").read_bytes()
                ).hexdigest()
                for item in manifests
            }
            final = {
                "schema_version": 1,
                "benchmark_plan_sha256": self.plan.resolved_hash(),
                "run_sha256": hashlib.sha256((output / "run.json").read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
                "episode_manifest_sha256": manifest_hashes,
                "n_complete": len(manifests),
                "finalized_ns": time.time_ns(),
            }
            final["manifest_sha256"] = mapping_sha256(final)
            _atomic_write(output / "final.json", canonical_json_bytes(final))
        if errors:
            raise RuntimeError(f"benchmark invocation had {errors} unrecordable episode errors")
        planned = len(self.plan.episodes)
        n_errored = int(report["n_errored"])
        if planned and n_errored > MAX_ERRORED_EPISODE_FRACTION * planned:
            raise RuntimeError(
                f"benchmark invocation errored on {n_errored}/{planned} episodes "
                f"(> {MAX_ERRORED_EPISODE_FRACTION:.0%}); treating as systematic breakage, not rollout noise"
            )
        if not report["complete"]:
            raise RuntimeError("benchmark invocation is incomplete")
        if header["benchmark_plan_sha256"] != report["benchmark_plan_sha256"]:
            raise RuntimeError("benchmark report plan hash differs")
        return report


def verify_benchmark_output(
    path: str | Path,
    *,
    plan_validator: Callable[[BenchmarkPlan], Any] = _validate_standard_plan,
    metric_function: Callable[[tuple[EpisodeManifest, ...]], dict[str, Any]] | None = None,
    artifact_metric_function: Callable[[tuple[EpisodeManifest, ...], Path], dict[str, Any]] | None = None,
    runtime_function: Callable[[tuple[EpisodeManifest, ...], Path], dict[str, Any]] | None = None,
    episode_manifest_validator: Callable[[EpisodeManifest], Any] | None = None,
    profile_suites: tuple[str, ...] = SUITES,
) -> dict[str, Any]:
    if metric_function is not None and artifact_metric_function is not None:
        raise StrictSchemaError("benchmark verifier accepts one metric function")
    resolved_metric_function = _benchmark_metrics if metric_function is None and artifact_metric_function is None else metric_function
    root = Path(path).resolve()
    header = fields(
        _load_json(root / "run.json"),
        {
            "schema_version",
            "benchmark_plan",
            "benchmark_plan_sha256",
            "profile_sha256",
            "code_sha256",
            "service_identities",
            "created_ns",
        },
        path="benchmark_run",
    )
    if integer(header["schema_version"], "benchmark_run.schema_version") != 1:
        raise StrictSchemaError("benchmark run schema version differs")
    plan = BenchmarkPlan.from_mapping(header["benchmark_plan"])
    if header["benchmark_plan"] != plan.to_mapping():
        raise StrictSchemaError("benchmark run plan is not canonical")
    plan_validator(plan)
    if plan.resolved_hash() != sha256(header["benchmark_plan_sha256"], "benchmark_run.plan_sha256"):
        raise StrictSchemaError("benchmark run plan hash differs")
    code_hash = sha256(header["code_sha256"], "benchmark_run.code_sha256")
    scaffold = root / "scaffold.py"
    if (
        not scaffold.is_file()
        or scaffold.is_symlink()
        or stat.S_IMODE(scaffold.stat().st_mode) != 0o444
        or mapping_sha256({"scaffold.py": hashlib.sha256(scaffold.read_bytes()).hexdigest()}) != code_hash
    ):
        raise StrictSchemaError("benchmark preserved scaffold differs")
    integer(header["created_ns"], "benchmark_run.created_ns", minimum=0)
    identities = tuple(
        ServiceIdentity.from_mapping(item)
        for item in sequence(header["service_identities"], "benchmark_run.service_identities")
    )
    if not identities:
        raise StrictSchemaError("benchmark run service identities are empty")
    profile_hashes = fields(header["profile_sha256"], set(profile_suites), path="benchmark_run.profile_sha256")
    for suite, value in profile_hashes.items():
        sha256(value, f"benchmark_run.profile_sha256.{suite}")
    report_fields = {
        "schema_version",
        "benchmark_plan_sha256",
        "n_expected",
        "n_complete",
        "n_pending",
        "errors_this_invocation",
        "n_errored",
        "complete",
        "metrics",
        "updated_ns",
    }
    if runtime_function is not None:
        report_fields.add("runtime")
    report = fields(
        _load_json(root / "report.json"),
        report_fields,
        path="benchmark_report",
    )
    if integer(report["schema_version"], "benchmark_report.schema_version") != 1:
        raise StrictSchemaError("benchmark report schema version differs")
    if boolean(report["complete"], "benchmark_report.complete") is not True:
        raise StrictSchemaError("benchmark report is incomplete")
    if sha256(report["benchmark_plan_sha256"], "benchmark_report.plan_sha256") != plan.resolved_hash():
        raise StrictSchemaError("benchmark report is incomplete or has a different plan")
    expected_count = len(plan.episodes)
    if integer(report["n_expected"], "benchmark_report.n_expected", minimum=0) != expected_count:
        raise StrictSchemaError("benchmark report expected count differs")
    if integer(report["n_complete"], "benchmark_report.n_complete", minimum=0) != expected_count:
        raise StrictSchemaError("benchmark report complete count differs")
    if integer(report["n_pending"], "benchmark_report.n_pending", minimum=0) != 0:
        raise StrictSchemaError("benchmark report pending count differs")
    if integer(report["errors_this_invocation"], "benchmark_report.errors_this_invocation", minimum=0) != 0:
        raise StrictSchemaError("benchmark report contains invocation errors")
    integer(report["updated_ns"], "benchmark_report.updated_ns", minimum=0)
    episode_root = root / "episodes"
    if not episode_root.is_dir() or episode_root.is_symlink():
        raise StrictSchemaError("benchmark episode root differs")
    expected_ids = {key.artifact_id() for key in plan.episodes}
    if {item.name for item in episode_root.iterdir()} != expected_ids:
        raise StrictSchemaError("benchmark episode directory set differs")
    manifests = tuple(
        _verify_episode_directory(episode_root / key.artifact_id(), key)
        for key in plan.episodes
    )
    if episode_manifest_validator is not None:
        for manifest in manifests:
            episode_manifest_validator(manifest)
    expected_metrics = (
        artifact_metric_function(manifests, episode_root)
        if artifact_metric_function is not None
        else resolved_metric_function(manifests)
    )
    if canonical_json_bytes(report["metrics"]) != canonical_json_bytes(expected_metrics):
        raise StrictSchemaError("benchmark report metrics differ from episode manifests")
    if runtime_function is not None:
        expected_runtime = runtime_function(manifests, episode_root)
        if canonical_json_bytes(report["runtime"]) != canonical_json_bytes(expected_runtime):
            raise StrictSchemaError("benchmark report runtime differs from episode artifacts")
    final = dict(
        fields(
            _load_json(root / "final.json"),
            {
                "schema_version",
                "benchmark_plan_sha256",
                "run_sha256",
                "report_sha256",
                "episode_manifest_sha256",
                "n_complete",
                "finalized_ns",
                "manifest_sha256",
            },
            path="benchmark_final",
        )
    )
    if integer(final["schema_version"], "benchmark_final.schema_version") != 1:
        raise StrictSchemaError("benchmark final schema version differs")
    integer(final["finalized_ns"], "benchmark_final.finalized_ns", minimum=0)
    manifest_hash = sha256(final.pop("manifest_sha256"), "benchmark_final.manifest_sha256")
    if mapping_sha256(final) != manifest_hash:
        raise StrictSchemaError("benchmark final manifest hash differs")
    final["manifest_sha256"] = manifest_hash
    if final["benchmark_plan_sha256"] != plan.resolved_hash():
        raise StrictSchemaError("benchmark final plan differs")
    if final["run_sha256"] != hashlib.sha256((root / "run.json").read_bytes()).hexdigest():
        raise StrictSchemaError("benchmark run hash differs")
    if final["report_sha256"] != hashlib.sha256((root / "report.json").read_bytes()).hexdigest():
        raise StrictSchemaError("benchmark report hash differs")
    episode_hashes = final["episode_manifest_sha256"]
    if not isinstance(episode_hashes, Mapping) or set(episode_hashes) != expected_ids:
        raise StrictSchemaError("benchmark episode manifest set differs")
    for key in plan.episodes:
        source = root / "episodes" / key.artifact_id() / "episode.json"
        recorded = sha256(
            episode_hashes[key.artifact_id()],
            f"benchmark_final.episode_manifest_sha256.{key.artifact_id()}",
        )
        if recorded != hashlib.sha256(source.read_bytes()).hexdigest():
            raise StrictSchemaError("benchmark episode manifest hash differs")
    if integer(final["n_complete"], "benchmark_final.n_complete") != len(plan.episodes):
        raise StrictSchemaError("benchmark final episode count differs")
    return final
