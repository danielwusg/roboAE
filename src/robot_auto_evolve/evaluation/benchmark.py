from __future__ import annotations

import json
import os
import shutil
import stat
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from robot_auto_evolve.agent import ToolEndpoint
from robot_auto_evolve.benchmarks.libero_suites import LIBERO_TASK_SUITE
from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import (
    boolean,
    fields,
    integer,
    json_object,
    reject_json_constant,
)
from robot_auto_evolve.provenance import (
    ArtifactDescriptor,
    BenchmarkPlan,
    EpisodeKey,
    EpisodeManifest,
    canonical_json_bytes,
)
from robot_auto_evolve.services import MsgpackServiceClient, ReplicaScheduler, ServiceReplica

MAX_ERRORED_EPISODE_FRACTION = 0.5
MAX_EPISODE_RETRY_ATTEMPTS = 1

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


def _preserve_scaffold(output: Path, scaffold: Path) -> None:
    payload = (scaffold / "scaffold.py").read_bytes()
    target = output / "scaffold.py"
    if not target.exists():
        _atomic_write(target, payload)
        target.chmod(0o444)
        return
    if not target.is_file() or target.is_symlink():
        raise StrictSchemaError("benchmark preserved scaffold differs")
    if target.read_bytes() != payload:
        raise StrictSchemaError(
            "benchmark output already holds a different scaffold.py; this evaluation directory "
            "belongs to another scaffold"
        )
    if stat.S_IMODE(target.stat().st_mode) != 0o444:
        target.chmod(0o444)


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
        if not source.is_file() or source.is_symlink():
            raise StrictSchemaError("benchmark episode artifact is absent")
    return manifest


class CanonicalBenchmarkEvaluator:
    def __init__(
        self,
        profiles: Mapping[str, Profile],
        plan: BenchmarkPlan,
        *,
        agent_python: Path,
        simulator_python: Path,
        simulator_source: Path,
        coordinator: Any,
        task_suites: Mapping[str, str] | None = None,
        artifact_metric_function: Callable[[tuple[EpisodeManifest, ...], Path], dict[str, Any]] | None = None,
        episode_manifest_validator: Callable[[EpisodeManifest], Any] | None = None,
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
        primary = (
            self.profiles["libero_spatial"]
            if "libero_spatial" in self.profiles
            else next(iter(self.profiles.values()))
        )
        for suite, profile in self.profiles.items():
            if not isinstance(profile, Profile) or profile.environment.suite != suite:
                raise StrictSchemaError("benchmark evaluator profile suite differs")
            profile.validate()
            if profile.policy.to_mapping() != primary.policy.to_mapping():
                raise StrictSchemaError("benchmark evaluator policy profiles differ")
            if [tool.to_mapping() for tool in profile.tools] != [tool.to_mapping() for tool in primary.tools]:
                raise StrictSchemaError("benchmark evaluator tool profiles differ")
        if primary.policy.deployment_mode != "replicated":
            raise StrictSchemaError("benchmark evaluator requires replicated policy services")
        self.task_suites = dict(LIBERO_TASK_SUITE if task_suites is None else task_suites)
        if {item.task_id for item in plan.episodes} - set(self.task_suites):
            raise StrictSchemaError("benchmark evaluator task-to-suite map is incomplete")
        if set(self.task_suites.values()) - set(self.profiles):
            raise StrictSchemaError("benchmark evaluator task-to-suite map references an absent profile")
        self.artifact_metric_function = artifact_metric_function
        self.episode_manifest_validator = episode_manifest_validator
        self.plan = plan
        self.primary = primary
        self.coordinator = coordinator
        self.agent_python = Path(agent_python).resolve()
        self.simulator_python = Path(simulator_python).resolve()
        self.simulator_source = Path(simulator_source).resolve()
        self.tool_profiles = {
            tool.capability: tool for tool in primary.tools if tool.enabled and tool.service is not None
        }

    def _policy_replicas(self, runtime_plan: Any) -> tuple[ServiceReplica, ...]:
        clients = dict(runtime_plan.policy_clients)
        if not clients:
            return ()
        replicas = []
        for item in self.primary.policy.replicas:
            key = (item.identity.service_name, item.identity.replica_id)
            if key not in clients:
                raise StrictSchemaError("benchmark evaluator policy client is absent")
            replicas.append(ServiceReplica(item.endpoint, clients[key].validate_identity(), clients[key]))
        return tuple(replicas)

    def _tool_endpoints(self, runtime_plan: Any) -> dict[str, ToolEndpoint]:
        endpoints = {}
        for capability in runtime_plan.capabilities:
            if capability not in self.tool_profiles:
                continue
            tool = self.tool_profiles[capability]
            endpoints[capability] = ToolEndpoint(
                tool.service.endpoint,
                tool.service.identity,
                tool.required,
                timeout_s=TOOL_CALL_TIMEOUT_S,
            )
        return endpoints

    def _header(self, created_ns: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark_plan": self.plan.to_mapping(),
            "created_ns": created_ns,
        }

    def _prepare(self, output: Path) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        (output / "episodes").mkdir(exist_ok=True)
        path = output / "run.json"
        if not path.exists():
            value = self._header(time.time_ns())
            _atomic_write(path, canonical_json_bytes(value))
            return value
        value = fields(
            _load_json(path),
            {"schema_version", "benchmark_plan", "created_ns"},
            path="benchmark_run",
        )
        if value["benchmark_plan"] != self.plan.to_mapping():
            raise StrictSchemaError("benchmark run episode plan differs from the plan being evaluated")
        integer(value["created_ns"], "benchmark_run.created_ns", minimum=0)
        return dict(value)

    def _load_manifests(self, output: Path) -> tuple[EpisodeManifest, ...]:
        key_by_id = {key.artifact_id(): key for key in self.plan.episodes}
        directories = {path.name: path for path in (output / "episodes").iterdir()}
        unknown = set(directories) - set(key_by_id)
        if unknown:
            raise StrictSchemaError(f"benchmark output contains unknown episode {sorted(unknown)[0]}")
        manifests = tuple(
            _verify_episode_directory(directories[artifact_id], key)
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
            ArtifactDescriptor(name, len(payload)) for name, payload in sorted(execution.artifacts.items())
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
        if complete and self.artifact_metric_function is not None:
            metrics = self.artifact_metric_function(manifests, output / "episodes")
        return {
            "schema_version": 1,
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
        self._prepare(output)
        _preserve_scaffold(output, scaffold)
        if (output / "final.json").is_file():
            verify_benchmark_output(
                output,
                plan=self.plan,
                artifact_metric_function=self.artifact_metric_function,
                episode_manifest_validator=self.episode_manifest_validator,
            )
            return dict(_load_json(output / "report.json"))
        existing = self._load_manifests(output)
        retry = [item.key for item in existing if item.state == "error"]
        for key in retry:
            shutil.rmtree(output / "episodes" / key.artifact_id(), ignore_errors=True)
        existing_keys = {item.key for item in existing if item.state == "complete"}
        pending = [key for key in self.plan.episodes if key not in existing_keys]
        agent_pool = None
        sim_pool = None
        runners: dict[str, Any] = {}
        staging = invocation / "episode_staging"
        error_root = invocation / "errors"
        workers = 1
        if pending:
            runtime_plan = self.coordinator.plan_for((scaffold / "scaffold.py").read_text(encoding="utf-8"))
            _atomic_write(output / "runtime_plan.json", canonical_json_bytes(runtime_plan.to_mapping()))
            workers = runtime_plan.workers
            tool_endpoints = self._tool_endpoints(runtime_plan)
            replicas = self._policy_replicas(runtime_plan)
            replica_count = len(replicas)
            scheduler = None
            assignments = {}
            if replica_count:
                scheduler = ReplicaScheduler(
                    replicas, max_sessions_per_replica=runtime_plan.sessions_per_policy
                )
                assignments = {
                    key.artifact_id(): replicas[index % replica_count].identity.replica_id
                    for index, key in enumerate(self.plan.episodes)
                }
            render_gpu_ids = runtime_plan.render_gpu_ids
            render_gpu_assignments = {
                key.artifact_id(): render_gpu_ids[index % len(render_gpu_ids)]
                for index, key in enumerate(self.plan.episodes)
            }
            agent_pool = AgentGatewayPool(invocation / "agent_pool") if self.reuse_agent else None
            sim_pool = SimulatorProcessPool(invocation / "simulator_pool") if self.reuse_sim else None
            runners = {
                suite: ProfileEpisodeRunner(
                    profile,
                    scaffold,
                    scheduler,
                    tool_endpoints,
                    agent_python=self.agent_python,
                    simulator_python=self.simulator_python,
                    simulator_source=self.simulator_source,
                    runtime_root=invocation / "runtime",
                    replica_assignments=assignments,
                    render_gpu_assignments=render_gpu_assignments,
                    gateway_pool=agent_pool,
                    simulator_pool=sim_pool,
                )
                for suite, profile in self.profiles.items()
            }

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

        def run_pass(keys: list[EpisodeKey]) -> tuple[int, int]:
            failed = 0
            unrecordable = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(execute, key) for key in keys]
                for future in as_completed(futures):
                    code = future.result()
                    failed += int(code == 1)
                    unrecordable += int(code == 2)
            return failed, unrecordable

        errored = 0
        errors = 0
        attempts: list[dict[str, Any]] = []
        try:
            if pending:
                errored, errors = run_pass(list(pending))
                attempts.append({"attempt": 0, "n_run": len(pending), "n_errored": errored})
                for index in range(MAX_EPISODE_RETRY_ATTEMPTS):
                    if errored == 0 or errors:
                        break
                    again = [item.key for item in self._load_manifests(output) if item.state == "error"]
                    if not again:
                        break
                    for key in again:
                        shutil.rmtree(output / "episodes" / key.artifact_id(), ignore_errors=True)
                    errored, errors = run_pass(again)
                    attempts.append({"attempt": index + 1, "n_run": len(again), "n_errored": errored})
        finally:
            if agent_pool is not None:
                agent_pool.close_all()
            if sim_pool is not None:
                sim_pool.close_all()
        if len(attempts) > 1:
            _atomic_write(
                output / "retries.json",
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "max_retry_attempts": MAX_EPISODE_RETRY_ATTEMPTS,
                        "attempts": attempts,
                        "updated_ns": time.time_ns(),
                    }
                ),
            )
        manifests = self._load_manifests(output)
        report = self._report(manifests, errors, output)
        _atomic_write(output / "report.json", canonical_json_bytes(report))
        if report["complete"]:
            _atomic_write(
                output / "final.json",
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "n_complete": len(manifests),
                        "finalized_ns": time.time_ns(),
                    }
                ),
            )
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
        return report


def verify_benchmark_output(
    path: str | Path,
    *,
    plan: BenchmarkPlan | None = None,
    artifact_metric_function: Callable[[tuple[EpisodeManifest, ...], Path], dict[str, Any]] | None = None,
    episode_manifest_validator: Callable[[EpisodeManifest], Any] | None = None,
) -> dict[str, Any]:
    root = Path(path).resolve()
    header = fields(
        _load_json(root / "run.json"),
        {"schema_version", "benchmark_plan", "created_ns"},
        path="benchmark_run",
    )
    if integer(header["schema_version"], "benchmark_run.schema_version") != 1:
        raise StrictSchemaError("benchmark run schema version differs")
    recorded_plan = BenchmarkPlan.from_mapping(header["benchmark_plan"])
    if plan is not None and recorded_plan.to_mapping() != plan.to_mapping():
        raise StrictSchemaError("benchmark resumed plan differs")
    integer(header["created_ns"], "benchmark_run.created_ns", minimum=0)
    scaffold = root / "scaffold.py"
    if not scaffold.is_file() or scaffold.is_symlink():
        raise StrictSchemaError("benchmark preserved scaffold is absent")
    report = fields(
        _load_json(root / "report.json"),
        {
            "schema_version",
            "n_expected",
            "n_complete",
            "n_pending",
            "errors_this_invocation",
            "n_errored",
            "complete",
            "metrics",
            "updated_ns",
        },
        path="benchmark_report",
    )
    if integer(report["schema_version"], "benchmark_report.schema_version") != 1:
        raise StrictSchemaError("benchmark report schema version differs")
    if boolean(report["complete"], "benchmark_report.complete") is not True:
        raise StrictSchemaError("benchmark report is incomplete")
    expected_count = len(recorded_plan.episodes)
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
    expected_ids = {key.artifact_id() for key in recorded_plan.episodes}
    if {item.name for item in episode_root.iterdir()} != expected_ids:
        raise StrictSchemaError("benchmark episode directory set differs")
    manifests = tuple(
        _verify_episode_directory(episode_root / key.artifact_id(), key) for key in recorded_plan.episodes
    )
    if episode_manifest_validator is not None:
        for manifest in manifests:
            episode_manifest_validator(manifest)
    if artifact_metric_function is not None:
        expected_metrics = artifact_metric_function(manifests, episode_root)
        if canonical_json_bytes(report["metrics"]) != canonical_json_bytes(expected_metrics):
            raise StrictSchemaError("benchmark report metrics differ from episode manifests")
    final = fields(
        _load_json(root / "final.json"),
        {"schema_version", "n_complete", "finalized_ns"},
        path="benchmark_final",
    )
    if integer(final["schema_version"], "benchmark_final.schema_version") != 1:
        raise StrictSchemaError("benchmark final schema version differs")
    integer(final["finalized_ns"], "benchmark_final.finalized_ns", minimum=0)
    if integer(final["n_complete"], "benchmark_final.n_complete") != expected_count:
        raise StrictSchemaError("benchmark final episode count differs")
    return dict(final)
