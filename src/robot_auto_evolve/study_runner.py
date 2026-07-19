from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from robot_auto_evolve.config import Profile
from robot_auto_evolve.benchmarks.libero_pro import HARNESS_SUITES, HARNESS_TASK_SUITE
from robot_auto_evolve.benchmarks.libero_suites import LIBERO_SUITE_TASKS, LIBERO_TASK_SUITE
from robot_auto_evolve.evaluation import BenchmarkOutcome, compute_benchmark_scalar
from robot_auto_evolve.evaluation.benchmark import CanonicalBenchmarkEvaluator
from robot_auto_evolve.evolution import (
    BenchmarkEvolutionDriver,
    CanonicalBenchmarkEvolutionAdapter,
    ClaudeRevisionBackend,
    RelayLimits,
    canonical_outcome_metrics,
)
from robot_auto_evolve.evolution.free_backend import ClaudeFreeRevisionBackend
from robot_auto_evolve.operator_catalog import (
    StudyRequest,
    materialize_runtime_profile,
    materialize_runtime_profiles,
)
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest
from robot_auto_evolve.runtime import ProfileServiceRuntime, resolve_profile_launch_paths
from robot_auto_evolve.runtime_paths import RuntimePaths, assert_clean_import_origin, project_root_from_package


CLAUDE_CREDENTIAL_ENV = "ROBOT_AE_CLAUDE_CREDENTIAL_DIR"


@dataclass(frozen=True)
class StudyContext:
    request: StudyRequest
    request_path: Path
    project_root: Path
    run_root: Path
    runtime_root: Path
    runtime_profile_path: Path
    runtime_profile_paths: Mapping[str, Path]
    profile: Profile
    profiles: Mapping[str, Profile]
    runtime_paths: RuntimePaths
    launch_paths: Mapping[str, Path]
    seed_scaffold: Path
    claude_executable: Path
    claude_credential_dir: Path | None
    claude_isolation_dir: Path
    meta_backend: str
    smoke_episodes: int
    smoke_horizon: int


def study_runtime_paths(project_root: str | Path, study_id: str) -> dict[str, Path]:
    if type(study_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", study_id) is None:
        raise StrictSchemaError("study ID differs")
    root = Path(project_root).resolve()
    run = root / "runs" / study_id
    runtime = run / "runtime"
    return {
        "run": run,
        "runtime": runtime,
        "evolution": run / "evolution",
        "invocations": run / "invocations",
        "runtime_invocations": runtime / "invocations",
        "claude_isolation": runtime / "claude_isolation",
    }


def _project_path(root: Path, value: Any, name: str) -> Path:
    if type(value) is not str:
        raise StrictSchemaError(f"{name} path differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise StrictSchemaError(f"{name} path differs")
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise StrictSchemaError(f"{name} path escapes project")
    return result


def _validate_credential_directory(value: str, project_root: Path) -> Path:
    path = Path(value).absolute()
    directory = path.lstat()
    if (
        not stat.S_ISDIR(directory.st_mode)
        or stat.S_ISLNK(directory.st_mode)
        or directory.st_uid != os.getuid()
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise PermissionError("Claude credential directory must be private and owned by the current user")
    token = path / "oauth_token"
    token_stat = token.lstat()
    if (
        not stat.S_ISREG(token_stat.st_mode)
        or stat.S_ISLNK(token_stat.st_mode)
        or token_stat.st_uid != os.getuid()
        or token_stat.st_nlink != 1
        or stat.S_IMODE(token_stat.st_mode) != 0o600
        or not 1 <= token_stat.st_size <= 8192
    ):
        raise PermissionError("Claude OAuth token file must be private, regular, and singly linked")
    resolved = path.resolve()
    if resolved == project_root or resolved.is_relative_to(project_root):
        raise PermissionError("Claude credential directory must be outside the project")
    return resolved


def _claude_executable() -> Path:
    value = shutil.which("claude")
    if value is None:
        raise FileNotFoundError("claude executable is absent from PATH")
    path = Path(value).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise PermissionError("claude executable is not an executable regular file")
    return path


def load_study_context(
    study_request_path: str | Path,
    *,
    target_candidates: int,
    finalize: bool,
    run_transfer: bool,
    project_root: str | Path | None = None,
    meta_backend: str = "claude",
    smoke_episodes: int = 0,
    smoke_horizon: int = 0,
) -> StudyContext:
    root = Path(project_root or project_root_from_package()).resolve()
    assert_clean_import_origin(root)
    source = Path(study_request_path).resolve()
    request = StudyRequest.load(source, root)
    run_root = source.parent
    layout = study_runtime_paths(root, request.study_id)
    expected = layout["run"]
    if source.name != "study_request.json" or run_root != expected:
        raise StrictSchemaError("study request must be runs/<study-id>/study_request.json")
    if type(target_candidates) is not int or not 0 <= target_candidates <= request.candidate_budget:
        raise ValueError("target candidates falls outside the route candidate budget")
    if run_transfer and not finalize:
        raise ValueError("--run-transfer requires --finalize")
    if run_transfer and request.transfer_plan is None:
        raise ValueError("--run-transfer requires a related-transfer study request")
    runtime_profile_path = materialize_runtime_profile(request, run_root)
    runtime_profile_paths = materialize_runtime_profiles(request, run_root)
    profiles = {
        key: Profile.load(path, project_root=root)
        for key, path in sorted(runtime_profile_paths.items())
    }
    primary_key = request.route_spec["primary_profile_key"]
    if primary_key not in profiles:
        raise StrictSchemaError("runtime primary profile differs")
    profile = profiles[primary_key]
    runtime_paths = RuntimePaths.load(root)
    launch_paths = resolve_profile_launch_paths(profile, root, runtime_paths.environment_root)
    starting_agent = request.route_spec.get("starting_agent")
    if not isinstance(starting_agent, dict):
        raise StrictSchemaError("route starting agent differs")
    seed_scaffold = _project_path(root, starting_agent.get("scaffold"), "route starting scaffold")
    if not seed_scaffold.is_dir() or seed_scaffold.is_symlink():
        raise FileNotFoundError("route starting scaffold is absent")
    executable = _claude_executable()
    if profile.meta_loop.coding_backend != "claude" or profile.meta_loop.coding_model is None:
        raise StrictSchemaError("runtime profile does not declare a Claude coding model")
    runtime_root = layout["runtime"]
    isolation = layout["claude_isolation"]
    if meta_backend == "claude_free":
        # Revision 8: the freer backend uses the ambient Claude credential (like the
        # prior multimodel/roboAutoEvol mechanism) -- no private oauth_token dir, no relay.
        credential_dir = None
    else:
        credential_value = os.environ.get(CLAUDE_CREDENTIAL_ENV)
        if not credential_value:
            raise PermissionError(f"set {CLAUDE_CREDENTIAL_ENV} to the private Claude credential directory")
        credential_dir = _validate_credential_directory(credential_value, root)
        if credential_dir == isolation or credential_dir.is_relative_to(runtime_root):
            raise PermissionError("Claude credentials must stay outside run runtime state")
    return StudyContext(
        request=request,
        request_path=source,
        project_root=root,
        run_root=run_root,
        runtime_root=runtime_root,
        runtime_profile_path=runtime_profile_path,
        runtime_profile_paths=runtime_profile_paths,
        profile=profile,
        profiles=profiles,
        runtime_paths=runtime_paths,
        launch_paths=launch_paths,
        seed_scaffold=seed_scaffold,
        claude_executable=executable,
        claude_credential_dir=credential_dir,
        claude_isolation_dir=isolation,
        meta_backend=meta_backend,
        smoke_episodes=smoke_episodes,
        smoke_horizon=smoke_horizon,
    )


def _simulator_source_key(suite: str) -> str:
    if suite == "robocerebra_public60":
        return "robocerebra_source"
    if suite.startswith("libero_pro_"):
        return "libero_pro_source"
    if suite.startswith("calvin"):
        return "calvin_source"
    if suite.startswith("simpler_"):
        return "simpler_source"
    if suite == "robotwin2_demo_clean":
        return "robotwin2_source"
    if suite == "vlabench_xvla_tracks_1_4":
        return "vlabench_source"
    if suite == "robocasa365_target":
        return "robocasa365_source"
    if suite == "robolab120_droid_jointpos":
        return "robolab_source"
    return "libero_source"


def benchmark_scalar_report(metric: str, outcomes: tuple[BenchmarkOutcome, ...]) -> dict[str, Any]:
    scalar = compute_benchmark_scalar(metric, outcomes)
    return {
        "metric": scalar.metric,
        "score": scalar.value,
        "details": dict(scalar.details),
    }


def _canonical_metric_report(
    manifests: tuple[EpisodeManifest, ...],
    episode_root: Path,
    metric: str,
) -> dict[str, Any]:
    outcomes = tuple(
        BenchmarkOutcome(
            manifest.key,
            canonical_outcome_metrics(episode_root / manifest.key.artifact_id(), manifest, metric),
        )
        for manifest in manifests
    )
    return benchmark_scalar_report(metric, outcomes)


def _canonical_evaluator(
    context: StudyContext,
    plan: BenchmarkPlan,
    clients: Mapping[tuple[str, str], Any],
    invocation_root: Path,
) -> CanonicalBenchmarkEvolutionAdapter:
    suite = context.profile.environment.suite
    profile_keys = set(context.profiles)
    if len(profile_keys) == 1:
        task_suites = {item.task_id: suite for item in plan.episodes}
    elif profile_keys == set(LIBERO_SUITE_TASKS):
        task_suites = {item.task_id: LIBERO_TASK_SUITE[item.task_id] for item in plan.episodes}
    elif profile_keys == set(HARNESS_SUITES):
        task_suites = {item.task_id: HARNESS_TASK_SUITE[item.task_id] for item in plan.episodes}
    else:
        raise StrictSchemaError("study runtime profile set has no benchmark task routing")
    evaluator = CanonicalBenchmarkEvaluator(
        context.profiles,
        plan,
        agent_python=context.launch_paths["agent_python"],
        simulator_python=context.launch_paths["simulator_python"],
        simulator_source=context.launch_paths[_simulator_source_key(suite)],
        live_clients=clients,
        task_suites=task_suites,
        artifact_metric_function=lambda manifests, root: _canonical_metric_report(
            manifests,
            root,
            context.request.scalar_metric,
        ),
        render_gpu_ids=tuple(context.request.mapping["resources"]["render_gpu_ids"]),
    )
    return CanonicalBenchmarkEvolutionAdapter(
        evaluator,
        plan,
        context.request.scalar_metric,
        invocation_root=invocation_root,
    )


def _revision_backend(context: StudyContext):
    loop = context.profile.meta_loop
    if context.meta_backend == "claude_free":
        # Revision 8: plain `claude` subprocess with a shell, edits scaffold.py in place,
        # prior-isolated. Matches the prior multimodel/roboAutoEvol mechanism.
        return ClaudeFreeRevisionBackend(
            context.claude_executable,
            str(loop.coding_model),
            timeout_s=loop.timeout_s,
            max_turns=loop.max_turns,
        )
    return ClaudeRevisionBackend(
        context.claude_executable,
        context.claude_isolation_dir,
        str(loop.coding_model),
        loop.timeout_s,
        loop.max_turns,
        credential_dir=context.claude_credential_dir,
        relay_limits=RelayLimits(
            max_requests=loop.api_request_budget,
            max_request_bytes=loop.api_request_max_bytes,
            max_response_bytes=loop.api_response_max_bytes,
            deadline_s=loop.timeout_s,
            provider_timeout_s=min(300.0, loop.timeout_s),
        ),
        public_evidence_format="full_benchmark",
    )


def _frozen_paths(context: StudyContext) -> tuple[Path, ...]:
    request = context.request
    references = (
        request.mapping["route_spec"],
        request.mapping["benchmark_plan"],
        request.mapping["standard_source_plan"],
        *request.mapping["profiles"].values(),
    )
    paths = [
        context.request_path,
        context.runtime_profile_path,
        context.runtime_profile_path.with_name("profile_materialization.json"),
        *context.runtime_profile_paths.values(),
        context.runtime_paths.config_path,
        context.project_root / "locks" / "runtime_artifacts.json",
        context.project_root / "src" / "robot_auto_evolve",
        context.seed_scaffold,
    ]
    paths.extend(_project_path(context.project_root, item["path"], "study reference") for item in references)
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _next_invocation(context: StudyContext) -> tuple[Path, Path]:
    layout = study_runtime_paths(context.project_root, context.request.study_id)
    root = layout["invocations"]
    scratch_root = layout["runtime_invocations"]
    root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    indices = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    index = max(indices, default=0) + 1
    while True:
        target = root / f"{index:06d}"
        try:
            target.mkdir()
        except FileExistsError:
            index += 1
            continue
        scratch = scratch_root / target.name
        try:
            scratch.mkdir()
            return target, scratch
        except FileExistsError:
            target.rmdir()
            index += 1
        except BaseException:
            target.rmdir()
            raise


def _write_invocation(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _capture_system(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    commands = {
        "hostname.txt": ("hostname",),
        "nproc.txt": ("nproc",),
        "gpus.txt": ("nvidia-smi", "-L"),
        "gpu_state.csv": (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ),
        "gpu_processes.csv": (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv",
        ),
        "host_processes.txt": ("ps", "-eo", "pid,ppid,pgid,sid,stat,etimes,cmd", "--sort=pid"),
    }
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, check=False, timeout=10.0)
        (path / name).write_bytes(result.stdout + result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"system capture failed: {' '.join(command)}")


def _smoke_plan(plan: BenchmarkPlan, episodes_per_task: int, horizon_cap: int) -> BenchmarkPlan:
    """Runtime-only shrink of a benchmark plan for a smoke test: keep at most
    ``episodes_per_task`` episodes per task and cap each episode's horizon. This is
    not a hash-pinned artifact -- the driver runs whatever plan object it is handed,
    so a smoke plan needs no re-pin. Used only when --smoke-episodes > 0."""
    if episodes_per_task < 1:
        raise ValueError("smoke episodes per task must be positive")
    counts: dict[str, int] = {}
    kept = []
    for episode in plan.episodes:
        taken = counts.get(episode.task_id, 0)
        if taken >= episodes_per_task:
            continue
        counts[episode.task_id] = taken + 1
        horizon = episode.horizon if horizon_cap <= 0 else min(episode.horizon, horizon_cap)
        kept.append(replace(episode, horizon=horizon))
    if not kept:
        raise ValueError("smoke plan is empty")
    return replace(plan, episodes=tuple(sorted(kept)))


def execute_study(
    context: StudyContext,
    *,
    target_candidates: int,
    finalize: bool,
    run_transfer: bool,
) -> dict[str, Any]:
    invocation, scratch = _next_invocation(context)
    started = time.time_ns()
    _write_invocation(
        invocation / "request.json",
        {
            "schema_version": 1,
            "study_id": context.request.study_id,
            "target_candidates": target_candidates,
            "finalize": finalize,
            "run_transfer": run_transfer,
            "started_ns": started,
        },
    )
    failure: BaseException | None = None
    evidence_failure: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        _capture_system(invocation / "system_before")
        runtime = ProfileServiceRuntime(
            context.profile,
            project_root=context.project_root,
            environment_root=context.runtime_paths.environment_root,
            log_root=scratch / "services",
        )
        with runtime as clients:
            _capture_system(invocation / "system_ready")
            evolve_plan = context.request.evolve_plan
            transfer_plan = context.request.transfer_plan
            if context.smoke_episodes > 0:
                evolve_plan = _smoke_plan(evolve_plan, context.smoke_episodes, context.smoke_horizon)
                transfer_plan = (
                    None
                    if transfer_plan is None
                    else _smoke_plan(transfer_plan, context.smoke_episodes, context.smoke_horizon)
                )
            evolve_evaluator = _canonical_evaluator(
                context,
                evolve_plan,
                clients,
                scratch / "evaluators" / "evolve",
            )
            transfer_evaluator = (
                None
                if transfer_plan is None
                else _canonical_evaluator(
                    context,
                    transfer_plan,
                    clients,
                    scratch / "evaluators" / "transfer",
                )
            )
            driver = BenchmarkEvolutionDriver(
                seed_scaffold=context.seed_scaffold,
                run_dir=study_runtime_paths(context.project_root, context.request.study_id)["evolution"],
                plan=evolve_plan,
                scalar_metric=context.request.scalar_metric,
                evaluator=evolve_evaluator,
                revision_backend=_revision_backend(context),
                candidate_budget=context.request.candidate_budget,
                frozen_paths=_frozen_paths(context),
                transfer_plan=transfer_plan,
                transfer_metric=context.request.scalar_metric if transfer_plan is not None else None,
                transfer_evaluator=transfer_evaluator,
            )
            state = driver.advance_to(target_candidates, finalize=finalize)
            transfer = driver.run_transfer().to_mapping() if run_transfer else None
            result = {"state": state, "transfer": transfer}
            _write_invocation(invocation / "result.json", result)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            _capture_system(invocation / "system_after_cleanup")
        except BaseException as exc:
            evidence_failure = exc
        _write_invocation(
            invocation / "finished.json",
            {
                "schema_version": 1,
                "started_ns": started,
                "finished_ns": time.time_ns(),
                "status": "complete" if failure is None and evidence_failure is None else "failed",
                "error_type": (
                    None
                    if failure is None and evidence_failure is None
                    else type(failure if failure is not None else evidence_failure).__name__
                ),
            },
        )
    if failure is not None:
        raise failure
    if evidence_failure is not None:
        raise evidence_failure
    if result is None:
        raise RuntimeError("study execution ended without a result")
    return result


def run_study(
    study_request_path: str | Path,
    *,
    target_candidates: int,
    finalize: bool = False,
    run_transfer: bool = False,
    meta_backend: str = "claude",
    smoke_episodes: int = 0,
    smoke_horizon: int = 0,
) -> dict[str, Any]:
    context = load_study_context(
        study_request_path,
        target_candidates=target_candidates,
        finalize=finalize,
        run_transfer=run_transfer,
        meta_backend=meta_backend,
        smoke_episodes=smoke_episodes,
        smoke_horizon=smoke_horizon,
    )
    return execute_study(
        context,
        target_candidates=target_candidates,
        finalize=finalize,
        run_transfer=run_transfer,
    )


__all__ = [
    "CLAUDE_CREDENTIAL_ENV",
    "StudyContext",
    "benchmark_scalar_report",
    "execute_study",
    "load_study_context",
    "run_study",
    "study_runtime_paths",
]
