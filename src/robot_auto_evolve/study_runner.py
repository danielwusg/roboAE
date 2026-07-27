from __future__ import annotations

import json
import os
import re
import shutil
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
    ClaudeFreeRevisionBackend,
    canonical_outcome_metrics,
)
from robot_auto_evolve.evolution.profile_evaluator import reuse_sim_allowed
from robot_auto_evolve.operator_catalog import (
    StudyRequest,
    materialize_runtime_profile,
    materialize_runtime_profiles,
)
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest
from robot_auto_evolve.runtime import ProfileServiceRuntime, resolve_profile_launch_paths
from robot_auto_evolve.runtime_paths import RuntimePaths, assert_clean_import_origin, project_root_from_package




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
    claude_isolation_dir: Path
    meta_backend: str
    smoke_episodes: int
    smoke_horizon: int
    smoke_no_tools: bool


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




def _claude_executable() -> Path:
    value = shutil.which("claude")
    if value is None:
        raise FileNotFoundError("claude executable is absent from PATH")
    path = Path(value).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise PermissionError("claude executable is not an executable regular file")
    return path


def _suppress_tools(profile: Profile) -> Profile:
    """Runtime-only: copy the profile with every tool suppressed (disabled, no service)
    for --smoke-no-tools. No tool services are launched and the scaffold sees no tools;
    the pinned profile file is untouched (this rebuilds the in-memory object only)."""
    if not profile.tools:
        return profile
    suppressed = tuple(
        replace(
            tool,
            enabled=False,
            required=False,
            availability="unavailable",
            blocker="tools suppressed for smoke (--smoke-no-tools)",
            service=None,
        )
        for tool in profile.tools
    )
    return replace(profile, tools=suppressed)


def load_study_context(
    study_request_path: str | Path,
    *,
    target_candidates: int,
    finalize: bool,
    run_transfer: bool,
    project_root: str | Path | None = None,
    meta_backend: str = "claude_free",
    smoke_episodes: int = 0,
    smoke_horizon: int = 0,
    smoke_no_tools: bool = False,
    seed_scaffold_override: str | Path | None = None,
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
    if smoke_no_tools:
        profiles = {key: _suppress_tools(value) for key, value in profiles.items()}
    primary_key = request.route_spec["primary_profile_key"]
    if primary_key not in profiles:
        raise StrictSchemaError("runtime primary profile differs")
    profile = profiles[primary_key]
    runtime_paths = RuntimePaths.load(root)
    launch_paths = resolve_profile_launch_paths(profile, root, runtime_paths.environment_root)
    starting_agent = request.route_spec.get("starting_agent")
    if not isinstance(starting_agent, dict):
        raise StrictSchemaError("route starting agent differs")
    # --seed-scaffold overrides the route's default starting scaffold (e.g. to run the bare
    # policy_passthrough_seed instead of the designed volo_harness_seed) without editing route.json.
    scaffold_rel = seed_scaffold_override if seed_scaffold_override else starting_agent.get("scaffold")
    seed_scaffold = _project_path(root, scaffold_rel, "route starting scaffold")
    if not seed_scaffold.is_dir() or seed_scaffold.is_symlink():
        raise FileNotFoundError("route starting scaffold is absent")
    executable = _claude_executable()
    if profile.meta_loop.coding_backend != "claude" or profile.meta_loop.coding_model is None:
        raise StrictSchemaError("runtime profile does not declare a Claude coding model")
    runtime_root = layout["runtime"]
    isolation = layout["claude_isolation"]
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
        claude_isolation_dir=isolation,
        meta_backend=meta_backend,
        smoke_episodes=smoke_episodes,
        smoke_horizon=smoke_horizon,
        smoke_no_tools=smoke_no_tools,
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
        reuse_agent=bool(context.request.mapping["resources"].get("reuse_agent", False)),
        # W3-C3: honor --reuse-sim only for suites PROVEN byte-equivalent under subprocess reuse
        # (fail-safe allowlist in profile_evaluator.reuse_sim_allowed) AND only when the whole route
        # runs a SINGLE suite. A MULTI-suite route -- the LIBERO-Pro 8-cell aggregate
        # `rlinf_pi05_libero_pro`, whose profiles span 8 cell-suites -- shares ONE SimulatorProcessPool
        # (keyed by worker-thread + render-GPU) across its per-suite runners (benchmark.py:500-527), so a
        # subprocess first built for cell A is later handed a cell-B episode and the worker's suite check
        # rejects it ("LIBERO-Pro episode task and profile suite differ", libero_pro_worker.py:103).
        # Such routes MUST use the proven per-episode-subprocess path. (s20: found on the first aggregate
        # full-loop test -- 62/80 baseline episodes errored; single-cell LIBERO-Pro routes are one suite
        # and keep reuse-sim.) Every other suite likewise runs per-episode even when --reuse-sim is set.
        reuse_sim=bool(context.request.mapping["resources"].get("reuse_sim", False))
        and reuse_sim_allowed(context.profile.environment.suite)
        and len({profile.environment.suite for profile in context.profiles.values()}) == 1,
    )
    return CanonicalBenchmarkEvolutionAdapter(
        evaluator,
        plan,
        context.request.scalar_metric,
        invocation_root=invocation_root,
    )


def _route_notes(context: StudyContext) -> str:
    """Plain-language, per-route facts pasted into the coding agent's revision prompt.

    Revision 1 (§2.4) of the 26 July plan: every line here is something the agent previously had
    to discover by reading harness source or by burning a candidate. It is built from the LIVE
    runtime profile of this study, so it cannot drift from what the run actually serves.
    """
    from robot_auto_evolve.agent.motion import make_controller

    profile = context.profile
    lines: list[str] = []

    served = [tool for tool in profile.tools if tool.enabled and tool.service is not None]
    missing = [tool for tool in profile.tools if not (tool.enabled and tool.service is not None)]
    if served:
        lines.append(
            "- Tools actually served on this route (tools.has(...) is True for these): "
            + ", ".join(f"{tool.capability} = {tool.service.identity.model_id}" for tool in sorted(served, key=lambda item: item.capability))
            + "."
        )
    if missing:
        lines.append(
            "- Tools NOT served here -- tools.has(...) is False and calling them raises: "
            + "; ".join(
                f"{tool.capability} ({tool.blocker or 'no service configured'})"
                for tool in sorted(missing, key=lambda item: item.capability)
            )
        )

    cameras = profile.environment.cameras
    with_depth = [item for item in cameras if item.has_depth]
    camera_list = ", ".join(f"{item.name} ({item.width}x{item.height})" for item in cameras)
    if with_depth:
        lines.append(
            f"- Cameras: {camera_list}. 3D sensing is ON for: "
            + ", ".join(item.name for item in with_depth)
            + ". Those cameras carry .depth_m (metres, pixel-aligned with .rgb), .depth_valid, "
            ".intrinsics and .camera_to_world. Use robot_auto_evolve.agent.geometry rather than "
            "doing the projection by hand -- it knows this route's camera convention "
            f"({cameras[0].optical_convention})."
        )
    else:
        lines.append(
            f"- Cameras: {camera_list}. 3D sensing is OFF on this route: every camera's .depth_m, "
            ".intrinsics and .camera_to_world are None, so anything needing metric 3D (including "
            "tools.grasp) cannot work here."
        )

    spec = profile.policy.action_spec
    lines.append(
        "- The action the policy returns, and the action you must return: channels "
        + ", ".join(f"{name} ({semantic})" for name, semantic in zip(spec.channel_names, spec.channel_semantics))
        + f"; coordinate frame {spec.coordinate_frame}; rotation {spec.rotation_representation}; "
        f"gripper convention {spec.gripper_convention}; values are {spec.value_encoding}"
        + (f" with per-channel scale {list(spec.controller_output_scale)}" if spec.controller_output_scale else "")
        + f". Exactly {profile.policy.execution_count} action(s) execute per step and the chunk horizon limit is "
        f"{profile.policy.chunk_horizon}."
    )

    eef_frames = sorted(
        {item.reference_frame for item in profile.environment.robot_state if item.quantity == "end_effector_pose"}
    )
    if eef_frames:
        lines.append(
            "- The end-effector position in observation.proprioception is reported in the "
            f"'{eef_frames[0]}' frame. Points from geometry.pixel_to_world and targets for "
            "agent.motion are in that SAME frame, so they can be compared directly."
        )

    controller = make_controller(spec)
    if controller is None:
        lines.append(
            "- Movement primitives: robot_auto_evolve.agent.motion does NOT support this route's "
            "action layout, so make_controller(spec) returns None here. The frozen policy is the "
            "only way to move this robot."
        )
    else:
        lines.append(
            "- Movement primitives: robot_auto_evolve.agent.motion CAN drive this route "
            f"(layout '{controller.layout}'). make_controller(chunk.spec) gives you move_to / "
            "nudge / hold / set_gripper. You may return one of those instead of the policy's "
            "action on any step -- but keep asking the policy every step anyway (see the rule "
            "above). Destinations must be computed from the observation."
        )
        if not controller.is_delta:
            lines.append(
                "- This route commands an ABSOLUTE end-effector pose, so a movement command has "
                "to put some rotation in the action. Pass the policy's own chunk to "
                "controller.note(chunk) each step and it will keep that rotation, which is both "
                "the reliable option and the sensible one. Do not rely on the rotation the "
                "controller derives from the gripper pose here: this route's profile labels its "
                "rotation channels "
                f"'{spec.rotation_representation}', and on the SimplerEnv X-VLA routes the "
                "controller downstream actually reads them as a rotation vector -- an upstream "
                "mismatch that predates this harness and is faithful to X-VLA's own client."
            )

    if "xvla" in profile.policy.adapter.lower():
        lines.append(
            "- This route's policy is X-VLA, the one policy that reads VLARequest.context: passing "
            "`policy_resample_index=<n>` in the context tuple makes it re-draw its action for the "
            "same observation with a different sampling seed. No scaffold in any previous run has "
            "used it."
        )
    lines.append(
        "- The scored metric is "
        f"`{context.request.scalar_metric}`; see robot_auto_evolve/evaluation/scalars.py for exactly how "
        "it is computed from the per-episode outcomes."
    )
    return "\n".join(lines)


def _revision_backend(context: StudyContext):
    # Revision 8 / D1: the freer coding agent is the ONLY coding backend. Plain `claude`
    # subprocess with a shell that edits scaffold.py in place, prior-isolated -- matching the
    # prior multimodel/roboAutoEvol mechanism. The old OS-sandboxed network-relay backend
    # (ClaudeRevisionBackend) was removed in the s12 restructure.
    loop = context.profile.meta_loop
    return ClaudeFreeRevisionBackend(
        context.claude_executable,
        str(loop.coding_model),
        timeout_s=loop.timeout_s,
        max_turns=loop.max_turns,
        # Run the coding agent at MAX reasoning effort for every route (matches the prior
        # multimodel mechanism). `--effort max` is a valid claude CLI level
        # (low|medium|high|xhigh|max); effort was previously unset, so the CLI used its own
        # default rather than max. Hardcoded uniformly; promote to meta_loop if per-route
        # control is ever needed.
        effort="max",
    )


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
                # Test-only: when a smoke horizon cap is active, tell the simulator
                # workers (started later by SimulatorProcess in scrubbed-env subprocesses)
                # to skip their strict episode.horizon==catalog check, because the plan
                # above has capped each episode's horizon below its protocol value.
                if context.smoke_horizon > 0:
                    os.environ["ROBOT_AE_SMOKE_HORIZON"] = str(context.smoke_horizon)
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
                transfer_plan=transfer_plan,
                transfer_metric=context.request.scalar_metric if transfer_plan is not None else None,
                transfer_evaluator=transfer_evaluator,
                route_notes=_route_notes(context),
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
    meta_backend: str = "claude_free",
    smoke_episodes: int = 0,
    smoke_horizon: int = 0,
    smoke_no_tools: bool = False,
    seed_scaffold: str | Path | None = None,
) -> dict[str, Any]:
    context = load_study_context(
        study_request_path,
        target_candidates=target_candidates,
        finalize=finalize,
        run_transfer=run_transfer,
        meta_backend=meta_backend,
        smoke_episodes=smoke_episodes,
        smoke_horizon=smoke_horizon,
        smoke_no_tools=smoke_no_tools,
        seed_scaffold_override=seed_scaffold,
    )
    return execute_study(
        context,
        target_candidates=target_candidates,
        finalize=finalize,
        run_transfer=run_transfer,
    )


__all__ = [
    "StudyContext",
    "benchmark_scalar_report",
    "execute_study",
    "load_study_context",
    "run_study",
    "study_runtime_paths",
]
