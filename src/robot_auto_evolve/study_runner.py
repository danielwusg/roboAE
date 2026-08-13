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
    RuntimeConfig,
    StudyRequest,
    materialize_runtime_profile,
    materialize_runtime_profiles,
)
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest
from robot_auto_evolve.runtime import (
    ProfileServiceRuntime,
    ScaffoldRuntimeCoordinator,
    renders_with_egl,
    resolve_profile_launch_paths,
)
from robot_auto_evolve.runtime_paths import RuntimePaths, assert_clean_import_origin, project_root_from_package


@dataclass(frozen=True)
class StudyContext:
    request: StudyRequest
    runtime_config: RuntimeConfig
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
    fairness_guard: bool
    scaffold_memory: bool


MEMORY_NOTES = (
    "## A place to leave notes for later episodes\n"
    "On this run your scaffold can keep something between episodes. `tools.has_memory()` says whether it is "
    "switched on; check it before you use it, because it is off on most runs and calling into it when it is off "
    "raises `ToolUnavailableError` like any other unavailable tool.\n"
    "\n"
    "- `tools.remember(key, value)` stores one value under one name. `key` is a string. `value` may be any mixture "
    "of numbers, strings, true/false, null, lists and dictionaries with string names. It may not hold pictures, "
    "arrays or objects. Storing the same name again replaces what was there.\n"
    "- `tools.recall(key)` gives back what was stored under that name, or `None` if nothing was.\n"
    "- `tools.memory_keys()` lists the names in use. `tools.forget(key)` removes one.\n"
    "\n"
    "Every episode of one scoring run shares the same store, and episodes run several at a time, so an episode may "
    "read something an earlier episode wrote. The store starts empty every time your scaffold is scored, so the "
    "first episodes fill it and the later ones can use it. Nothing carries over to the next scoring run, and "
    "nothing carries over to a different setup.\n"
    "\n"
    "What to keep there, in what shape, when to read it, what to do with what you read, and whether to use it at "
    "all are yours to decide. The same rules as everywhere else apply to what you store: it must be worked out "
    "from what the robot could see, and it must not be a place to write down an answer for a particular task. A "
    "measured offset from something the camera found this episode is fine. A position in the room, or anything "
    "keyed to one task name, is not.\n"
)


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
    fairness_guard: bool = False,
    scaffold_memory: bool = False,
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
    runtime_config = RuntimeConfig.load(run_root)
    runtime_profile_path = materialize_runtime_profile(request, runtime_config, run_root)
    runtime_profile_paths = materialize_runtime_profiles(request, runtime_config, run_root)
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
        runtime_config=runtime_config,
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
        fairness_guard=fairness_guard,
        scaffold_memory=bool(scaffold_memory),
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
    coordinator: ScaffoldRuntimeCoordinator,
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
        coordinator=coordinator,
        task_suites=task_suites,
        artifact_metric_function=lambda manifests, root: _canonical_metric_report(
            manifests,
            root,
            context.request.scalar_metric,
        ),
        scaffold_memory=context.scaffold_memory,
        reuse_agent=context.runtime_config.reuse_agent,
        reuse_sim=context.runtime_config.reuse_sim
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
    from robot_auto_evolve.agent.motion import make_controller

    profile = context.profile
    lines: list[str] = []

    served = [tool for tool in profile.tools if tool.enabled and tool.service is not None]
    missing = [tool for tool in profile.tools if not (tool.enabled and tool.service is not None)]
    if served:
        lines.append(
            "- Models available on this setup: "
            + ", ".join(
                f"{tool.capability} = {tool.service.identity.model_id}"
                for tool in sorted(served, key=lambda item: item.capability)
            )
            + ". One is loaded only if `SCAFFOLD_CONFIG` lists it and your file mentions it, so name in a "
            "`tools.<name>(...)` call, a `tools.has(\"<name>\")` check or one of its request types every model "
            "you intend to use. Check with `tools.has(...)` before calling."
        )
    if missing:
        lines.append(
            "- Never available here (tools.has(...) is False even if you call it, and calling one raises): "
            + ", ".join(tool.capability for tool in sorted(missing, key=lambda item: item.capability))
            + "."
        )
    if any(tool.capability == "grasp" for tool in served):
        lines.append(
            "- The grasp model: give it a "
            "picture, that picture's depth, the camera's lens matrix and pose, the camera's optical convention, and "
            "a mask of the object, and it returns ranked six-degree-of-freedom grasp poses in the same coordinates "
            "the gripper is reported in -- `tools.grasp(GraspRequest(rgb, depth_m, intrinsics, camera_to_world, "
            "optical_convention, mask))` -> `GraspResult(.candidates)`, each `GraspCandidate(.pose_world` 4x4`, "
            ".score, .width_m)`. Get the mask from the segmenter. It needs depth, and it was trained for this "
            "robot's gripper. A pose is where and how to hold the object; moving there is still your job, and on a "
            "setup whose action is a change from the current pose the ready-made movement commands will not turn "
            "the wrist, so you would have to write the rotation numbers yourself."
        )

    cameras = profile.environment.cameras
    with_depth = [item for item in cameras if item.has_depth]
    camera_list = ", ".join(f"{item.name} ({item.width}x{item.height})" for item in cameras)
    if with_depth:
        upright = "the BOTTOM" if cameras[0].optical_convention == "opengl_rub" else "the TOP"
        lines.append(
            f"- Cameras: {camera_list}. Depth is on for: "
            + ", ".join(item.name for item in with_depth)
            + ". Those cameras carry .depth_m (metres, one value per colour pixel), .depth_valid, .intrinsics and "
            ".camera_to_world. Use robot_auto_evolve.agent.geometry rather than projecting by hand -- here row 0 of "
            f"the stored picture is {upright} of the scene, and the module already accounts for that."
        )
    else:
        lines.append(
            f"- Cameras: {camera_list}. There is no depth here: every camera's .depth_m, .intrinsics and "
            ".camera_to_world are None, so anything needing a distance in metres cannot work."
        )

    lines.append(f"- The robot is a {profile.environment.embodiment}.")

    spec = profile.policy.action_spec
    absolute = "absolute" in spec.channel_semantics
    lines.append(
        "- The action the policy returns, and the action you must return: "
        + f"{len(spec.channel_names)} numbers -- "
        + ", ".join(spec.channel_names)
        + ". "
        + (
            "They are a target pose, not a change from where the gripper is now. "
            if absolute
            else "The movement numbers are a change from where the gripper is now, not a destination. "
        )
        + f"Rotation is {spec.rotation_representation.replace('_', '-')}. "
        + (
            "Every value must be between -1 and +1, and one unit means the matching per-channel scale, here "
            f"{list(spec.controller_output_scale)}, so the first number at 1.0 commands "
            f"{spec.controller_output_scale[0]} in its own unit. "
            if spec.value_encoding == "normalized_controller"
            else "The values are physical, in the units the channels name, and are not clipped to a range. "
        )
        + (
            "For the gripper, +1 is closed and -1 is open. "
            if spec.gripper_convention in ("closed_positive", "binary_closed_one")
            else "For the gripper, +1 is open and -1 is closed. "
            if spec.gripper_convention in ("open_positive", "binary_open_one")
            else ""
        )
        + f"Exactly {profile.policy.execution_count} action(s) execute per step and a chunk may hold at most "
        f"{profile.policy.chunk_horizon}."
    )

    eef_frames = sorted(
        {item.reference_frame for item in profile.environment.robot_state if item.quantity == "end_effector_pose"}
    )
    if eef_frames:
        lines.append(
            "- The gripper position in observation.proprioception is given in the "
            + (
                "same coordinates as the points geometry.pixel_to_world returns and the targets agent.motion takes, "
                "so you can compare them directly."
                if with_depth
                else f"'{eef_frames[0]}' frame, which is also the frame agent.motion takes targets in. There is no "
                "depth here, so a target has to come from proprioception or from your own reasoning about the "
                "scene, not from geometry.pixel_to_world, which returns None."
            )
        )

    controller = make_controller(spec)
    if controller is None:
        lines.append(
            "- Movement commands: robot_auto_evolve.agent.motion does NOT support this action layout, so "
            "make_controller(spec) returns None here. Either use the policy's action or compute every number "
            "yourself."
        )
    else:
        lines.append(
            "- Movement commands: robot_auto_evolve.agent.motion can drive this setup. Its layout is named "
            f"'{controller.layout}'; grep that name in agent/motion.py to see exactly how a move is computed. "
            "make_controller(chunk.spec) gives you move_to, nudge, hold and set_gripper. You may return one of "
            "those instead of the policy's action on any step, and work every destination out from the observation."
        )
        if not controller.is_delta:
            note = (
                "- Here the action IS a target pose, not a change, so a movement command has to supply a rotation "
                "as well. Call controller.note(chunk) with the policy's own chunk each step and it will reuse that "
                "rotation."
            )
            if profile.environment.suite.startswith("simpler_"):
                note += (
                    " Do not rely on the rotation the controller works out from the gripper pose here: this setup "
                    f"labels its rotation numbers '{spec.rotation_representation}', but the controller underneath "
                    "reads those three numbers as a rotation VECTOR."
                )
            lines.append(note)

    if "xvla" in profile.policy.adapter.lower():
        lines.append(
            "- This setup's policy reads VLARequest.context: passing "
            "`policy_resample_index=<n>` in the context tuple makes it draw a different action for the same picture."
        )

    render = context.runtime_config.render_gpu_ids or context.runtime_config.gpu_ids
    python = context.runtime_paths.environment_root / "core" / "bin" / "python"
    command = (
        f"PYTHONPATH={context.project_root}/src {python} -m robot_auto_evolve.replay "
        f"--episode <an episode folder> --out ../agent_workspace/replay --render-gpu {render[-1]} "
        "--steps 40 --depth"
    )
    lines.append(
        "- To re-run an episode in the simulator yourself, this is the command for this setup. Fill in the "
        "episode folder from `../public_input.json`:\n\n"
        f"  ```\n  {command}\n  ```\n\n"
        "  It writes one PNG per camera per step, optionally the depth as `.npy`, and a `replay.json` saying "
        "whether the episode succeeded. `--steps` stops early; leave it out for the whole episode. Point "
        "`--actions` at a different `trace.jsonl` to run a DIFFERENT action sequence in the same scene, which "
        "is how you check what would have happened if the robot had done something else. Add `--help` for the "
        "rest. Starting the simulator takes about half a minute, and a hundred steps takes about as long again."
    )
    lines.append(
        "- How the per-episode outcomes become this setup's single number is in "
        f"robot_auto_evolve/evaluation/scalars.py, under `{context.request.scalar_metric}`. Read it: it decides "
        "whether a gain on one task can pay for a loss on another."
    )
    return "\n\n".join(lines)


def _revision_backend(context: StudyContext):
    loop = context.profile.meta_loop
    return ClaudeFreeRevisionBackend(
        context.claude_executable,
        str(loop.coding_model),
        timeout_s=loop.timeout_s,
        max_turns=loop.max_turns,
        effort="max",
        fairness_guard=context.fairness_guard,
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
        coordinator = ScaffoldRuntimeCoordinator(
            runtime,
            gpu_ids=context.runtime_config.gpu_ids,
            render_gpu_ids_override=context.runtime_config.render_gpu_ids,
            workers_per_gpu=context.runtime_config.workers_per_gpu,
            workers_per_gpu_with_language=context.runtime_config.workers_per_gpu_with_language,
            policies_per_gpu=context.runtime_config.policies_per_gpu,
            egl=renders_with_egl(context.profile.environment.suite),
        )
        with runtime:
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
                if context.smoke_horizon > 0:
                    os.environ["ROBOT_AE_SMOKE_HORIZON"] = str(context.smoke_horizon)
            evolve_evaluator = _canonical_evaluator(
                context,
                evolve_plan,
                coordinator,
                scratch / "evaluators" / "evolve",
            )
            transfer_evaluator = (
                None
                if transfer_plan is None
                else _canonical_evaluator(
                    context,
                    transfer_plan,
                    coordinator,
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
                transfer_split="episode" if context.request.mode == "seed_transfer" else "task",
                memory_notes=MEMORY_NOTES if context.scaffold_memory else "",
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
    fairness_guard: bool = False,
    scaffold_memory: bool = False,
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
        fairness_guard=fairness_guard,
        scaffold_memory=scaffold_memory,
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
