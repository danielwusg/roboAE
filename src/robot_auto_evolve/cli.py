from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from robot_auto_evolve.evaluation import AcceptanceConfig
from robot_auto_evolve.config import Profile
from robot_auto_evolve.process_lifecycle import (
    OwnedProcessRegistry,
    RunInterrupted,
    current_process_registry,
    interruption_handlers,
    process_registry,
)
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.runtime import ProfileServiceRuntime, resolve_profile_launch_paths
from robot_auto_evolve.runtime_paths import assert_clean_import_origin
from robot_auto_evolve.study_runner import run_study

from .evolution import (
    ClaudeRevisionBackend,
    CommandEvaluator,
    EditablePolicy,
    EvolutionDriver,
    FixtureEvaluator,
    FixtureRevisionBackend,
    ProfileEvaluator,
    RoboLab120ProfileEvaluator,
    RelayLimits,
    file_sha256,
    relay_provenance,
    resolve_render_gpu_ids,
    tree_hashes,
    verify_tree_manifest,
    write_tree_manifest,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _evaluator(args: argparse.Namespace) -> Any:
    if args.evaluator == "fixture":
        if args.fixture_evaluations is None:
            raise ValueError("--fixture-evaluations is required")
        return FixtureEvaluator.load(args.fixture_evaluations)
    if args.evaluator_command is None:
        raise ValueError("--evaluator-command is required")
    if args.evaluator_timeout is None:
        raise ValueError("--evaluator-timeout is required")
    return CommandEvaluator(shlex.split(args.evaluator_command), args.evaluator_timeout)


def _backend(args: argparse.Namespace) -> Any:
    if args.meta_backend == "fixture":
        if args.fixture_revisions is None:
            raise ValueError("--fixture-revisions is required")
        return FixtureRevisionBackend.load(args.fixture_revisions)
    if (
        args.claude_executable is None
        or args.claude_isolation_dir is None
        or args.claude_model is None
        or args.claude_credential_dir is None
    ):
        raise ValueError("Claude backend requires executable, isolation, model, and credential directory")
    _require_disjoint(Path(args.claude_credential_dir), Path(args.run_dir), "Claude credential directory", "run")
    return ClaudeRevisionBackend(
        args.claude_executable,
        args.claude_isolation_dir,
        args.claude_model,
        args.claude_timeout,
        args.claude_max_turns,
        credential_dir=args.claude_credential_dir,
        relay_limits=RelayLimits(
            max_requests=args.claude_api_request_budget,
            max_request_bytes=args.claude_api_request_max_bytes,
            max_response_bytes=args.claude_api_response_max_bytes,
            deadline_s=args.claude_timeout,
            provider_timeout_s=min(300.0, args.claude_timeout),
        ),
    )


def _driver(args: argparse.Namespace) -> EvolutionDriver:
    acceptance = AcceptanceConfig(
        bootstrap_resamples=args.bootstrap_resamples,
        confidence_level=args.confidence_level,
        minimum_effect=args.minimum_effect,
        maximum_regression_probability=args.maximum_regression_probability,
        maximum_task_regression=args.maximum_task_regression,
        maximum_task_regression_probability=args.maximum_task_regression_probability,
        max_candidates=args.max_candidates,
        attempt_index=1,
        random_seed=args.acceptance_seed,
    )
    return EvolutionDriver(
        seed_scaffold=args.seed_scaffold,
        run_dir=args.run_dir,
        evaluator=_evaluator(args),
        revision_backend=_backend(args),
        acceptance=acceptance,
        frozen_paths=tuple(args.frozen_path),
    )


def _run_evolve(args: argparse.Namespace) -> int:
    driver = _driver(args)
    state = driver.advance_to(args.target_candidates, finalize=args.finalize)
    if args.run_transfer:
        if not args.finalize:
            raise ValueError("--run-transfer requires --finalize")
        transfer = driver.run_sealed_transfer()
        output = {"state": state, "transfer": transfer.to_mapping()}
    else:
        output = {"state": state}
    print(json.dumps(output, sort_keys=True))
    return 0


def _run_study(args: argparse.Namespace) -> int:
    result = run_study(
        args.study_request,
        target_candidates=args.target_candidates,
        finalize=args.finalize,
        run_transfer=args.run_transfer,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_claude_launch(args: argparse.Namespace) -> int:
    backend = ClaudeRevisionBackend(args.executable, args.isolation_dir, args.model, args.timeout, 1)
    result = backend.launch_only(args.candidate_dir, args.log_dir, args.startup_wait)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


def _run_claude_relay_probe(args: argparse.Namespace) -> int:
    backend = ClaudeRevisionBackend(args.executable, args.isolation_dir, args.model, args.timeout, 1)
    result = backend.offline_probe(args.candidate_dir, args.log_dir, args.timeout)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


def _run_validate_scaffold(args: argparse.Namespace) -> int:
    hashes = EditablePolicy().validate_tree(args.scaffold_dir)
    print(json.dumps(hashes, sort_keys=True))
    return 0


def _run_manifest(args: argparse.Namespace) -> int:
    manifest = write_tree_manifest(Path(args.run_dir))
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _run_verify_manifest(args: argparse.Namespace) -> int:
    manifest = verify_tree_manifest(Path(args.run_dir))
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _profile_backend(args: argparse.Namespace, profile: Profile) -> Any:
    if args.meta_backend != profile.meta_loop.coding_backend:
        raise ValueError("--meta-backend differs from the profile coding backend")
    if args.meta_backend == "fixture":
        if args.fixture_revisions is None:
            raise ValueError("--fixture-revisions is required for fixture meta backend")
        return FixtureRevisionBackend.load(args.fixture_revisions)
    if args.claude_executable is None or args.claude_isolation_dir is None or args.claude_credential_dir is None:
        raise ValueError("Claude backend requires executable, isolation, and credential directory")
    if profile.meta_loop.coding_model is None:
        raise ValueError("Claude profile has no coding model")
    return ClaudeRevisionBackend(
        args.claude_executable,
        args.claude_isolation_dir,
        profile.meta_loop.coding_model,
        profile.meta_loop.timeout_s,
        profile.meta_loop.max_turns,
        credential_dir=args.claude_credential_dir,
        relay_limits=RelayLimits(
            max_requests=profile.meta_loop.api_request_budget,
            max_request_bytes=profile.meta_loop.api_request_max_bytes,
            max_response_bytes=profile.meta_loop.api_response_max_bytes,
            deadline_s=profile.meta_loop.timeout_s,
            provider_timeout_s=min(300.0, profile.meta_loop.timeout_s),
        ),
    )


def _record_profile_invocation(
    path: Path,
    *,
    args: argparse.Namespace,
    profile: Profile,
    profile_path: Path,
    seed_scaffold: Path,
    environment_root: Path,
    launch_paths: dict[str, Path],
    render_gpu_ids: tuple[int, ...] | None = None,
) -> None:
    revision: dict[str, Any] = {
        "backend": args.meta_backend,
        "max_turns": profile.meta_loop.max_turns,
        "timeout_s": profile.meta_loop.timeout_s,
    }
    if args.meta_backend == "fixture":
        fixture = Path(args.fixture_revisions).resolve()
        revision["fixture_revisions"] = {"path": str(fixture), "sha256": file_sha256(fixture)}
    else:
        executable = Path(args.claude_executable).resolve()
        revision["claude_executable"] = {"path": str(executable), "sha256": file_sha256(executable)}
        revision["claude_isolation_dir"] = str(Path(args.claude_isolation_dir).resolve())
        limits = RelayLimits(
            max_requests=profile.meta_loop.api_request_budget,
            max_request_bytes=profile.meta_loop.api_request_max_bytes,
            max_response_bytes=profile.meta_loop.api_response_max_bytes,
            deadline_s=profile.meta_loop.timeout_s,
            provider_timeout_s=min(300.0, profile.meta_loop.timeout_s),
        )
        revision["relay"] = relay_provenance(str(profile.meta_loop.coding_model), limits)
    value = {
        "schema_version": 2,
        "command": "run-profile",
        "profile": {"path": str(profile_path), "sha256": profile.resolved_hash()},
        "seed_scaffold": {"path": str(seed_scaffold), "files": tree_hashes(seed_scaffold)},
        "environment_root": str(environment_root),
        "environments": _environment_provenance(launch_paths),
        "launch_paths": {name: str(item) for name, item in sorted(launch_paths.items())},
        "candidate_budget": profile.meta_loop.candidate_budget,
        "render_gpu_ids": None if render_gpu_ids is None else list(render_gpu_ids),
        "revision": revision,
    }
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("effective invocation differs from the existing run")
        return
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o444)


def _environment_provenance(launch_paths: dict[str, Path]) -> dict[str, Any]:
    python_roles = {
        name: Path(path).absolute()
        for name, path in launch_paths.items()
        if name == "agent_python" or name == "simulator_python" or name.startswith("policy_python:") or name.startswith("tool_python:")
    }
    prefixes = sorted({path.parent.parent for path in python_roles.values()}, key=str)
    identifiers = {prefix: f"environment_{index:03d}" for index, prefix in enumerate(prefixes, start=1)}
    records = []
    script = (
        "import hashlib,importlib.metadata,json,sys\n"
        "rows=[]\n"
        "for distribution in importlib.metadata.distributions():\n"
        " name=distribution.metadata.get('Name')\n"
        " if not name: continue\n"
        " direct=distribution.read_text('direct_url.json')\n"
        " rows.append({'name':name,'version':distribution.version,'direct_url_sha256':None if direct is None else hashlib.sha256(direct.encode()).hexdigest()})\n"
        "rows.sort(key=lambda item:(item['name'].lower(),item['version'],item['direct_url_sha256'] or ''))\n"
        "print(json.dumps({'python_version':sys.version,'packages':rows},sort_keys=True,separators=(',',':')))\n"
    )
    for prefix in prefixes:
        python = prefix / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"environment Python is missing: {python}")
        result = subprocess.run(
            [str(python), "-I", "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": os.pathsep.join((str(python.parent), "/usr/bin", "/bin")),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"cannot inspect environment {prefix}: {result.stderr.strip()}")
        installed = json.loads(result.stdout)
        if not isinstance(installed, dict) or set(installed) != {"python_version", "packages"}:
            raise RuntimeError(f"environment inventory is invalid: {prefix}")
        conda_meta = prefix / "conda-meta"
        conda_records = []
        if conda_meta.is_dir():
            for item in sorted(conda_meta.glob("*.json"), key=lambda value: value.name):
                conda_records.append({"name": item.name, "sha256": file_sha256(item)})
        records.append(
            {
                "id": identifiers[prefix],
                "prefix": str(prefix),
                "python": {"path": str(python), "sha256": file_sha256(python)},
                "python_version": installed["python_version"],
                "packages": installed["packages"],
                "conda_metadata": conda_records,
                "inventory_sha256": hashlib.sha256(result.stdout.strip().encode("utf-8")).hexdigest(),
            }
        )
    return {
        "roles": {name: identifiers[path.parent.parent] for name, path in sorted(python_roles.items())},
        "records": records,
    }


def _record_profile_progression(root: Path, args: argparse.Namespace) -> int:
    directory = root / "invocations"
    directory.mkdir(parents=True, exist_ok=True)
    indices = [int(path.stem) for path in directory.glob("*.json") if len(path.stem) == 6 and path.stem.isdigit()]
    index = max(indices, default=0) + 1
    while True:
        path = directory / f"{index:06d}.json"
        try:
            descriptor = path.open("x", encoding="utf-8")
        except FileExistsError:
            index += 1
            continue
        with descriptor:
            json.dump(
                {
                    "schema_version": 1,
                    "started_ns": time.time_ns(),
                    "target_candidates": args.target_candidates,
                    "finalize": bool(args.finalize),
                    "run_transfer": bool(args.run_transfer),
                },
                descriptor,
                sort_keys=True,
                indent=2,
            )
            descriptor.write("\n")
        path.chmod(0o444)
        return index


def _record_profile_completion(root: Path, index: int, status: str, error: BaseException | None = None) -> None:
    path = root / "invocations" / f"{index:06d}.finished.json"
    with path.open("x", encoding="utf-8") as descriptor:
        json.dump(
            {
                "schema_version": 1,
                "finished_ns": time.time_ns(),
                "status": status,
                "error_type": None if error is None else type(error).__name__,
            },
            descriptor,
            sort_keys=True,
            indent=2,
        )
        descriptor.write("\n")
    path.chmod(0o444)


def _record_interruption(path: Path, interruption: RunInterrupted) -> None:
    registry = current_process_registry()
    cleanup_path = path / "owned_process_cleanup.json"
    if registry is not None:
        registry.write(cleanup_path)
    value = {
        "schema_version": 1,
        "recorded_ns": time.time_ns(),
        "signal_number": interruption.signum,
        "signal_name": interruption.signal_name,
        "exit_status": interruption.exit_status,
        "owned_process_cleanup": cleanup_path.name if cleanup_path.is_file() else None,
    }
    (path / "interruption.json").write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _capture_system(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    commands = {
        "hostname.txt": ["hostname"],
        "gpus.txt": ["nvidia-smi", "-L"],
        "gpu_state.csv": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ],
        "processes.txt": [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv",
        ],
        "host_processes.txt": [
            "ps",
            "-eo",
            "pid,ppid,pgid,sid,stat,etimes,cmd",
            "--sort=pid",
        ],
        "nproc.txt": ["nproc"],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0 if command[0] == "nvidia-smi" else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"system capture exceeded five seconds: {' '.join(command)}") from exc
        if command[0] == "nvidia-smi" and result.returncode != 0:
            raise RuntimeError(f"system capture failed: {' '.join(command)}: {result.stderr.strip()}")
        (path / name).write_text(result.stdout + result.stderr, encoding="utf-8")


def _require_disjoint(first: Path, second: Path, first_name: str, second_name: str) -> None:
    first = Path(first).resolve()
    second = Path(second).resolve()
    if first == second or first in second.parents or second in first.parents:
        raise ValueError(f"{first_name} must be disjoint from {second_name}")


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


def _render_gpu_ids(profile: Profile, value: str | None) -> tuple[int, ...]:
    if value is None:
        return resolve_render_gpu_ids(profile, None)
    parts = value.split(",")
    if any(not part.isdigit() or (len(part) > 1 and part.startswith("0")) for part in parts):
        raise ValueError("--render-gpu-ids must be comma-separated nonnegative integers")
    try:
        return resolve_render_gpu_ids(profile, tuple(int(part) for part in parts))
    except StrictSchemaError as exc:
        raise ValueError(str(exc)) from exc


def _run_profile(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    assert_clean_import_origin(project_root)
    profile_path = Path(args.profile).resolve()
    profile = Profile.load(profile_path, project_root=project_root)
    plan = profile.episode_plan.load(project_root)
    render_gpu_ids = (
        _render_gpu_ids(profile, getattr(args, "render_gpu_ids", None))
        if isinstance(profile, Profile)
        else None
    )
    if args.target_candidates < 0 or args.target_candidates > profile.meta_loop.candidate_budget:
        raise ValueError("--target-candidates exceeds the profile candidate budget")
    if args.run_transfer and not args.finalize:
        raise ValueError("--run-transfer requires --finalize")
    experiment_root = Path(args.run_dir).resolve()
    runs_root = project_root / "runs"
    if not experiment_root.is_relative_to(runs_root) or len(experiment_root.relative_to(runs_root).parts) != 1:
        raise ValueError("--run-dir must be one direct child of the clean runs directory")
    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_root / "run_manifest.json"
    if manifest_path.exists():
        verify_tree_manifest(experiment_root)
        raise RuntimeError("run is complete and sealed")
    if args.meta_backend == "claude":
        if args.claude_executable is None or args.claude_isolation_dir is None or args.claude_credential_dir is None:
            raise ValueError("Claude backend requires executable, isolation, and credential directory")
        isolation_dir = Path(args.claude_isolation_dir).resolve()
        credential_dir = Path(args.claude_credential_dir).resolve()
        _require_disjoint(isolation_dir, experiment_root, "Claude isolation directory", "the entire run")
        _require_disjoint(credential_dir, project_root, "Claude credential directory", "the project")
        _require_disjoint(credential_dir, isolation_dir, "Claude credential directory", "Claude isolation")
        isolation_dir.mkdir(parents=True, exist_ok=True)
        isolation_dir.chmod(0o700)
    revision_backend = _profile_backend(args, profile)
    snapshot = experiment_root / "profile.json"
    resolved_profile = json.dumps(profile.to_mapping(), sort_keys=True, indent=2) + "\n"
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != resolved_profile:
        raise RuntimeError("run profile snapshot differs from the requested profile")
    if not snapshot.exists():
        snapshot.write_text(resolved_profile, encoding="utf-8")
        snapshot.chmod(0o444)
    environment_root = Path(args.environment_root).resolve()
    launch_paths = resolve_profile_launch_paths(profile, project_root, environment_root)
    _record_profile_invocation(
        experiment_root / "effective_invocation.json",
        args=args,
        profile=profile,
        profile_path=profile_path,
        seed_scaffold=Path(args.seed_scaffold).resolve(),
        environment_root=environment_root,
        launch_paths=launch_paths,
        render_gpu_ids=render_gpu_ids,
    )
    invocation_index = _record_profile_progression(experiment_root, args)
    invocation_root = experiment_root / "invocation_artifacts" / f"{invocation_index:06d}"
    run_runtime_root = experiment_root / "runtime"
    invocation_runtime_root = run_runtime_root / f"invocation-{invocation_index:06d}"
    failure: BaseException | None = None
    failure_traceback = None
    evidence_failure: BaseException | None = None
    state: Any = None
    transfer: Any = None
    try:
        invocation_root.mkdir(parents=True, exist_ok=False)
        _capture_system(invocation_root / "system_before")
        simulator_source_key = _simulator_source_key(profile.environment.suite)
        runtime = ProfileServiceRuntime(
            profile,
            project_root=project_root,
            environment_root=environment_root,
            log_root=invocation_runtime_root / "services",
        )
        with runtime as clients:
            _capture_system(invocation_root / "system_ready")
            evaluator = (
                RoboLab120ProfileEvaluator(
                    profile,
                    plan,
                    agent_python=Path(args.agent_python or environment_root / "agent" / "bin" / "python"),
                    simulator_python=Path(args.simulator_python or launch_paths["simulator_python"]),
                    simulator_source=launch_paths[simulator_source_key],
                    runtime_root=invocation_runtime_root / "robolab_simulators",
                    live_clients=clients,
                )
                if profile.environment.suite == "robolab120_droid_jointpos"
                else ProfileEvaluator(
                    profile,
                    plan,
                    agent_python=Path(args.agent_python or environment_root / "agent" / "bin" / "python"),
                    simulator_python=Path(args.simulator_python or launch_paths["simulator_python"]),
                    simulator_source=launch_paths[simulator_source_key],
                    live_clients=clients,
                    render_gpu_ids=render_gpu_ids,
                    runtime_root=invocation_runtime_root,
                )
            )
            preflight = getattr(evaluator, "preflight_rendering", None)
            if callable(preflight):
                report = preflight(invocation_runtime_root / "render_preflight")
                _write_json(invocation_root / "render_preflight.json", report)
            acceptance = AcceptanceConfig(
                bootstrap_resamples=profile.acceptance.bootstrap_resamples,
                confidence_level=profile.acceptance.confidence_level,
                minimum_effect=profile.acceptance.minimum_effect,
                maximum_regression_probability=profile.acceptance.maximum_regression_probability,
                maximum_task_regression=profile.acceptance.maximum_task_regression,
                maximum_task_regression_probability=profile.acceptance.maximum_task_regression_probability,
                max_candidates=profile.acceptance.max_candidates,
                random_seed=profile.acceptance.random_seed,
            )
            driver = EvolutionDriver(
                seed_scaffold=Path(args.seed_scaffold),
                run_dir=experiment_root / "evolution",
                evaluator=evaluator,
                revision_backend=revision_backend,
                acceptance=acceptance,
                frozen_paths=(
                    profile_path,
                    project_root / profile.episode_plan.path,
                    project_root / "locks" / "runtime_artifacts.json",
                    project_root / "runtime_paths.json",
                    project_root / "src" / "robot_auto_evolve",
                ),
            )
            try:
                state = driver.advance_to(args.target_candidates, finalize=args.finalize)
                if args.run_transfer:
                    transfer = driver.run_sealed_transfer().to_mapping()
            finally:
                close = getattr(evaluator, "close", None)
                if callable(close):
                    close(force=sys.exc_info()[0] is not None)
    except BaseException:
        failure = sys.exc_info()[1]
        failure_traceback = sys.exc_info()[2]
    finally:
        if invocation_root.is_dir():
            try:
                _capture_system(invocation_root / "system_after_cleanup")
            except BaseException as exc:
                evidence_failure = exc
            if isinstance(failure, RunInterrupted):
                try:
                    _record_interruption(invocation_root, failure)
                except BaseException as exc:
                    if evidence_failure is None:
                        evidence_failure = exc
        recorded_error = failure if failure is not None else evidence_failure
        status = "interrupted" if isinstance(failure, RunInterrupted) else "failed" if recorded_error is not None else "complete"
        _record_profile_completion(experiment_root, invocation_index, status, recorded_error)
    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    if evidence_failure is not None:
        raise evidence_failure
    print(json.dumps({"state": state, "transfer": transfer}, sort_keys=True))
    return 0


def _add_evolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed-scaffold", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-path", type=Path, action="append", required=True)
    parser.add_argument("--target-candidates", type=int, required=True)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--evaluator", choices=("fixture", "command"), required=True)
    parser.add_argument("--fixture-evaluations", type=Path)
    parser.add_argument("--evaluator-command")
    parser.add_argument("--evaluator-timeout", type=float)
    parser.add_argument("--meta-backend", choices=("fixture", "claude"), required=True)
    parser.add_argument("--fixture-revisions", type=Path)
    parser.add_argument("--claude-executable", type=Path)
    parser.add_argument("--claude-isolation-dir", type=Path)
    parser.add_argument("--claude-model")
    parser.add_argument("--claude-credential-dir", type=Path)
    parser.add_argument("--claude-timeout", type=float, default=900.0)
    parser.add_argument("--claude-max-turns", type=int, default=30)
    parser.add_argument("--claude-api-request-budget", type=int, default=64)
    parser.add_argument("--claude-api-request-max-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--claude-api-response-max-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-effect", type=float, default=0.0)
    parser.add_argument("--maximum-regression-probability", type=float, default=0.05)
    parser.add_argument("--maximum-task-regression", type=float, default=0.0)
    parser.add_argument("--maximum-task-regression-probability", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--acceptance-seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot-auto-evolve")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evolve = subparsers.add_parser("evolve")
    _add_evolution_arguments(evolve)
    evolve.set_defaults(handler=_run_evolve)
    study = subparsers.add_parser("run-study")
    study.add_argument("--study-request", type=Path, required=True)
    study.add_argument("--target-candidates", type=int, required=True)
    study.add_argument("--finalize", action="store_true")
    study.add_argument("--run-transfer", action="store_true")
    study.set_defaults(handler=_run_study)
    launch = subparsers.add_parser("claude-launch-only")
    launch.add_argument("--executable", type=Path, required=True)
    launch.add_argument("--isolation-dir", type=Path, required=True)
    launch.add_argument("--candidate-dir", type=Path, required=True)
    launch.add_argument("--log-dir", type=Path, required=True)
    launch.add_argument("--model", required=True)
    launch.add_argument("--timeout", type=float, default=30.0)
    launch.add_argument("--startup-wait", type=float, default=0.1)
    launch.set_defaults(handler=_run_claude_launch)
    relay_probe = subparsers.add_parser("claude-relay-probe")
    relay_probe.add_argument("--executable", type=Path, required=True)
    relay_probe.add_argument("--isolation-dir", type=Path, required=True)
    relay_probe.add_argument("--candidate-dir", type=Path, required=True)
    relay_probe.add_argument("--log-dir", type=Path, required=True)
    relay_probe.add_argument("--model", required=True)
    relay_probe.add_argument("--timeout", type=float, default=30.0)
    relay_probe.set_defaults(handler=_run_claude_relay_probe)
    validate = subparsers.add_parser("validate-scaffold")
    validate.add_argument("scaffold_dir", type=Path)
    validate.set_defaults(handler=_run_validate_scaffold)
    manifest = subparsers.add_parser("manifest-run")
    manifest.add_argument("run_dir", type=Path)
    manifest.set_defaults(handler=_run_manifest)
    verify_manifest = subparsers.add_parser("verify-run-manifest")
    verify_manifest.add_argument("run_dir", type=Path)
    verify_manifest.set_defaults(handler=_run_verify_manifest)
    profile = subparsers.add_parser("run-profile")
    profile.add_argument("--profile", type=Path, required=True)
    profile.add_argument("--project-root", type=Path, required=True)
    profile.add_argument("--run-dir", type=Path, required=True)
    profile.add_argument("--seed-scaffold", type=Path, required=True)
    profile.add_argument("--environment-root", type=Path, required=True)
    profile.add_argument("--agent-python", type=Path)
    profile.add_argument("--simulator-python", type=Path)
    profile.add_argument("--render-gpu-ids")
    profile.add_argument("--target-candidates", type=int, required=True)
    profile.add_argument("--finalize", action="store_true")
    profile.add_argument("--run-transfer", action="store_true")
    profile.add_argument("--meta-backend", choices=("fixture", "claude"), required=True)
    profile.add_argument("--fixture-revisions", type=Path)
    profile.add_argument("--claude-executable", type=Path)
    profile.add_argument("--claude-isolation-dir", type=Path)
    profile.add_argument(
        "--claude-credential-dir",
        type=Path,
        help="private external directory containing a 0600 oauth_token file",
    )
    profile.set_defaults(handler=_run_profile)
    return parser


def main(argv: list[str] | None = None) -> int:
    registry = OwnedProcessRegistry()
    with process_registry(registry), interruption_handlers(registry):
        try:
            try:
                parser = build_parser()
                args = parser.parse_args(argv)
                status = int(args.handler(args))
            finally:
                registry.terminate_all()
        except RunInterrupted as exc:
            print(f"interrupted: {exc.signal_name}", file=sys.stderr)
            return exc.exit_status
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return status
