from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from robot_auto_evolve.agent import AgentEvent
from robot_auto_evolve.benchmarks.openvla_simpler_worker import is_openvla_simpler_adapter
from robot_auto_evolve.benchmarks.render_integrity import validate_mujoco_rgb
from robot_auto_evolve.benchmarks.robocasa365 import validate_robocasa365_rgb
from robot_auto_evolve.benchmarks.simpler_worker import validate_simpler_rgb
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation import EpisodeOutcome
from robot_auto_evolve.evolution.evidence import PublicStepEvidence
from robot_auto_evolve.protocol import StrictSchemaError, decode_message
from robot_auto_evolve.provenance import ArtifactRun, EpisodeManifest
from robot_auto_evolve.services import ServiceIdentity


class SmokeIntegrityError(RuntimeError):
    pass


_SAM_FATAL_PATTERNS = (
    ("missing_cuda_header", re.compile(r"(?:fatal\s+error:\s*)?cuda\.h:\s*no\s+such\s+file", re.IGNORECASE)),
    ("skipped_postprocessing", re.compile(r"skipping\s+(?:the\s+)?post[- ]processing", re.IGNORECASE)),
    ("traceback", re.compile(r"traceback\s*\(most\s+recent\s+call\s+last\)", re.IGNORECASE)),
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_smoke_integrity_report(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_json(Path(path), value)


def _strict_json(path: Path) -> Any:
    def object_pairs(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise SmokeIntegrityError(f"invalid or duplicate JSON key in {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SmokeIntegrityError(f"invalid JSON constant {value} in {path}")
            ),
        )
    except SmokeIntegrityError:
        raise
    except Exception as exc:
        raise SmokeIntegrityError(f"failed to read JSON {path}: {exc}") from exc


def _mapping(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SmokeIntegrityError(f"{path}: invalid fields")
    return value


def verify_sam_service_logs(log_paths: Iterable[str | Path]) -> dict[str, Any]:
    paths = tuple(sorted((Path(path).resolve() for path in log_paths), key=str))
    if not paths:
        raise SmokeIntegrityError("SAM log guard requires at least one retained service log")
    if len(set(paths)) != len(paths):
        raise SmokeIntegrityError("SAM log guard received duplicate paths")
    records = []
    violations = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise SmokeIntegrityError(f"SAM retained log is not a regular file: {path}")
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="replace")
        records.append(
            {
                "path": str(path),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        for line_number, line in enumerate(text.splitlines(), start=1):
            for code, pattern in _SAM_FATAL_PATTERNS:
                if pattern.search(line):
                    violations.append(
                        {
                            "code": code,
                            "path": str(path),
                            "line": line_number,
                            "excerpt": line[:500],
                        }
                    )
    if violations:
        details = "; ".join(
            f"{item['code']} at {item['path']}:{item['line']}" for item in violations
        )
        raise SmokeIntegrityError(f"SAM retained service log failed: {details}")
    return {"schema_version": 1, "state": "complete", "logs": records, "violations": []}


_OPERATIONS = {
    "language": "generate",
    "vision": "describe",
    "detection": "detect",
    "segmentation": "segment",
    "pointing": "point",
    "grasp": "grasp",
    "vla": "act",
}


def _parse_trace(
    path: Path,
    manifest: EpisodeManifest,
    *,
    environment_suite: str,
    full_horizon_final_success: bool,
) -> tuple[tuple[PublicStepEvidence, ...], tuple[AgentEvent, ...]]:
    try:
        value = _mapping(
            decode_message(path.read_bytes()),
            {"outcome", "termination", "steps", "error"},
            str(path),
        )
        outcome = EpisodeOutcome.from_mapping(value["outcome"])
        if outcome != EpisodeOutcome.from_manifest(manifest):
            raise SmokeIntegrityError(f"trace outcome differs from episode manifest: {path}")
        if value["termination"] not in {"success", "horizon"} or value["error"] is not None:
            raise SmokeIntegrityError(f"trace has invalid clean termination: {path}")
        if full_horizon_final_success:
            if value["termination"] != "horizon":
                raise SmokeIntegrityError(f"full-horizon trace termination differs: {path}")
        elif outcome.success != (value["termination"] == "success"):
            raise SmokeIntegrityError(f"trace outcome and termination differ: {path}")
        if not isinstance(value["steps"], list) or not value["steps"]:
            raise SmokeIntegrityError(f"trace has no steps: {path}")
        steps = tuple(PublicStepEvidence.from_mapping(item) for item in value["steps"])
    except SmokeIntegrityError:
        raise
    except (OSError, StrictSchemaError, ValueError) as exc:
        raise SmokeIntegrityError(f"invalid trace {path}: {exc}") from exc
    indices = tuple(step.observation.step_index for step in steps)
    if indices != tuple(sorted(set(indices))):
        raise SmokeIntegrityError(f"trace step indices are not sorted and unique: {path}")
    if any(step.observation.episode_id != manifest.key.artifact_id() for step in steps):
        raise SmokeIntegrityError(f"trace episode identity differs: {path}")
    if (
        environment_suite == "robocasa365_target"
        or environment_suite == "vlabench_xvla_tracks_1_4"
        or environment_suite.startswith("simpler_")
        or environment_suite.startswith("libero")
    ):
        for step in steps:
            for name, camera in step.observation.cameras.items():
                try:
                    if environment_suite == "robocasa365_target":
                        validate_robocasa365_rgb(camera.rgb, name)
                    elif environment_suite.startswith("simpler_"):
                        validate_simpler_rgb(camera.rgb, name)
                    else:
                        validate_mujoco_rgb(camera.rgb, name)
                except StrictSchemaError as exc:
                    raise SmokeIntegrityError(
                        f"render integrity failed in {path} at step {step.observation.step_index}: {exc}"
                    ) from exc
    action_steps = sum(step.action is not None for step in steps)
    if action_steps != manifest.steps:
        raise SmokeIntegrityError(f"trace action count differs from episode manifest: {path}")
    if full_horizon_final_success and (
        manifest.steps != manifest.key.horizon
        or indices != tuple(range(manifest.key.horizon))
        or action_steps != len(steps)
    ):
        raise SmokeIntegrityError(f"OpenVLA SimplerEnv trace did not execute its full horizon: {path}")
    events = tuple(event for step in steps for event in step.events)
    return steps, events


def _verify_tool_calls(
    path: Path,
    manifest: EpisodeManifest,
    steps: tuple[PublicStepEvidence, ...],
    events: tuple[AgentEvent, ...],
    capabilities: frozenset[str],
) -> Counter[str]:
    expected = capabilities | {"vla"}
    bad = [
        event
        for event in events
        if event.capability in expected and event.status in {"optional_error", "infrastructure_error"}
    ]
    if bad:
        event = bad[0]
        raise SmokeIntegrityError(
            f"tool error in {path}: {event.capability} {event.event_type} {event.status}: {event.detail}"
        )
    successful = Counter()
    for capability in sorted(expected):
        calls = [event for event in events if event.event_type == "tool_call" and event.capability == capability]
        started = [event for event in calls if event.status == "started"]
        complete = [event for event in calls if event.status == "ok"]
        if not complete or len(started) != len(complete):
            raise SmokeIntegrityError(
                f"{path}: {capability} requires matched successful tool calls, got started={len(started)} ok={len(complete)}"
            )
        operation = _OPERATIONS[capability]
        if any(event.detail != operation for event in (*started, *complete)):
            raise SmokeIntegrityError(f"{path}: {capability} tool operation differs from {operation}")
        successful[capability] = len(complete)
    for step in steps:
        calls = [
            event
            for event in step.events
            if event.event_type == "tool_call" and event.capability == "vla"
        ]
        started = sum(event.status == "started" for event in calls)
        complete = sum(event.status == "ok" for event in calls)
        expected_vla = 1 if step.action is not None else 0
        if (started, complete) != (expected_vla, expected_vla):
            raise SmokeIntegrityError(
                f"{path}: step {step.observation.step_index} action and VLA call counts differ"
            )
    if manifest.key.horizon >= 9:
        boundary = [step for step in steps if step.observation.step_index == 8]
        if len(boundary) != 1:
            raise SmokeIntegrityError(f"{path}: horizon >=9 smoke did not reach progress-monitor step 8")
        step = boundary[0]
        vision_ok = any(
            event.event_type == "tool_call"
            and event.capability == "vision"
            and event.status == "ok"
            for event in step.events
        )
        monitor_handled = any(
            event.event_type == "monitor"
            and event.capability == "vision"
            and event.status in {"ok", "skipped"}
            for event in step.events
        )
        if not vision_ok or not monitor_handled:
            raise SmokeIntegrityError(f"{path}: step 8 progress monitor did not complete successfully")
    return successful


def _verify_evaluation(
    path: Path,
    split: str,
    profile: Profile,
    plan: Any,
    capabilities: frozenset[str],
) -> dict[str, Any]:
    try:
        final = ArtifactRun.verify(path / "artifacts")
    except Exception as exc:
        raise SmokeIntegrityError(f"artifact verification failed at {path}: {exc}") from exc
    expected_keys = plan.for_split(split)
    if (
        final["state"] != "complete"
        or final["scope_split"] != split
        or final["n_expected"] != len(expected_keys)
        or final["n_complete"] != len(expected_keys)
    ):
        raise SmokeIntegrityError(f"evaluation count or split differs at {path}")
    header = _mapping(
        _strict_json(path / "artifacts" / "run.json"),
        {
            "schema_version",
            "run_id",
            "profile_sha256",
            "episode_plan_sha256",
            "code_sha256",
            "created_ns",
            "service_identities",
            "episode_plan",
            "scope_split",
        },
        str(path / "artifacts" / "run.json"),
    )
    identities = tuple(ServiceIdentity.from_mapping(item) for item in header["service_identities"])
    try:
        profile.validate_service_identities(identities)
    except StrictSchemaError as exc:
        raise SmokeIntegrityError(f"evaluation service identities differ at {path}: {exc}") from exc
    if header["profile_sha256"] != profile.resolved_hash() or header["episode_plan_sha256"] != plan.resolved_hash():
        raise SmokeIntegrityError(f"evaluation profile or plan hash differs at {path}")
    replicas = {item.identity.replica_id: item.identity for item in profile.policy.replicas}
    replica_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    traces = {}
    executions = {}
    successes = {}
    full_horizon_final_success = is_openvla_simpler_adapter(profile.environment.adapter)
    for key in expected_keys:
        episode_root = path / "artifacts" / "episodes" / key.artifact_id()
        try:
            manifest = EpisodeManifest.from_mapping(_strict_json(episode_root / "episode.json"))
        except StrictSchemaError as exc:
            raise SmokeIntegrityError(f"invalid episode manifest at {episode_root}: {exc}") from exc
        if manifest.key != key or manifest.state != "complete":
            raise SmokeIntegrityError(f"episode key or state differs at {episode_root}")
        execution_path = episode_root / "execution.json"
        execution_value = _strict_json(execution_path)
        if isinstance(execution_value, Mapping) and execution_value.get("schema_version") == 1:
            execution = _mapping(
                execution_value,
                {"schema_version", "policy_service_name", "policy_replica_id", "gpu_ids"},
                str(execution_path),
            )
            render_gpu_id = execution["gpu_ids"][0] if len(execution["gpu_ids"]) == 1 else None
        else:
            execution = _mapping(
                execution_value,
                {"schema_version", "policy_service_name", "policy_replica_id", "gpu_ids", "render_gpu_id"},
                str(execution_path),
            )
            render_gpu_id = execution["render_gpu_id"]
        replica = replicas.get(execution["policy_replica_id"])
        if (
            execution["schema_version"] not in {1, 2}
            or replica is None
            or execution["policy_service_name"] != replica.service_name
            or execution["gpu_ids"] != list(replica.gpu_ids)
            or type(render_gpu_id) is not int
            or render_gpu_id not in profile.resources.gpu_ids
        ):
            raise SmokeIntegrityError(f"invalid policy execution identity at {execution_path}")
        trace_path = episode_root / "trace.msgpack"
        steps, events = _parse_trace(
            trace_path,
            manifest,
            environment_suite=profile.environment.suite,
            full_horizon_final_success=full_horizon_final_success,
        )
        tool_counts.update(_verify_tool_calls(trace_path, manifest, steps, events, capabilities))
        replica_counts[replica.replica_id] += 1
        traces[key.artifact_id()] = trace_path.read_bytes()
        executions[key.artifact_id()] = execution_path.read_bytes()
        successes[key.artifact_id()] = manifest.success
    expected_low = len(expected_keys) // len(replicas)
    expected_high = (len(expected_keys) + len(replicas) - 1) // len(replicas)
    if set(replica_counts) != set(replicas) or any(
        not expected_low <= replica_counts[replica] <= expected_high for replica in replicas
    ):
        raise SmokeIntegrityError(f"policy replicas are not balanced at {path}: {dict(replica_counts)}")
    return {
        "report": {
            "path": str(path),
            "split": split,
            "episodes": len(expected_keys),
            "policy_replica_episodes": dict(sorted(replica_counts.items())),
            "successful_tool_calls": dict(sorted(tool_counts.items())),
            "artifact_manifest_sha256": final["manifest_sha256"],
        },
        "traces": traces,
        "executions": executions,
        "successes": successes,
        "code_sha256": final["code_sha256"],
    }


def _verify_no_candidate(run_root: Path) -> None:
    attempts = run_root / "evolution" / "attempts"
    if not attempts.is_dir() or any(attempts.iterdir()):
        raise SmokeIntegrityError("no-candidate smoke contains candidate attempts")
    invocation_root = run_root / "invocations"
    started = tuple(
        sorted(
            path
            for path in invocation_root.glob("*.json")
            if re.fullmatch(r"[0-9]{6}\.json", path.name)
        )
    )
    if not started:
        raise SmokeIntegrityError("no-candidate smoke has no invocation record")
    for path in started:
        value = _mapping(
            _strict_json(path),
            {"schema_version", "started_ns", "target_candidates", "finalize", "run_transfer"},
            str(path),
        )
        if value["schema_version"] != 1 or value["target_candidates"] != 0:
            raise SmokeIntegrityError(f"no-candidate smoke invocation is invalid: {path}")
        finished_path = path.with_name(f"{path.stem}.finished.json")
        finished = _mapping(
            _strict_json(finished_path),
            {"schema_version", "finished_ns", "status", "error_type"},
            str(finished_path),
        )
        if finished["schema_version"] != 1 or finished["status"] != "complete" or finished["error_type"] is not None:
            raise SmokeIntegrityError(f"no-candidate smoke invocation did not complete: {finished_path}")
    final_invocation = _strict_json(started[-1])
    if final_invocation["finalize"] is not True or final_invocation["run_transfer"] is not True:
        raise SmokeIntegrityError("no-candidate smoke final invocation did not finalize and run transfer")
    baseline = run_root / "evolution" / "baseline" / "scaffold" / "scaffold.py"
    frozen = run_root / "evolution" / "frozen" / "scaffold" / "scaffold.py"
    if not baseline.is_file() or not frozen.is_file() or baseline.read_bytes() != frozen.read_bytes():
        raise SmokeIntegrityError("no-candidate smoke baseline and frozen scaffolds differ")


def verify_profile_smoke(
    project_root: str | Path,
    profile_path: str | Path,
    run_root: str | Path,
    *,
    expect_no_candidate: bool = False,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    source = Path(profile_path).resolve()
    root = Path(run_root).resolve()
    try:
        profile = Profile.load(source, project_root=project)
        snapshot = Profile.load(root / "profile.json", project_root=project)
        plan = snapshot.episode_plan.load(project)
    except (OSError, StrictSchemaError) as exc:
        raise SmokeIntegrityError(f"failed to load smoke profile: {exc}") from exc
    if profile.to_mapping() != snapshot.to_mapping():
        raise SmokeIntegrityError("run profile snapshot differs from requested smoke profile")
    capabilities = frozenset(tool.capability for tool in snapshot.tools if tool.enabled)
    unsupported = capabilities - set(_OPERATIONS)
    if unsupported:
        raise SmokeIntegrityError(f"smoke verifier has no operation contract for {sorted(unsupported)}")
    evaluation_specs = (
        ("baseline_evolve", root / "evolution" / "baseline" / "evolve", "evolve"),
        ("baseline_selection", root / "evolution" / "baseline" / "selection", "selection"),
        ("transfer_baseline", root / "evolution" / "transfer" / "baseline", "transfer"),
        ("transfer_evolved", root / "evolution" / "transfer" / "evolved", "transfer"),
    )
    expected_artifact_roots = {path / "artifacts" for _, path, _ in evaluation_specs}
    discovered_artifact_roots = set((root / "evolution").glob("**/artifacts"))
    if expect_no_candidate and discovered_artifact_roots != expected_artifact_roots:
        raise SmokeIntegrityError("no-candidate smoke evaluation artifact set differs")
    groups = {
        name: _verify_evaluation(path, split, snapshot, plan, capabilities)
        for name, path, split in evaluation_specs
    }
    if expect_no_candidate:
        _verify_no_candidate(root)
        baseline = groups["transfer_baseline"]
        evolved = groups["transfer_evolved"]
        if (
            baseline["code_sha256"] != evolved["code_sha256"]
            or baseline["traces"] != evolved["traces"]
            or baseline["executions"] != evolved["executions"]
            or baseline["successes"] != evolved["successes"]
        ):
            raise SmokeIntegrityError("no-candidate baseline and evolved transfer executions differ")
    policy_totals: Counter[str] = Counter()
    tool_totals: Counter[str] = Counter()
    for value in groups.values():
        policy_totals.update(value["report"]["policy_replica_episodes"])
        tool_totals.update(value["report"]["successful_tool_calls"])
    sam_enabled = any(
        tool.enabled and tool.service is not None and tool.service.identity.service_name == "sam3"
        for tool in snapshot.tools
    )
    sam_log_report = None
    if sam_enabled:
        logs = tuple(
            sorted(
                path
                for path in (root / "invocation_artifacts").glob("*/services/sam3/*.log")
                if path.name != "z"
            )
        )
        sam_log_report = verify_sam_service_logs(logs)
    return {
        "schema_version": 1,
        "state": "complete",
        "profile_sha256": snapshot.resolved_hash(),
        "episode_plan_sha256": plan.resolved_hash(),
        "expected_enabled_capabilities": sorted(capabilities),
        "evaluations": {name: value["report"] for name, value in groups.items()},
        "total_episode_executions": sum(value["report"]["episodes"] for value in groups.values()),
        "policy_replica_episodes": dict(sorted(policy_totals.items())),
        "successful_tool_calls": dict(sorted(tool_totals.items())),
        "sam_service_logs": sam_log_report,
        "no_candidate_transfer_equivalent": bool(expect_no_candidate),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m robot_auto_evolve.smoke_integrity")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sam_logs = subparsers.add_parser("sam-logs")
    sam_logs.add_argument("--output", type=Path, required=True)
    sam_logs.add_argument("logs", nargs="+", type=Path)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--project-root", type=Path, required=True)
    profile.add_argument("--profile", type=Path, required=True)
    profile.add_argument("--run-root", type=Path, required=True)
    profile.add_argument("--expect-no-candidate", action="store_true")
    profile.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "sam-logs":
            result = verify_sam_service_logs(args.logs)
        elif args.command == "profile":
            result = verify_profile_smoke(
                args.project_root,
                args.profile,
                args.run_root,
                expect_no_candidate=args.expect_no_candidate,
            )
        else:
            raise AssertionError(args.command)
        _atomic_json(args.output, result)
        return 0
    except (OSError, SmokeIntegrityError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
