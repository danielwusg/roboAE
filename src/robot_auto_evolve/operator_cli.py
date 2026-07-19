from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from robot_auto_evolve.operator_catalog import (
    build_study_request,
    list_route_tasks,
    load_catalog,
    load_route_spec,
    materialize_runtime_profile,
    materialize_runtime_profiles,
    materialize_study_request,
)
from robot_auto_evolve.protocol import StrictSchemaError


def _gpu_ids(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if len(parts) < 2 or any(re.fullmatch(r"0|[1-9][0-9]*", item) is None for item in parts):
        raise argparse.ArgumentTypeError("expected at least two comma-separated nonnegative GPU IDs")
    result = tuple(int(item) for item in parts)
    if result != tuple(sorted(set(result))):
        raise argparse.ArgumentTypeError("GPU IDs must be sorted and unique")
    return result


def _render_gpu_ids(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(re.fullmatch(r"0|[1-9][0-9]*", item) is None for item in parts):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative render GPU IDs")
    return tuple(int(item) for item in parts)


def _positive(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int(value)


def _nonnegative(value: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return int(value)


def _print_tasks(root: Path, route_id: str) -> None:
    spec, _, _ = load_route_spec(root, route_id)
    benchmark = spec["benchmark"]
    print(f"route: {route_id}")
    print(f"full_benchmark_status: {benchmark['status']}")
    if benchmark["blocker"] is not None:
        print(f"blocker: {benchmark['blocker']}")
    print("task_id\tstandard_rows\thorizons\tprotocols")
    for task in list_route_tasks(root, route_id):
        horizons = "-" if task["horizons"] is None else ",".join(str(item) for item in task["horizons"])
        protocols = ",".join(task["protocols"])
        count = "-" if task["episode_count"] is None else str(task["episode_count"])
        print(f"{task['id']}\t{count}\t{horizons}\t{protocols}")
    for preset in spec.get("related_transfer_presets", []):
        print(f"preset: {preset['preset_id']}")
        print("preset_evolve: " + ",".join(preset["evolve_task_ids"]))
        print("preset_transfer: " + ",".join(preset["transfer_task_ids"]))


def _run_route(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    if args.list_tasks:
        if args.run_id is not None:
            raise ValueError("--list-tasks does not accept RUN_ID")
        _print_tasks(root, args.route)
        return 0
    if args.run_id is None:
        raise ValueError("RUN_ID is required")
    if args.target_candidates is None:
        raise ValueError("--target-candidates is required")
    if args.task_preset is not None and (args.evolve_task or args.transfer_task):
        raise ValueError("--task-preset cannot be combined with explicit task flags")
    if args.task_preset is None and bool(args.evolve_task) != bool(args.transfer_task):
        raise ValueError("explicit task selection requires both --evolve-task and --transfer-task")
    if args.run_transfer and not args.finalize:
        raise ValueError("--run-transfer requires --finalize")
    if args.run_transfer and not (args.transfer_task or args.task_preset):
        raise ValueError("--run-transfer requires an explicit held-out --transfer-task set")
    render_ids = args.render_gpu_ids or args.gpu_ids
    request = build_study_request(
        root,
        args.route,
        args.run_id,
        evolve_task_ids=args.evolve_task,
        transfer_task_ids=args.transfer_task,
        task_preset=args.task_preset,
        gpu_ids=args.gpu_ids,
        render_gpu_ids=render_ids,
        workers_per_gpu=args.workers_per_gpu,
        port_offset=args.port_offset,
    )
    if args.target_candidates > request.candidate_budget:
        raise ValueError(f"--target-candidates exceeds route candidate budget {request.candidate_budget}")
    study_id = request.study_id
    run_root = root / "runs" / study_id
    if args.prepare_only:
        path = materialize_study_request(request, run_root)
        profile_path = materialize_runtime_profile(request, run_root)
        profile_paths = materialize_runtime_profiles(request, run_root)
        print(
            json.dumps(
                {
                    "study_request": str(path),
                    "runtime_profile": str(profile_path),
                    "runtime_profiles": {key: str(value) for key, value in profile_paths.items()},
                },
                sort_keys=True,
            )
        )
        return 0
    if shutil.which("claude") is None:
        raise ValueError("claude is not on PATH")
    # claude_free uses the ambient Claude credential (no private oauth_token dir).
    request_path = materialize_study_request(request, run_root)
    materialize_runtime_profile(request, run_root)
    command = [
        sys.executable,
        "-m",
        "robot_auto_evolve",
        "run-study",
        "--study-request",
        str(request_path),
        "--target-candidates",
        str(args.target_candidates),
        "--meta-backend",
        args.meta_backend,
    ]
    if args.smoke_episodes > 0:
        command += ["--smoke-episodes", str(args.smoke_episodes), "--smoke-horizon", str(args.smoke_horizon)]
    if args.smoke_no_tools:
        command.append("--smoke-no-tools")
    if args.finalize:
        command.append("--finalize")
    if args.run_transfer:
        command.append("--run-transfer")
    return subprocess.call(command, cwd=root)


def _run_group(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    catalog = load_catalog(root)
    matches = [item for item in catalog["groups"] if item["group_id"] == args.group]
    if len(matches) != 1:
        raise ValueError(f"unknown route group: {args.group}")
    group = matches[0]
    if args.list_tasks:
        if args.arguments:
            raise ValueError("--list-tasks does not accept additional arguments")
        for route_id in group["route_ids"]:
            _print_tasks(root, route_id)
            print()
        return 0
    if not args.arguments:
        raise ValueError("group launch requires RUN_ID and route arguments")
    if any(
        item == "--evolve-task"
        or item.startswith("--evolve-task=")
        or item == "--transfer-task"
        or item.startswith("--transfer-task=")
        for item in args.arguments
    ):
        raise ValueError("group convenience launchers do not accept task filters; launch member routes directly")
    run_id, *options = args.arguments
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError("RUN_ID is invalid")
    print(group["scope_note"], file=sys.stderr)
    for route_id in group["route_ids"]:
        spec, _, _ = load_route_spec(root, route_id)
        command = [str(root / spec["wrapper"]), f"{run_id}_{route_id}", *options]
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pinned robot-agent evolution study.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    route = subparsers.add_parser("route", help="Run one model-environment route.")
    route.add_argument("--project-root", type=Path, required=True)
    route.add_argument("--route", required=True)
    route.add_argument("run_id", nargs="?", help="Run label; the route ID is prepended in runs/.")
    route.add_argument("--list-tasks", action="store_true", help="Print exact task IDs and exit without starting services.")
    route.add_argument("--evolve-task", action="append", default=[], help="Evolution task ID; repeat for each selected task.")
    route.add_argument("--transfer-task", action="append", default=[], help="Held-out task ID; repeat for each selected task.")
    route.add_argument("--task-preset", help="Audited task split, normally 'related'.")
    route.add_argument("--gpu-ids", type=_gpu_ids, default=(0, 1), help="Sorted GPU pool, such as 0,1 or 0,1,2,3.")
    route.add_argument("--render-gpu-ids", type=_render_gpu_ids, help="One render GPU per policy replica; defaults to --gpu-ids.")
    route.add_argument("--workers-per-gpu", type=_positive, help="Simulator workers per selected GPU; route default if omitted.")
    route.add_argument("--port-offset", type=_nonnegative, default=0, help="Add this offset to every service port.")
    route.add_argument("--target-candidates", type=_nonnegative, help="Total completed proposals requested, excluding baseline.")
    route.add_argument("--meta-backend", choices=("claude_free",), default="claude_free", help="Coding-agent backend (only claude_free; the legacy sandboxed relay was removed).")
    route.add_argument("--smoke-episodes", type=_positive, default=0, help="Smoke test: keep at most this many episodes per task (runtime shrink).")
    route.add_argument("--smoke-horizon", type=_nonnegative, default=0, help="Smoke test: cap every episode horizon to this many steps.")
    route.add_argument("--smoke-no-tools", action="store_true", help="Smoke test: suppress all tool services (policy-only).")
    route.add_argument("--finalize", action="store_true", help="Freeze after this target was completed in an earlier call.")
    route.add_argument("--run-transfer", action="store_true", help="After finalization, compare baseline and frozen on held-out tasks.")
    route.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    route.set_defaults(handler=_run_route)
    group = subparsers.add_parser("group", help="Run independent suite or cell slice studies in sequence.")
    group.add_argument("--project-root", type=Path, required=True)
    group.add_argument("--group", required=True)
    group.add_argument("--list-tasks", action="store_true")
    group.add_argument("arguments", nargs=argparse.REMAINDER)
    group.set_defaults(handler=_run_group)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except (OSError, PermissionError, RuntimeError, StrictSchemaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
