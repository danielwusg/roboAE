from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_auto_evolve.process_lifecycle import (
    OwnedProcessRegistry,
    RunInterrupted,
    interruption_handlers,
    process_registry,
)
from robot_auto_evolve.study_runner import run_study

from .evolution import EditablePolicy


def _run_study(args: argparse.Namespace) -> int:
    result = run_study(
        args.study_request,
        target_candidates=args.target_candidates,
        finalize=args.finalize,
        run_transfer=args.run_transfer,
        meta_backend=args.meta_backend,
        smoke_episodes=args.smoke_episodes,
        smoke_horizon=args.smoke_horizon,
        smoke_no_tools=args.smoke_no_tools,
        seed_scaffold=args.seed_scaffold,
        fairness_guard=args.fairness_guard,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_validate_scaffold(args: argparse.Namespace) -> int:
    hashes = EditablePolicy().validate_tree(args.scaffold_dir)
    print(json.dumps(hashes, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot-auto-evolve")
    subparsers = parser.add_subparsers(dest="command", required=True)
    study = subparsers.add_parser("run-study")
    study.add_argument("--study-request", type=Path, required=True)
    study.add_argument("--target-candidates", type=int, required=True)
    study.add_argument("--finalize", action="store_true")
    study.add_argument("--run-transfer", action="store_true")
    study.add_argument("--meta-backend", choices=("claude_free",), default="claude_free")
    study.add_argument(
        "--smoke-episodes", type=int, default=0,
        help="If >0, keep at most this many episodes per task (runtime smoke shrink).",
    )
    study.add_argument(
        "--smoke-horizon", type=int, default=0,
        help="If >0, cap every episode horizon to this many steps (smoke).",
    )
    study.add_argument(
        "--smoke-no-tools", action="store_true",
        help="Suppress all tool services (policy-only smoke): the scaffold sees no tools.",
    )
    study.add_argument(
        "--seed-scaffold", default=None,
        help="Override the route's starting scaffold with this project-relative scaffold dir "
             "(e.g. scaffolds/policy_passthrough_seed for the bare-policy seed).",
    )
    study.add_argument(
        "--fairness-guard", action=argparse.BooleanOptionalAction, default=False,
        help="Run the optional static grep-guard over each revised scaffold. OFF by default."
    )
    study.set_defaults(handler=_run_study)
    validate = subparsers.add_parser("validate-scaffold")
    validate.add_argument("scaffold_dir", type=Path)
    validate.set_defaults(handler=_run_validate_scaffold)
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
