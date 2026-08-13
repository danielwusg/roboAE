from __future__ import annotations

import argparse
import json
from pathlib import Path


def _outcomes(result_path: Path) -> dict[str, list[bool]]:
    value = json.loads(result_path.read_text(encoding="utf-8"))
    per_task: dict[str, list[bool]] = {}
    for item in value["outcomes"]:
        key = item["key"]
        metrics = item["metrics"]
        success = bool(metrics.get("success", False))
        per_task.setdefault(key["task_id"], []).append(success)
    return per_task


def balanced_split(rates: dict[str, float]) -> tuple[list[str], list[str]]:
    ordered = sorted(rates.items(), key=lambda pair: (-pair[1], pair[0]))
    evolve: list[str] = []
    held_out: list[str] = []
    for index, (task_id, _) in enumerate(ordered):
        (evolve if index % 2 == 0 else held_out).append(task_id)
    return sorted(evolve), sorted(held_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read a finished full-task-set baseline and deal the tasks into two halves of matched "
            "difficulty, best and worst alternating, so neither half is all the tasks the policy "
            "already solves or all the ones it never solves."
        )
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        required=True,
        help="runs/<study-id>/evolution/baseline/benchmark_result.json from a full-task-set run",
    )
    args = parser.parse_args(argv)
    per_task = _outcomes(args.baseline_result)
    rates = {task: sum(values) / len(values) for task, values in sorted(per_task.items())}
    evolve, held_out = balanced_split(rates)
    payload = {
        "n_tasks": len(rates),
        "per_task_success": rates,
        "evolve_task_ids": evolve,
        "transfer_task_ids": held_out,
        "evolve_mean_success": (
            sum(rates[task] for task in evolve) / len(evolve) if evolve else 0.0
        ),
        "transfer_mean_success": (
            sum(rates[task] for task in held_out) / len(held_out) if held_out else 0.0
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
