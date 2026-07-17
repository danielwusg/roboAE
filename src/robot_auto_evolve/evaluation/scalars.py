from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from robot_auto_evolve.benchmarks.libero_suites import LIBERO_SUITE_TASKS, LIBERO_TASK_SUITE
from robot_auto_evolve.benchmarks.robocasa365 import TARGET_TASK_GROUPS
from robot_auto_evolve.benchmarks.robocerebra import CONDITIONS, parse_task_id
from robot_auto_evolve.benchmarks.vlabench_worker import parse_vlabench_scenario
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import EpisodeKey


SCALAR_METRICS = frozenset(
    {
        "task_macro_success",
        "equal_task_macro_success",
        "equal_suite_task_macro_success",
        "equal_cell_task_macro_success",
        "equal_track_task_macro_progress_score",
        "equal_group_task_macro_success",
        "equal_condition_case_macro_success",
        "calvin_average_chain_length",
        "mean_completed_subtasks_per_sequence",
    }
)


@dataclass(frozen=True)
class BenchmarkOutcome:
    key: EpisodeKey
    metrics: Mapping[str, bool | float]

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey) or self.key.split != "benchmark":
            raise StrictSchemaError("benchmark outcome requires a benchmark EpisodeKey")
        if not isinstance(self.metrics, Mapping) or "success" not in self.metrics:
            raise StrictSchemaError("benchmark outcome requires public success")
        checked: dict[str, bool | float] = {}
        for name, value in sorted(self.metrics.items()):
            if type(name) is not str or not name or len(name) > 64:
                raise StrictSchemaError("benchmark outcome metric name differs")
            if type(value) is bool:
                checked[name] = value
            elif type(value) in (int, float) and math.isfinite(float(value)):
                checked[name] = float(value)
            else:
                raise StrictSchemaError(f"benchmark outcome metric {name!r} is not a finite scalar")
        if type(checked["success"]) is not bool:
            raise StrictSchemaError("benchmark outcome success must be bool")
        object.__setattr__(self, "metrics", checked)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkOutcome":
        if not isinstance(value, Mapping) or set(value) != {"key", "metrics"}:
            raise StrictSchemaError("benchmark outcome fields differ")
        return cls(EpisodeKey.from_mapping(value["key"]), value["metrics"])

    def to_mapping(self) -> dict[str, Any]:
        return {"key": self.key.to_mapping(), "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class BenchmarkScalar:
    metric: str
    value: float
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.metric not in SCALAR_METRICS:
            raise StrictSchemaError("benchmark scalar metric differs")
        if type(self.value) not in (int, float) or not math.isfinite(float(self.value)):
            raise StrictSchemaError("benchmark scalar value is not finite")
        if not isinstance(self.details, Mapping):
            raise StrictSchemaError("benchmark scalar details must be a mapping")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "details", dict(self.details))

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkScalar":
        if not isinstance(value, Mapping) or set(value) != {"metric", "value", "details"}:
            raise StrictSchemaError("benchmark scalar fields differ")
        return cls(value["metric"], value["value"], value["details"])

    def to_mapping(self) -> dict[str, Any]:
        return {"metric": self.metric, "value": self.value, "details": dict(self.details)}


def _checked(values: Iterable[BenchmarkOutcome]) -> tuple[BenchmarkOutcome, ...]:
    rows = tuple(sorted(values, key=lambda item: item.key))
    if not rows or any(not isinstance(item, BenchmarkOutcome) for item in rows):
        raise StrictSchemaError("benchmark scalar requires outcomes")
    if len({item.key for item in rows}) != len(rows):
        raise StrictSchemaError("benchmark scalar outcomes contain duplicate keys")
    return rows


def _mean(values: Iterable[float], path: str) -> float:
    rows = tuple(float(value) for value in values)
    if not rows or any(not math.isfinite(value) for value in rows):
        raise StrictSchemaError(f"{path}: expected finite values")
    return sum(rows) / len(rows)


def _task_means(rows: tuple[BenchmarkOutcome, ...], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.metrics.get(metric)
        if type(value) is bool:
            number = float(value)
        elif type(value) in (int, float) and math.isfinite(float(value)):
            number = float(value)
        else:
            raise StrictSchemaError(f"benchmark outcome lacks numeric {metric!r}")
        grouped.setdefault(row.key.task_id, []).append(number)
    return {task: _mean(values, f"task {task}") for task, values in sorted(grouped.items())}


def _task_macro_success(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    tasks = _task_means(rows, "success")
    return BenchmarkScalar(
        "task_macro_success",
        _mean(tasks.values(), "task macro"),
        {"task_success": tasks, "n_tasks": len(tasks), "n_episodes": len(rows)},
    )


def _libero(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    tasks = _task_means(rows, "success")
    unknown = set(tasks) - set(LIBERO_TASK_SUITE)
    if unknown:
        raise StrictSchemaError(f"LIBERO scalar contains unknown task {sorted(unknown)[0]!r}")
    suites: dict[str, list[float]] = {}
    for task, value in tasks.items():
        suites.setdefault(LIBERO_TASK_SUITE[task], []).append(value)
    suite_success = {suite: _mean(values, suite) for suite, values in sorted(suites.items())}
    return BenchmarkScalar(
        "equal_suite_task_macro_success",
        _mean(suite_success.values(), "LIBERO suite macro"),
        {"suite_success": suite_success, "task_success": tasks, "n_episodes": len(rows)},
    )


def _libero_pro(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    tasks = _task_means(rows, "success")
    cells: dict[str, list[float]] = {}
    for task, value in tasks.items():
        if "::" not in task:
            raise StrictSchemaError("LIBERO-Pro task lacks a cell namespace")
        cell, task_name = task.split("::", 1)
        if not cell.startswith("libero_pro_") or not task_name:
            raise StrictSchemaError("LIBERO-Pro task namespace differs")
        cells.setdefault(cell, []).append(value)
    cell_success = {cell: _mean(values, cell) for cell, values in sorted(cells.items())}
    return BenchmarkScalar(
        "equal_cell_task_macro_success",
        _mean(cell_success.values(), "LIBERO-Pro cell macro"),
        {"cell_success": cell_success, "task_success": tasks, "n_episodes": len(rows)},
    )


def _vlabench(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = row.metrics.get("progress_score")
        if type(value) not in (int, float) or type(value) is bool or not 0.0 <= float(value) <= 1.0:
            raise StrictSchemaError("VLABench outcome requires progress_score in [0, 1]")
        track, _ = parse_vlabench_scenario(row.key.scenario_id)
        grouped.setdefault(track, {}).setdefault(row.key.task_id, []).append(float(value))
    task_progress = {
        track: {task: _mean(values, f"{track}/{task}") for task, values in sorted(tasks.items())}
        for track, tasks in sorted(grouped.items())
    }
    track_progress = {
        track: _mean(tasks.values(), track) for track, tasks in task_progress.items()
    }
    return BenchmarkScalar(
        "equal_track_task_macro_progress_score",
        _mean(track_progress.values(), "VLABench track macro"),
        {"track_progress": track_progress, "task_progress": task_progress, "n_episodes": len(rows)},
    )


def _robocasa(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    tasks = _task_means(rows, "success")
    expected = {task for values in TARGET_TASK_GROUPS.values() for task in values}
    unknown = set(tasks) - expected
    if unknown:
        raise StrictSchemaError(f"RoboCasa scalar contains unknown task {sorted(unknown)[0]!r}")
    groups = {
        group: _mean((tasks[task] for task in group_tasks if task in tasks), group)
        for group, group_tasks in TARGET_TASK_GROUPS.items()
        if any(task in tasks for task in group_tasks)
    }
    return BenchmarkScalar(
        "equal_group_task_macro_success",
        _mean(groups.values(), "RoboCasa group macro"),
        {"group_success": groups, "task_success": tasks, "n_episodes": len(rows)},
    )


def _robocerebra(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    tasks = _task_means(rows, "success")
    grouped: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    for task, value in tasks.items():
        condition, _ = parse_task_id(task)
        grouped[condition].append(value)
    conditions = {
        condition: _mean(values, condition)
        for condition, values in grouped.items()
        if values
    }
    return BenchmarkScalar(
        "equal_condition_case_macro_success",
        _mean(conditions.values(), "RoboCerebra condition macro"),
        {"condition_success": conditions, "case_success": tasks, "n_episodes": len(rows)},
    )


def _calvin(rows: tuple[BenchmarkOutcome, ...]) -> BenchmarkScalar:
    lengths = []
    for row in rows:
        value = row.metrics.get("completed_subtasks")
        if type(value) not in (int, float) or type(value) is bool or float(value) not in {0, 1, 2, 3, 4, 5}:
            raise StrictSchemaError("CALVIN outcome requires integer completed_subtasks in [0, 5]")
        lengths.append(int(value))
    success_at = {
        f"success_at_{threshold}": sum(value >= threshold for value in lengths) / len(lengths)
        for threshold in range(1, 6)
    }
    return BenchmarkScalar(
        "calvin_average_chain_length",
        _mean(lengths, "CALVIN chain length"),
        {"average_chain_length": _mean(lengths, "CALVIN chain length"), **success_at, "n_sequences": len(rows)},
    )


def compute_benchmark_scalar(metric: str, values: Iterable[BenchmarkOutcome]) -> BenchmarkScalar:
    if metric not in SCALAR_METRICS:
        raise StrictSchemaError(f"unsupported benchmark scalar metric {metric!r}")
    rows = _checked(values)
    result = {
        "task_macro_success": _task_macro_success,
        "equal_task_macro_success": _task_macro_success,
        "equal_suite_task_macro_success": _libero,
        "equal_cell_task_macro_success": _libero_pro,
        "equal_track_task_macro_progress_score": _vlabench,
        "equal_group_task_macro_success": _robocasa,
        "equal_condition_case_macro_success": _robocerebra,
        "calvin_average_chain_length": _calvin,
        "mean_completed_subtasks_per_sequence": _calvin,
    }[metric](rows)
    return result if result.metric == metric else BenchmarkScalar(metric, result.value, result.details)
