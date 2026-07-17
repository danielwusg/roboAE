from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Mapping, Protocol

from robot_auto_evolve.protocol.schema import StrictSchemaError, boolean, enum, integer, mapping, string
from robot_auto_evolve.provenance import ArtifactRun, EpisodeKey, EpisodeManifest, EpisodePlan

from .metrics import EpisodeOutcome, TaskMacroMetrics, compute_task_macro_metrics


class EpisodeRunner(Protocol):
    def __call__(self, key: EpisodeKey) -> "EpisodeExecution": ...


@dataclass(frozen=True)
class EpisodeExecution:
    state: str
    success: bool | None
    steps: int
    artifacts: Mapping[str, bytes]
    error: str | None = None

    def __post_init__(self) -> None:
        state = enum(self.state, {"complete", "partial", "error"}, "execution.state")
        success = None if self.success is None else boolean(self.success, "execution.success")
        steps = integer(self.steps, "execution.steps", minimum=0)
        artifacts = mapping(self.artifacts, "execution.artifacts")
        checked: dict[str, bytes] = {}
        for name, value in sorted(artifacts.items()):
            name = string(name, "execution.artifact name")
            if type(value) is not bytes:
                raise StrictSchemaError(f"execution.artifacts.{name}: expected bytes")
            checked[name] = value
        error = None if self.error is None else string(self.error, "execution.error")
        if state == "complete" and (success is None or error is not None or "trace.msgpack" not in checked):
            raise StrictSchemaError("execution: complete requires success, trace.msgpack, and null error")
        if state == "partial" and (success is not None or error is not None):
            raise StrictSchemaError("execution: partial requires null success and error")
        if state == "error" and (success is not None or error is None):
            raise StrictSchemaError("execution: error requires null success and non-null error")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "artifacts", checked)
        object.__setattr__(self, "error", error)


@dataclass(frozen=True)
class EvaluationSummary:
    split: str
    n_expected: int
    n_recorded: int
    n_complete: int
    n_partial: int
    n_error: int
    complete: bool
    metrics: TaskMacroMetrics | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "split": self.split,
            "n_expected": self.n_expected,
            "n_recorded": self.n_recorded,
            "n_complete": self.n_complete,
            "n_partial": self.n_partial,
            "n_error": self.n_error,
            "complete": self.complete,
            "metrics": None if self.metrics is None else self.metrics.to_mapping(),
        }


def _execute_one(run: ArtifactRun, runner: EpisodeRunner, key: EpisodeKey) -> EpisodeManifest:
    started = time.time_ns()
    try:
        result = runner(key)
        if not isinstance(result, EpisodeExecution):
            raise StrictSchemaError("episode runner returned wrong type")
    except Exception as exc:
        result = EpisodeExecution(
            state="error",
            success=None,
            steps=0,
            artifacts={"error.txt": f"{type(exc).__name__}: {exc}".encode("utf-8")},
            error=f"runner_exception:{type(exc).__name__}",
        )
    return run.record_episode(
        key,
        state=result.state,
        success=result.success,
        steps=result.steps,
        artifacts=result.artifacts,
        error=result.error,
        started_ns=started,
        finished_ns=time.time_ns(),
    )


def summarize_split(plan: EpisodePlan, manifests: tuple[EpisodeManifest, ...], split: str) -> EvaluationSummary:
    expected = plan.for_split(split)
    expected_set = set(expected)
    rows = tuple(sorted((item for item in manifests if item.key in expected_set), key=lambda item: item.key))
    complete_rows = tuple(item for item in rows if item.state == "complete")
    is_complete = len(rows) == len(expected) and len(complete_rows) == len(expected)
    metrics = None
    if is_complete:
        metrics = compute_task_macro_metrics(EpisodeOutcome.from_manifest(item) for item in complete_rows)
    return EvaluationSummary(
        split=split,
        n_expected=len(expected),
        n_recorded=len(rows),
        n_complete=len(complete_rows),
        n_partial=sum(item.state == "partial" for item in rows),
        n_error=sum(item.state == "error" for item in rows),
        complete=is_complete,
        metrics=metrics,
    )


def evaluate_supplied_runner(
    plan: EpisodePlan,
    run: ArtifactRun,
    runner: EpisodeRunner,
    *,
    split: str,
    workers: int = 1,
) -> EvaluationSummary:
    if not isinstance(plan, EpisodePlan) or plan.resolved_hash() != run.plan.resolved_hash():
        raise StrictSchemaError("evaluation: plan differs from artifact run")
    split = enum(split, {"evolve", "selection", "transfer"}, "evaluation.split")
    if run.scope_split is not None and run.scope_split != split:
        raise StrictSchemaError("evaluation: split differs from artifact run scope")
    workers = integer(workers, "evaluation.workers", minimum=1)
    existing = {item.key for item in run.episode_manifests()}
    pending = [key for key in plan.for_split(split) if key not in existing]
    if workers == 1:
        for key in pending:
            _execute_one(run, runner, key)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_one, run, runner, key): key for key in pending}
            for future in as_completed(futures):
                future.result()
    return summarize_split(plan, run.episode_manifests(), split)
