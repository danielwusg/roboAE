from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from robot_auto_evolve.evaluation.private_metrics import validate_private_metrics
from robot_auto_evolve.evaluation.scalars import SCALAR_METRICS, BenchmarkOutcome, compute_benchmark_scalar
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest, mapping_sha256

from .benchmark_models import BenchmarkEvaluationData


_SCALAR_OUTCOME_METRICS = {
    "equal_track_task_macro_progress_score": frozenset({"progress_score"}),
    "calvin_average_chain_length": frozenset({"completed_subtasks"}),
    "mean_completed_subtasks_per_sequence": frozenset({"completed_subtasks"}),
}


def canonical_outcome_metrics(
    path: Path,
    manifest: EpisodeManifest,
    scalar_metric: str,
) -> dict[str, bool | float]:
    if scalar_metric not in SCALAR_METRICS:
        raise StrictSchemaError("canonical benchmark scalar metric differs")
    required = _SCALAR_OUTCOME_METRICS.get(scalar_metric, frozenset())
    if manifest.state == "error":
        return {"success": False, **{name: 0.0 for name in sorted(required)}}
    metrics: dict[str, bool | float] = {"success": bool(manifest.success)}
    if not required:
        return metrics
    source = Path(path) / "private_metrics.json"
    if not source.is_file() or source.is_symlink():
        raise StrictSchemaError("canonical benchmark required outcome metrics artifact differs")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StrictSchemaError(f"canonical benchmark outcome metrics are invalid: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "kind", "metrics"}:
        raise StrictSchemaError("canonical benchmark outcome metrics fields differ")
    if value["schema_version"] != 1 or value["kind"] != "private_evaluator_metrics":
        raise StrictSchemaError("canonical benchmark outcome metrics identity differs")
    private = validate_private_metrics(value["metrics"])
    if "success" in private and private["success"] is not manifest.success:
        raise StrictSchemaError("canonical benchmark private success differs")
    missing = required - set(private)
    if missing:
        raise StrictSchemaError(f"canonical benchmark lacks required outcome metric {sorted(missing)[0]!r}")
    metrics.update({name: private[name] for name in sorted(required)})
    return metrics


class CanonicalBenchmarkEvolutionAdapter:
    def __init__(
        self,
        evaluator: Any,
        plan: BenchmarkPlan,
        scalar_metric: str,
        *,
        invocation_root: Path | None = None,
    ) -> None:
        if (
            not isinstance(plan, BenchmarkPlan)
            or not callable(getattr(evaluator, "evaluate", None))
            or scalar_metric not in SCALAR_METRICS
        ):
            raise StrictSchemaError("canonical benchmark adapter inputs differ")
        self.evaluator = evaluator
        self.plan = plan
        self.scalar_metric = scalar_metric
        self.invocation_root = None if invocation_root is None else Path(invocation_root).resolve()

    def _metrics(self, path: Path, manifest: EpisodeManifest) -> dict[str, bool | float]:
        return canonical_outcome_metrics(path, manifest, self.scalar_metric)

    def evaluate(self, scaffold_dir: Path, output_dir: Path) -> BenchmarkEvaluationData:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        evaluation = output / "canonical"
        if self.invocation_root is None:
            invocation = output / "invocation"
        else:
            self.invocation_root.mkdir(parents=True, exist_ok=True)
            invocation = self.invocation_root / f"evaluation-{uuid.uuid4().hex}"
        report = self.evaluator.evaluate(Path(scaffold_dir).resolve(), evaluation, invocation)
        if not isinstance(report, dict) or report.get("complete") is not True:
            raise RuntimeError("canonical benchmark evaluation is incomplete")
        rows = []
        for key in self.plan.episodes:
            root = evaluation / "episodes" / key.artifact_id()
            manifest = EpisodeManifest.from_mapping(json.loads((root / "episode.json").read_text(encoding="utf-8")))
            if (
                manifest.key != key
                or manifest.state not in {"complete", "error"}
                or (manifest.state == "complete" and manifest.success is None)
            ):
                raise StrictSchemaError("canonical benchmark episode differs from exact plan")
            rows.append(BenchmarkOutcome(key, self._metrics(root, manifest)))
        scalar = compute_benchmark_scalar(self.scalar_metric, rows)
        report_metrics = report.get("metrics")
        if (
            not isinstance(report_metrics, dict)
            or report_metrics.get("metric") != scalar.metric
            or report_metrics.get("score") != scalar.value
            or ("details" in report_metrics and report_metrics["details"] != scalar.details)
        ):
            raise StrictSchemaError("canonical benchmark report and route scalar differ")
        return BenchmarkEvaluationData(
            outcomes=tuple(rows),
            metadata={
                "canonical_report_sha256": mapping_sha256(report),
                "canonical_plan_sha256": self.plan.resolved_hash(),
            },
        )
