from __future__ import annotations

import hashlib
import io
import json
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from robot_auto_evolve.agent import AgentEvent
from robot_auto_evolve.benchmarks.vlabench_worker import parse_vlabench_scenario
from robot_auto_evolve.evaluation import EpisodeOutcome
from robot_auto_evolve.evaluation.private_metrics import validate_private_metrics
from robot_auto_evolve.evaluation.scalars import SCALAR_METRICS, BenchmarkOutcome, compute_benchmark_scalar
from robot_auto_evolve.protocol import StrictSchemaError, decode_message
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeManifest, mapping_sha256

from .benchmark_models import BenchmarkEvaluationData, PublicDiagnostic
from .evidence import PublicStepEvidence


MAX_TRAJECTORY_TASK_UNITS = 80
MAX_TRAJECTORY_FRAME_EPISODES = 8
MAX_TRAJECTORY_FRAMES_PER_EPISODE = 2
MAX_TRAJECTORY_EVENTS_PER_EPISODE = 8
MAX_EVENT_DETAIL_BYTES = 160
MAX_DIAGNOSTIC_TEXT_BYTES = 1_024


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
    metrics: dict[str, bool | float] = {"success": bool(manifest.success)}
    required = _SCALAR_OUTCOME_METRICS.get(scalar_metric, frozenset())
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


def _bounded_text(value: str, maximum: int) -> str:
    payload = value.encode("utf-8")
    if len(payload) <= maximum:
        return value
    return payload[:maximum].decode("utf-8", errors="ignore")


def _png(rgb: Any) -> bytes:
    if (
        getattr(rgb, "dtype", None) is None
        or rgb.dtype.name != "uint8"
        or getattr(rgb, "ndim", None) != 3
        or rgb.shape[2] != 3
    ):
        raise StrictSchemaError("canonical benchmark diagnostic RGB differs")
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", compress_level=9)
    return output.getvalue()


def _trace_rank(manifest: EpisodeManifest) -> str:
    return hashlib.sha256(
        f"{manifest.key.task_id}\0{manifest.success}\0{manifest.key.artifact_id()}".encode()
    ).hexdigest()


def _trajectory_unit(manifest: EpisodeManifest, scalar_metric: str) -> str:
    if scalar_metric == "equal_track_task_macro_progress_score":
        track, _ = parse_vlabench_scenario(manifest.key.scenario_id)
        return f"{track}::{manifest.key.task_id}"
    return manifest.key.task_id


def _select_text_manifests(
    manifests: tuple[EpisodeManifest, ...],
    scalar_metric: str,
) -> tuple[EpisodeManifest, ...]:
    groups: dict[str, list[EpisodeManifest]] = {}
    for manifest in manifests:
        groups.setdefault(_trajectory_unit(manifest, scalar_metric), []).append(manifest)
    selected = []
    for unit in sorted(groups):
        selected.append(min(groups[unit], key=_trace_rank))
        if len(selected) == MAX_TRAJECTORY_TASK_UNITS:
            break
    return tuple(selected)


def _select_frame_manifests(manifests: tuple[EpisodeManifest, ...]) -> tuple[EpisodeManifest, ...]:
    groups: dict[tuple[str, bool], list[EpisodeManifest]] = {}
    for manifest in manifests:
        groups.setdefault((manifest.key.task_id, bool(manifest.success)), []).append(manifest)
    for rows in groups.values():
        rows.sort(key=_trace_rank)
    selected: list[EpisodeManifest] = []
    offset = 0
    while len(selected) < MAX_TRAJECTORY_FRAME_EPISODES:
        added = False
        for group in sorted(groups):
            rows = groups[group]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == MAX_TRAJECTORY_FRAME_EPISODES:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def _trajectory_diagnostics(
    root: Path,
    manifest: EpisodeManifest,
    outcome: BenchmarkOutcome,
    *,
    include_text: bool,
    include_frames: bool,
) -> tuple[PublicDiagnostic, ...]:
    trace_path = root / "trace.msgpack"
    if not trace_path.is_file() or trace_path.is_symlink():
        raise StrictSchemaError("canonical benchmark trace artifact differs")
    trace = decode_message(trace_path.read_bytes())
    if not isinstance(trace, dict) or set(trace) != {"outcome", "termination", "steps", "error"}:
        raise StrictSchemaError("canonical benchmark public trace fields differ")
    trace_outcome = EpisodeOutcome.from_mapping(trace["outcome"])
    if trace_outcome.key != manifest.key or trace_outcome.success is not manifest.success:
        raise StrictSchemaError("canonical benchmark public trace outcome differs")
    if trace["termination"] not in {"success", "horizon"} or trace["error"] is not None:
        raise StrictSchemaError("canonical benchmark public trace termination differs")
    if not isinstance(trace["steps"], list) or not trace["steps"]:
        raise StrictSchemaError("canonical benchmark public trace steps differ")
    steps = tuple(PublicStepEvidence.from_mapping(item) for item in trace["steps"])
    indices = tuple(item.observation.step_index for item in steps)
    if indices != tuple(sorted(set(indices))):
        raise StrictSchemaError("canonical benchmark public trace step order differs")
    if any(item.observation.episode_id != manifest.key.artifact_id() for item in steps):
        raise StrictSchemaError("canonical benchmark public trace episode identity differs")
    # An episode may legitimately carry more than one instruction (e.g. RoboCerebra's
    # anchor/resume protocol changes the subgoal mid-episode); record the ordered distinct
    # sequence rather than asserting a single constant instruction.
    ordered_instructions: list[str] = []
    for item in steps:
        text = item.observation.instruction
        if not ordered_instructions or ordered_instructions[-1] != text:
            ordered_instructions.append(text)

    events: list[AgentEvent] = []
    for step in steps:
        events.extend(step.events)
    event_lines = []
    for event in events[:MAX_TRAJECTORY_EVENTS_PER_EPISODE]:
        capability = "none" if event.capability is None else event.capability
        detail = _bounded_text(event.detail.replace("\n", " "), MAX_EVENT_DETAIL_BYTES)
        event_lines.append(
            f"event step={event.step_index} type={event.event_type} status={event.status} "
            f"capability={capability} detail={detail}"
        )
    instruction = _bounded_text(" -> ".join(ordered_instructions).replace("\n", " "), 256)
    lines = [
        f"instruction: {instruction}",
        f"termination: {trace['termination']}",
        f"success: {str(bool(manifest.success)).lower()}",
        f"observations: {len(steps)}",
        f"actions: {sum(item.action is not None for item in steps)}",
        f"events: {len(events)} (showing {len(event_lines)})",
        *event_lines,
    ]
    diagnostics = []
    if include_text:
        text_payload = _bounded_text("\n".join(lines), MAX_DIAGNOSTIC_TEXT_BYTES).encode("utf-8")
        diagnostics.append(PublicDiagnostic(outcome, "trajectory", "text/plain", text_payload))

    positions = (0, len(steps) - 1) if include_frames else ()
    seen_positions = set()
    for position in positions:
        if position in seen_positions or len(seen_positions) == MAX_TRAJECTORY_FRAMES_PER_EPISODE:
            continue
        seen_positions.add(position)
        observation = steps[position].observation
        if not observation.cameras:
            raise StrictSchemaError("canonical benchmark public trace has no RGB camera")
        camera_name = sorted(observation.cameras)[0]
        diagnostics.append(
            PublicDiagnostic(
                outcome,
                f"rgb-step-{observation.step_index:08d}",
                "image/png",
                _png(observation.cameras[camera_name].rgb),
            )
        )
    return tuple(diagnostics)


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
        output.mkdir(parents=True, exist_ok=False)
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
        manifests = []
        roots: dict[object, Path] = {}
        for key in self.plan.episodes:
            root = evaluation / "episodes" / key.artifact_id()
            manifest = EpisodeManifest.from_mapping(json.loads((root / "episode.json").read_text(encoding="utf-8")))
            if manifest.key != key or manifest.state != "complete" or manifest.success is None:
                raise StrictSchemaError("canonical benchmark episode differs from exact plan")
            rows.append(BenchmarkOutcome(key, self._metrics(root, manifest)))
            manifests.append(manifest)
            roots[key] = root
        outcomes_by_key = {item.key: item for item in rows}
        scalar = compute_benchmark_scalar(self.scalar_metric, rows)
        report_metrics = report.get("metrics")
        if (
            not isinstance(report_metrics, dict)
            or report_metrics.get("metric") != scalar.metric
            or report_metrics.get("score") != scalar.value
            or ("details" in report_metrics and report_metrics["details"] != scalar.details)
        ):
            raise StrictSchemaError("canonical benchmark report and route scalar differ")
        diagnostics = []
        selected_text = _select_text_manifests(tuple(manifests), self.scalar_metric)
        selected_frames = _select_frame_manifests(tuple(manifests))
        text_keys = {item.key for item in selected_text}
        frame_keys = {item.key for item in selected_frames}
        selected = tuple(
            item for item in manifests if item.key in text_keys or item.key in frame_keys
        )
        for manifest in selected:
            diagnostics.extend(
                _trajectory_diagnostics(
                    roots[manifest.key],
                    manifest,
                    outcomes_by_key[manifest.key],
                    include_text=manifest.key in text_keys,
                    include_frames=manifest.key in frame_keys,
                )
            )
        return BenchmarkEvaluationData(
            outcomes=tuple(rows),
            diagnostics=tuple(diagnostics),
            metadata={
                "canonical_report_sha256": mapping_sha256(report),
                "canonical_plan_sha256": self.plan.resolved_hash(),
            },
        )
