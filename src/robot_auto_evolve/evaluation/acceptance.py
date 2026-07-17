from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from robot_auto_evolve.protocol.schema import StrictSchemaError, integer, number
from robot_auto_evolve.provenance import EpisodeManifest

from .metrics import EpisodeOutcome, _outcomes, task_macro_success


@dataclass(frozen=True)
class AcceptanceConfig:
    bootstrap_resamples: int = 10000
    confidence_level: float = 0.95
    minimum_effect: float = 0.0
    maximum_regression_probability: float = 0.05
    maximum_task_regression: float = 0.0
    maximum_task_regression_probability: float = 0.05
    max_candidates: int = 1
    attempt_index: int = 1
    random_seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bootstrap_resamples",
            integer(self.bootstrap_resamples, "acceptance.bootstrap_resamples", minimum=100),
        )
        confidence = number(
            self.confidence_level,
            "acceptance.confidence_level",
            minimum=0.5,
            maximum=1.0,
        )
        if confidence >= 1:
            raise StrictSchemaError("acceptance.confidence_level: expected < 1")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "minimum_effect",
            number(self.minimum_effect, "acceptance.minimum_effect", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "maximum_regression_probability",
            number(
                self.maximum_regression_probability,
                "acceptance.maximum_regression_probability",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.maximum_regression_probability >= 1.0:
            raise StrictSchemaError("acceptance.maximum_regression_probability: expected < 1")
        object.__setattr__(
            self,
            "maximum_task_regression",
            number(
                self.maximum_task_regression,
                "acceptance.maximum_task_regression",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_task_regression_probability",
            number(
                self.maximum_task_regression_probability,
                "acceptance.maximum_task_regression_probability",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.maximum_task_regression_probability >= 1.0:
            raise StrictSchemaError("acceptance.maximum_task_regression_probability: expected < 1")
        maximum_candidates = integer(self.max_candidates, "acceptance.max_candidates", minimum=1)
        object.__setattr__(self, "max_candidates", maximum_candidates)
        object.__setattr__(
            self,
            "attempt_index",
            integer(
                self.attempt_index,
                "acceptance.attempt_index",
                minimum=1,
                maximum=maximum_candidates,
            ),
        )
        object.__setattr__(self, "random_seed", integer(self.random_seed, "acceptance.random_seed", minimum=0))


@dataclass(frozen=True)
class PairedBootstrapResult:
    observed_delta: float
    confidence_low: float
    confidence_high: float
    probability_improvement: float
    probability_regression: float
    probability_tie: float
    bootstrap_mean: float
    n_tasks: int
    n_pairs: int
    random_seed: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "observed_delta": self.observed_delta,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "probability_improvement": self.probability_improvement,
            "probability_regression": self.probability_regression,
            "probability_tie": self.probability_tie,
            "bootstrap_mean": self.bootstrap_mean,
            "n_tasks": self.n_tasks,
            "n_pairs": self.n_pairs,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class MultiplicityEvidence:
    familywise_confidence_level: float
    max_candidates: int
    attempt_index: int
    comparisons_per_candidate: int
    total_planned_comparisons: int
    per_comparison_alpha: float
    adjusted_confidence_level: float
    adjusted_overall_regression_probability: float
    adjusted_task_regression_probability: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "familywise_confidence_level": self.familywise_confidence_level,
            "max_candidates": self.max_candidates,
            "attempt_index": self.attempt_index,
            "comparisons_per_candidate": self.comparisons_per_candidate,
            "total_planned_comparisons": self.total_planned_comparisons,
            "per_comparison_alpha": self.per_comparison_alpha,
            "adjusted_confidence_level": self.adjusted_confidence_level,
            "adjusted_overall_regression_probability": self.adjusted_overall_regression_probability,
            "adjusted_task_regression_probability": self.adjusted_task_regression_probability,
        }


@dataclass(frozen=True)
class TaskNoninferiorityEvidence:
    task_id: str
    observed_delta: float
    confidence_low: float
    confidence_high: float
    probability_beyond_allowed_regression: float
    maximum_allowed_regression: float
    maximum_probability: float
    n_pairs: int
    accepted: bool
    reason: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "observed_delta": self.observed_delta,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "probability_beyond_allowed_regression": self.probability_beyond_allowed_regression,
            "maximum_allowed_regression": self.maximum_allowed_regression,
            "maximum_probability": self.maximum_probability,
            "n_pairs": self.n_pairs,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    incumbent_metric: float
    candidate_metric: float
    bootstrap: PairedBootstrapResult
    minimum_effect: float
    maximum_regression_probability: float
    maximum_task_regression: float
    maximum_task_regression_probability: float
    multiplicity: MultiplicityEvidence
    task_noninferiority: tuple[TaskNoninferiorityEvidence, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "incumbent_metric": self.incumbent_metric,
            "candidate_metric": self.candidate_metric,
            "bootstrap": self.bootstrap.to_mapping(),
            "minimum_effect": self.minimum_effect,
            "maximum_regression_probability": self.maximum_regression_probability,
            "maximum_task_regression": self.maximum_task_regression,
            "maximum_task_regression_probability": self.maximum_task_regression_probability,
            "multiplicity": self.multiplicity.to_mapping(),
            "task_noninferiority": [item.to_mapping() for item in self.task_noninferiority],
        }


def _paired(
    incumbent: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    candidate: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
) -> tuple[tuple[EpisodeOutcome, EpisodeOutcome], ...]:
    incumbent_rows = _outcomes(incumbent)
    candidate_rows = _outcomes(candidate)
    incumbent_by_key = {item.key: item for item in incumbent_rows}
    candidate_by_key = {item.key: item for item in candidate_rows}
    if set(incumbent_by_key) != set(candidate_by_key):
        missing = sorted(item.artifact_id() for item in set(incumbent_by_key) - set(candidate_by_key))
        unknown = sorted(item.artifact_id() for item in set(candidate_by_key) - set(incumbent_by_key))
        raise StrictSchemaError(f"paired outcomes differ: missing={missing}, unknown={unknown}")
    splits = {item.split for item in incumbent_by_key}
    if len(splits) != 1:
        raise StrictSchemaError("paired outcomes: expected one split")
    return tuple((incumbent_by_key[key], candidate_by_key[key]) for key in sorted(incumbent_by_key))


def _task_deltas(
    pairs: tuple[tuple[EpisodeOutcome, EpisodeOutcome], ...],
) -> dict[str, np.ndarray]:
    rows_by_task: dict[str, list[float]] = {}
    for incumbent_row, candidate_row in pairs:
        rows_by_task.setdefault(incumbent_row.key.task_id, []).append(
            float(candidate_row.success) - float(incumbent_row.success)
        )
    return {
        task: np.asarray(rows, dtype=np.float64)
        for task, rows in sorted(rows_by_task.items())
    }


def _multiplicity(config: AcceptanceConfig, n_tasks: int) -> MultiplicityEvidence:
    comparisons = n_tasks + 1
    total = config.max_candidates * comparisons
    alpha = (1.0 - config.confidence_level) / total
    return MultiplicityEvidence(
        familywise_confidence_level=config.confidence_level,
        max_candidates=config.max_candidates,
        attempt_index=config.attempt_index,
        comparisons_per_candidate=comparisons,
        total_planned_comparisons=total,
        per_comparison_alpha=alpha,
        adjusted_confidence_level=1.0 - alpha,
        adjusted_overall_regression_probability=min(config.maximum_regression_probability, alpha),
        adjusted_task_regression_probability=min(config.maximum_task_regression_probability, alpha),
    )


def _task_noninferiority(
    task_deltas: Mapping[str, np.ndarray],
    config: AcceptanceConfig,
    multiplicity: MultiplicityEvidence,
) -> tuple[TaskNoninferiorityEvidence, ...]:
    alpha = (1.0 - multiplicity.adjusted_confidence_level) / 2.0
    threshold = -config.maximum_task_regression
    evidence: list[TaskNoninferiorityEvidence] = []
    for task_index, (task, rows) in enumerate(task_deltas.items()):
        rng = np.random.default_rng(np.random.SeedSequence([config.random_seed, task_index + 1]))
        row_indices = rng.integers(0, len(rows), size=(config.bootstrap_resamples, len(rows)))
        samples = np.mean(rows[row_indices], axis=1)
        observed = float(np.mean(rows))
        confidence_low = float(np.quantile(samples, alpha))
        confidence_high = float(np.quantile(samples, 1.0 - alpha))
        probability = float(np.mean(samples < threshold))
        if observed < threshold:
            accepted = False
            reason = "observed_regression_exceeds_margin"
        elif confidence_low < threshold:
            accepted = False
            reason = "confidence_interval_exceeds_margin"
        elif probability > multiplicity.adjusted_task_regression_probability:
            accepted = False
            reason = "regression_probability_too_high"
        else:
            accepted = True
            reason = "noninferior"
        evidence.append(
            TaskNoninferiorityEvidence(
                task_id=task,
                observed_delta=observed,
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                probability_beyond_allowed_regression=probability,
                maximum_allowed_regression=config.maximum_task_regression,
                maximum_probability=multiplicity.adjusted_task_regression_probability,
                n_pairs=len(rows),
                accepted=accepted,
                reason=reason,
            )
        )
    return tuple(evidence)


def paired_hierarchical_bootstrap(
    incumbent: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    candidate: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> PairedBootstrapResult:
    config = AcceptanceConfig(
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    pairs = _paired(incumbent, candidate)
    task_deltas = _task_deltas(pairs)
    tasks = tuple(task_deltas)
    observed = float(np.mean([np.mean(task_deltas[task]) for task in tasks]))
    rng = np.random.default_rng(config.random_seed)
    samples = np.empty(config.bootstrap_resamples, dtype=np.float64)
    for bootstrap_index in range(config.bootstrap_resamples):
        sampled_task_indices = rng.integers(0, len(tasks), size=len(tasks))
        task_means = np.empty(len(tasks), dtype=np.float64)
        for output_index, task_index in enumerate(sampled_task_indices):
            rows = task_deltas[tasks[int(task_index)]]
            row_indices = rng.integers(0, len(rows), size=len(rows))
            task_means[output_index] = float(np.mean(rows[row_indices]))
        samples[bootstrap_index] = float(np.mean(task_means))
    alpha = (1.0 - config.confidence_level) / 2.0
    return PairedBootstrapResult(
        observed_delta=observed,
        confidence_low=float(np.quantile(samples, alpha)),
        confidence_high=float(np.quantile(samples, 1.0 - alpha)),
        probability_improvement=float(np.mean(samples > 0.0)),
        probability_regression=float(np.mean(samples < 0.0)),
        probability_tie=float(np.mean(samples == 0.0)),
        bootstrap_mean=float(np.mean(samples)),
        n_tasks=len(tasks),
        n_pairs=len(pairs),
        random_seed=config.random_seed,
    )


def decide_acceptance(
    incumbent: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    candidate: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    config: AcceptanceConfig,
) -> AcceptanceDecision:
    if not isinstance(config, AcceptanceConfig):
        raise StrictSchemaError("acceptance.config: expected AcceptanceConfig")
    incumbent_rows = _outcomes(incumbent)
    candidate_rows = _outcomes(candidate)
    pairs = _paired(incumbent_rows, candidate_rows)
    task_deltas = _task_deltas(pairs)
    multiplicity = _multiplicity(config, len(task_deltas))
    bootstrap = paired_hierarchical_bootstrap(
        incumbent_rows,
        candidate_rows,
        bootstrap_resamples=config.bootstrap_resamples,
        confidence_level=multiplicity.adjusted_confidence_level,
        random_seed=config.random_seed,
    )
    task_noninferiority = _task_noninferiority(task_deltas, config, multiplicity)
    incumbent_metric = task_macro_success(incumbent_rows)
    candidate_metric = task_macro_success(candidate_rows)
    if candidate_metric <= incumbent_metric:
        accepted = False
        reason = "not_strictly_better"
    elif bootstrap.observed_delta <= config.minimum_effect:
        accepted = False
        reason = "effect_below_minimum"
    elif bootstrap.confidence_low <= config.minimum_effect:
        accepted = False
        reason = "confidence_interval_crosses_minimum"
    elif bootstrap.probability_regression > multiplicity.adjusted_overall_regression_probability:
        accepted = False
        reason = "regression_probability_too_high"
    elif any(not item.accepted for item in task_noninferiority):
        accepted = False
        reason = "task_noninferiority_failed"
    else:
        accepted = True
        reason = "accepted"
    return AcceptanceDecision(
        accepted=accepted,
        reason=reason,
        incumbent_metric=incumbent_metric,
        candidate_metric=candidate_metric,
        bootstrap=bootstrap,
        minimum_effect=config.minimum_effect,
        maximum_regression_probability=config.maximum_regression_probability,
        maximum_task_regression=config.maximum_task_regression,
        maximum_task_regression_probability=config.maximum_task_regression_probability,
        multiplicity=multiplicity,
        task_noninferiority=task_noninferiority,
    )


def paired_acceptance(
    incumbent: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    candidate: Iterable[EpisodeOutcome | EpisodeManifest | Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10000,
    confidence_level: float = 0.95,
    minimum_effect: float = 0.0,
    maximum_regression_probability: float = 0.05,
    maximum_task_regression: float = 0.0,
    maximum_task_regression_probability: float = 0.05,
    max_candidates: int = 1,
    attempt_index: int = 1,
    random_seed: int = 0,
) -> AcceptanceDecision:
    return decide_acceptance(
        incumbent,
        candidate,
        AcceptanceConfig(
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            minimum_effect=minimum_effect,
            maximum_regression_probability=maximum_regression_probability,
            maximum_task_regression=maximum_task_regression,
            maximum_task_regression_probability=maximum_task_regression_probability,
            max_candidates=max_candidates,
            attempt_index=attempt_index,
            random_seed=random_seed,
        ),
    )
