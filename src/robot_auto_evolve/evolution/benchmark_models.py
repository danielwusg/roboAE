from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from robot_auto_evolve.evaluation.scalars import BenchmarkOutcome, BenchmarkScalar
from robot_auto_evolve.protocol import StrictSchemaError


MAX_BENCHMARK_OUTCOMES = 10_000


@dataclass(frozen=True)
class PublicDiagnostic:
    outcome: BenchmarkOutcome
    label: str
    media_type: str
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, BenchmarkOutcome):
            raise StrictSchemaError("public diagnostic outcome differs")
        if type(self.label) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.label) is None:
            raise StrictSchemaError("public diagnostic label differs")
        if self.media_type not in {"text/plain", "image/png"}:
            raise StrictSchemaError("public diagnostic media type differs")
        if type(self.payload) is not bytes:
            raise StrictSchemaError("public diagnostic payload must be bytes")

    def rank(self) -> str:
        return hashlib.sha256(
            (
                self.outcome.key.artifact_id()
                + "\0"
                + self.label
                + "\0"
                + self.media_type
                + "\0"
                + hashlib.sha256(self.payload).hexdigest()
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class BenchmarkEvaluationData:
    outcomes: tuple[BenchmarkOutcome, ...]
    diagnostics: tuple[PublicDiagnostic, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        outcomes = tuple(sorted(self.outcomes, key=lambda item: item.key))
        if not 1 <= len(outcomes) <= MAX_BENCHMARK_OUTCOMES:
            raise StrictSchemaError(f"benchmark evaluation requires 1..{MAX_BENCHMARK_OUTCOMES} outcomes")
        if any(not isinstance(item, BenchmarkOutcome) for item in outcomes) or len({item.key for item in outcomes}) != len(outcomes):
            raise StrictSchemaError("benchmark evaluation outcomes differ")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, PublicDiagnostic) for item in diagnostics):
            raise StrictSchemaError("benchmark evaluation diagnostics differ")
        keys = {item.key for item in outcomes}
        if any(item.outcome.key not in keys for item in diagnostics):
            raise StrictSchemaError("benchmark diagnostic references an unknown outcome")
        if not isinstance(self.metadata, Mapping):
            raise StrictSchemaError("benchmark evaluation metadata differs")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class BenchmarkEvaluationResult:
    plan_sha256: str
    scalar: BenchmarkScalar
    outcomes: tuple[BenchmarkOutcome, ...]
    metadata: Mapping[str, Any]
    evidence_sha256: str
    evidence_episodes: int

    def __post_init__(self) -> None:
        for name, value in (("plan_sha256", self.plan_sha256), ("evidence_sha256", self.evidence_sha256)):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise StrictSchemaError(f"benchmark result {name} differs")
        if not isinstance(self.scalar, BenchmarkScalar):
            raise StrictSchemaError("benchmark result scalar differs")
        outcomes = tuple(sorted(self.outcomes, key=lambda item: item.key))
        if not 1 <= len(outcomes) <= MAX_BENCHMARK_OUTCOMES or len({item.key for item in outcomes}) != len(outcomes):
            raise StrictSchemaError("benchmark result outcomes differ")
        if self.evidence_episodes != len(outcomes):
            raise StrictSchemaError("benchmark result evidence count differs")
        if not isinstance(self.metadata, Mapping):
            raise StrictSchemaError("benchmark result metadata differs")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkEvaluationResult":
        expected = {
            "plan_sha256",
            "scalar",
            "outcomes",
            "metadata",
            "evidence_sha256",
            "evidence_episodes",
        }
        if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value["outcomes"], list):
            raise StrictSchemaError("benchmark result fields differ")
        return cls(
            plan_sha256=value["plan_sha256"],
            scalar=BenchmarkScalar.from_mapping(value["scalar"]),
            outcomes=tuple(BenchmarkOutcome.from_mapping(item) for item in value["outcomes"]),
            metadata=value["metadata"],
            evidence_sha256=value["evidence_sha256"],
            evidence_episodes=value["evidence_episodes"],
        )

    @classmethod
    def load(cls, path: Path) -> "BenchmarkEvaluationResult":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "scalar": self.scalar.to_mapping(),
            "outcomes": [item.to_mapping() for item in self.outcomes],
            "metadata": dict(self.metadata),
            "evidence_sha256": self.evidence_sha256,
            "evidence_episodes": self.evidence_episodes,
        }


@dataclass(frozen=True)
class ScalarDecision:
    accepted: bool
    incumbent: float
    candidate: float
    delta: float
    reason: str

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise StrictSchemaError("scalar decision accepted differs")
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in (self.incumbent, self.candidate, self.delta)):
            raise StrictSchemaError("scalar decision value differs")
        expected = float(self.candidate) - float(self.incumbent)
        if not math.isclose(float(self.delta), expected, rel_tol=0.0, abs_tol=1e-15):
            raise StrictSchemaError("scalar decision delta differs")
        if self.accepted != (float(self.candidate) > float(self.incumbent)):
            raise StrictSchemaError("scalar decision is not strict improvement")
        if self.reason != ("strict_improvement" if self.accepted else "not_strictly_better"):
            raise StrictSchemaError("scalar decision reason differs")

    @classmethod
    def create(cls, incumbent: float, candidate: float) -> "ScalarDecision":
        accepted = candidate > incumbent
        return cls(
            accepted=accepted,
            incumbent=float(incumbent),
            candidate=float(candidate),
            delta=float(candidate) - float(incumbent),
            reason="strict_improvement" if accepted else "not_strictly_better",
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "ScalarDecision":
        if not isinstance(value, Mapping) or set(value) != {"accepted", "incumbent", "candidate", "delta", "reason"}:
            raise StrictSchemaError("scalar decision fields differ")
        return cls(**value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "incumbent": self.incumbent,
            "candidate": self.candidate,
            "delta": self.delta,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BenchmarkTransferComparison:
    baseline: BenchmarkEvaluationResult
    evolved: BenchmarkEvaluationResult

    def __post_init__(self) -> None:
        if self.baseline.plan_sha256 != self.evolved.plan_sha256:
            raise StrictSchemaError("transfer results use different plans")
        if self.baseline.scalar.metric != self.evolved.scalar.metric:
            raise StrictSchemaError("transfer results use different metrics")
        if {item.key for item in self.baseline.outcomes} != {item.key for item in self.evolved.outcomes}:
            raise StrictSchemaError("transfer results use different episode keys")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_mapping(),
            "evolved": self.evolved.to_mapping(),
            "delta": self.evolved.scalar.value - self.baseline.scalar.value,
            "affected_acceptance": False,
        }


class BenchmarkEvaluator(Protocol):
    def evaluate(self, scaffold_dir: Path, output_dir: Path) -> BenchmarkEvaluationData: ...


class RevisionBackend(Protocol):
    def revise(self, prompt: str, candidate_dir: Path, log_dir: Path, attempt_index: int) -> None: ...
