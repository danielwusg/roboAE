from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from robot_auto_evolve.evaluation import EpisodeOutcome, PairedBootstrapResult, TaskMacroMetrics
from robot_auto_evolve.protocol import StrictSchemaError


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    outcomes: tuple[EpisodeOutcome, ...]
    metadata: Mapping[str, Any]
    public_evidence_sha256: str | None = None
    public_evidence_episodes: int = 0

    def __post_init__(self) -> None:
        if self.split not in {"evolve", "selection", "transfer"}:
            raise StrictSchemaError("evaluation_result.split: unsupported split")
        outcomes = tuple(self.outcomes)
        if not outcomes or any(not isinstance(item, EpisodeOutcome) for item in outcomes):
            raise StrictSchemaError("evaluation_result.outcomes: expected nonempty EpisodeOutcome entries")
        if any(item.key.split != self.split for item in outcomes):
            raise StrictSchemaError("evaluation_result.outcomes: split mismatch")
        if len({item.key for item in outcomes}) != len(outcomes):
            raise StrictSchemaError("evaluation_result.outcomes: duplicate keys")
        if not isinstance(self.metadata, Mapping):
            raise StrictSchemaError("evaluation_result.metadata: expected mapping")
        if self.split == "evolve":
            digest = self.public_evidence_sha256
            if type(digest) is not str or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise StrictSchemaError("evaluation_result.public_evidence_sha256: expected sha256 for evolve")
            if self.public_evidence_episodes != len(outcomes):
                raise StrictSchemaError("evaluation_result.public_evidence_episodes: outcome count mismatch")
        elif self.public_evidence_sha256 is not None or self.public_evidence_episodes != 0:
            raise StrictSchemaError("evaluation_result: public evidence is allowed only for evolve")
        object.__setattr__(self, "outcomes", tuple(sorted(outcomes, key=lambda item: item.key)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, value: Any) -> "EvaluationResult":
        if not isinstance(value, Mapping) or set(value) != {
            "split",
            "outcomes",
            "metadata",
            "public_evidence_sha256",
            "public_evidence_episodes",
        }:
            raise StrictSchemaError("evaluation_result: invalid fields")
        return cls(
            split=value["split"],
            outcomes=tuple(EpisodeOutcome.from_mapping(item) for item in value["outcomes"]),
            metadata=value["metadata"],
            public_evidence_sha256=value["public_evidence_sha256"],
            public_evidence_episodes=value["public_evidence_episodes"],
        )

    @classmethod
    def load(cls, path: Path) -> "EvaluationResult":
        return cls.from_mapping(json.loads(path.read_text()))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "outcomes": [item.to_mapping() for item in self.outcomes],
            "metadata": dict(self.metadata),
            "public_evidence_sha256": self.public_evidence_sha256,
            "public_evidence_episodes": self.public_evidence_episodes,
        }


@dataclass(frozen=True)
class TransferEvaluation:
    baseline: EvaluationResult
    evolved: EvaluationResult
    baseline_metrics: TaskMacroMetrics
    evolved_metrics: TaskMacroMetrics
    paired_bootstrap: PairedBootstrapResult

    def __post_init__(self) -> None:
        if self.baseline.split != "transfer" or self.evolved.split != "transfer":
            raise StrictSchemaError("transfer_evaluation: expected transfer results")
        if {item.key for item in self.baseline.outcomes} != {item.key for item in self.evolved.outcomes}:
            raise StrictSchemaError("transfer_evaluation: baseline and evolved episode keys differ")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_mapping(),
            "evolved": self.evolved.to_mapping(),
            "baseline_metrics": self.baseline_metrics.to_mapping(),
            "evolved_metrics": self.evolved_metrics.to_mapping(),
            "paired_bootstrap": self.paired_bootstrap.to_mapping(),
            "affected_acceptance": False,
        }


class Evaluator(Protocol):
    def evaluate(self, scaffold_dir: Path, split: str, output_dir: Path) -> EvaluationResult: ...


class RevisionBackend(Protocol):
    def revise(
        self,
        prompt: str,
        candidate_dir: Path,
        log_dir: Path,
        attempt_index: int,
    ) -> None: ...
