from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from robot_auto_evolve.evaluation import (
    AcceptanceConfig,
    compute_task_macro_metrics,
    decide_acceptance,
    paired_hierarchical_bootstrap,
)
from robot_auto_evolve.protocol import StrictSchemaError

from .evidence import PublicEvolutionEvidence
from .hashing import EditablePolicy, FrozenHashGuard, file_sha256, mapping_sha256, tree_hashes
from .models import EvaluationResult, Evaluator, RevisionBackend, TransferEvaluation


PENDING_COMMIT_NAME = "pending_commit.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _make_read_only(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.chmod(0o444)
        elif item.is_dir():
            item.chmod(0o555)
    path.chmod(0o555)


def _seal_directory(path: Path) -> dict[str, Any]:
    if (path / "seal.json").exists():
        raise RuntimeError(f"directory is already sealed: {path}")
    hashes = tree_hashes(path)
    seal = {
        "schema_version": 1,
        "created_ns": time.time_ns(),
        "files": hashes,
        "content_sha256": mapping_sha256(hashes),
    }
    _atomic_json(path / "seal.json", seal)
    _make_read_only(path)
    return seal


def _verify_seal(path: Path) -> None:
    seal = _load_json(path / "seal.json")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"schema_version", "created_ns", "files", "content_sha256"}
        or seal["schema_version"] != 1
        or type(seal["created_ns"]) is not int
        or not isinstance(seal["files"], Mapping)
        or type(seal["content_sha256"]) is not str
    ):
        raise StrictSchemaError(f"invalid seal at {path}")
    actual = tree_hashes(path)
    actual.pop("seal.json", None)
    if actual != seal["files"] or mapping_sha256(actual) != seal["content_sha256"]:
        raise RuntimeError(f"immutable directory changed after sealing: {path}")


def _finish_seal(path: Path) -> None:
    if (path / "seal.json").exists():
        _verify_seal(path)
        _make_read_only(path)
    else:
        _seal_directory(path)


class EvolutionDriver:
    def __init__(
        self,
        seed_scaffold: Path,
        run_dir: Path,
        evaluator: Evaluator,
        revision_backend: RevisionBackend,
        acceptance: AcceptanceConfig,
        frozen_paths: tuple[Path, ...],
    ) -> None:
        self.seed_scaffold = Path(seed_scaffold).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.evaluator = evaluator
        self.revision_backend = revision_backend
        self.acceptance = acceptance
        self.editable = EditablePolicy()
        if any(self.run_dir == path.resolve() or path.resolve() in self.run_dir.parents for path in frozen_paths):
            raise StrictSchemaError("run directory must not be inside a frozen hash root")
        self.frozen_paths = tuple(Path(path).resolve() for path in frozen_paths)

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def pending_commit_path(self) -> Path:
        return self.run_dir / PENDING_COMMIT_NAME

    def _acceptance_mapping(self) -> dict[str, Any]:
        return {
            "bootstrap_resamples": self.acceptance.bootstrap_resamples,
            "confidence_level": self.acceptance.confidence_level,
            "minimum_effect": self.acceptance.minimum_effect,
            "maximum_regression_probability": self.acceptance.maximum_regression_probability,
            "maximum_task_regression": self.acceptance.maximum_task_regression,
            "maximum_task_regression_probability": self.acceptance.maximum_task_regression_probability,
            "max_candidates": self.acceptance.max_candidates,
            "random_seed": self.acceptance.random_seed,
        }

    def _write_state(self, state: Mapping[str, Any]) -> None:
        stable = self._validate_state_mapping(state)
        stable["state_sha256"] = mapping_sha256(stable)
        _atomic_json(self.state_path, stable)

    @staticmethod
    def _validate_state_mapping(state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise StrictSchemaError("evolution state is invalid")
        stable = dict(state)
        common = {"schema_version", "phase", "next_attempt", "incumbent", "created_ns"}
        allowed = common | ({"frozen_scaffold_sha256"} if stable.get("phase") == "frozen" else set())
        if set(stable) != allowed or stable.get("schema_version") != 1 or stable.get("phase") not in {"active", "frozen"}:
            raise StrictSchemaError("evolution state fields are invalid")
        if type(stable["next_attempt"]) is not int or stable["next_attempt"] < 1:
            raise StrictSchemaError("evolution state next_attempt is invalid")
        if type(stable["incumbent"]) is not str or not stable["incumbent"] or type(stable["created_ns"]) is not int:
            raise StrictSchemaError("evolution state value type is invalid")
        if stable["phase"] == "frozen":
            digest = stable["frozen_scaffold_sha256"]
            if type(digest) is not str or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise StrictSchemaError("evolution frozen scaffold hash is invalid")
        return stable

    def _load_state(self) -> dict[str, Any]:
        state = _load_json(self.state_path)
        if not isinstance(state, Mapping) or "state_sha256" not in state:
            raise StrictSchemaError("evolution state is invalid")
        stable = dict(state)
        digest = stable.pop("state_sha256")
        if type(digest) is not str or mapping_sha256(stable) != digest:
            raise StrictSchemaError("evolution state checksum mismatch")
        return self._validate_state_mapping(stable)

    def _load_pending_commit(self) -> dict[str, Any]:
        value = _load_json(self.pending_commit_path)
        if not isinstance(value, Mapping) or "journal_sha256" not in value:
            raise StrictSchemaError("pending commit journal is invalid")
        stable = dict(value)
        digest = stable.pop("journal_sha256")
        expected_fields = {
            "schema_version",
            "kind",
            "attempt",
            "source",
            "target",
            "pre_state",
            "post_state",
            "created_ns",
        }
        if set(stable) != expected_fields or stable.get("schema_version") != 1:
            raise StrictSchemaError("pending commit journal fields are invalid")
        if type(digest) is not str or mapping_sha256(stable) != digest:
            raise StrictSchemaError("pending commit journal checksum mismatch")
        kind = stable["kind"]
        attempt = stable["attempt"]
        if kind not in {"baseline", "attempt", "freeze", "transfer"}:
            raise StrictSchemaError("pending commit kind is invalid")
        if kind == "attempt":
            if type(attempt) is not int or attempt < 1:
                raise StrictSchemaError("pending attempt index is invalid")
            expected_source = f"attempts/.{attempt:04d}-staging"
            expected_target = f"attempts/{attempt:04d}"
        else:
            if attempt is not None:
                raise StrictSchemaError("pending non-attempt commit has an attempt index")
            expected_source, expected_target = {
                "baseline": (".baseline-staging", "baseline"),
                "freeze": (".frozen-staging", "frozen"),
                "transfer": (".transfer-staging", "transfer"),
            }[kind]
        if stable["source"] != expected_source or stable["target"] != expected_target:
            raise StrictSchemaError("pending commit paths are invalid")
        if type(stable["created_ns"]) is not int or stable["created_ns"] < 0:
            raise StrictSchemaError("pending commit timestamp is invalid")
        if stable["pre_state"] is not None:
            stable["pre_state"] = self._validate_state_mapping(stable["pre_state"])
        stable["post_state"] = self._validate_state_mapping(stable["post_state"])
        return stable

    def _begin_commit(
        self,
        kind: str,
        source: Path,
        target: Path,
        pre_state: Mapping[str, Any] | None,
        post_state: Mapping[str, Any],
        *,
        attempt: int | None = None,
    ) -> None:
        if self.pending_commit_path.exists():
            raise RuntimeError("another directory commit is pending")
        source = source.resolve()
        target = target.resolve()
        try:
            source_relative = source.relative_to(self.run_dir).as_posix()
            target_relative = target.relative_to(self.run_dir).as_posix()
        except ValueError as exc:
            raise StrictSchemaError("commit path escapes run directory") from exc
        if not source.is_dir() or target.exists():
            raise RuntimeError("commit source or target state is invalid")
        stable = {
            "schema_version": 1,
            "kind": kind,
            "attempt": attempt,
            "source": source_relative,
            "target": target_relative,
            "pre_state": None if pre_state is None else self._validate_state_mapping(pre_state),
            "post_state": self._validate_state_mapping(post_state),
            "created_ns": time.time_ns(),
        }
        stable["journal_sha256"] = mapping_sha256(stable)
        _atomic_json(self.pending_commit_path, stable)
        self._load_pending_commit()

    def _clear_pending_commit(self) -> None:
        self.pending_commit_path.unlink()
        _fsync_directory(self.run_dir)

    def _saved_evaluation(self, root: Path, split: str) -> EvaluationResult:
        result = EvaluationResult.load(root / f"{split}_result.json")
        if result.split != split:
            raise StrictSchemaError(f"saved {split} evaluation has the wrong split")
        if split == "evolve":
            evidence = PublicEvolutionEvidence.load(root / "evolve" / "public_evidence")
            evidence.validate_outcomes(result.outcomes)
            if evidence.bundle_sha256 != result.public_evidence_sha256:
                raise StrictSchemaError("saved evolve evidence hash differs")
        return result

    def _load_transfer_result(self, root: Path) -> TransferEvaluation:
        _verify_seal(root)
        stored = _load_json(root / "transfer_comparison.json")
        expected = {
            "baseline",
            "evolved",
            "baseline_metrics",
            "evolved_metrics",
            "paired_bootstrap",
            "affected_acceptance",
        }
        if not isinstance(stored, Mapping) or set(stored) != expected:
            raise StrictSchemaError("stored transfer comparison fields are invalid")
        baseline = EvaluationResult.from_mapping(stored["baseline"])
        evolved = EvaluationResult.from_mapping(stored["evolved"])
        baseline_metrics = compute_task_macro_metrics(baseline.outcomes)
        evolved_metrics = compute_task_macro_metrics(evolved.outcomes)
        comparison = paired_hierarchical_bootstrap(
            baseline.outcomes,
            evolved.outcomes,
            bootstrap_resamples=self.acceptance.bootstrap_resamples,
            confidence_level=self.acceptance.confidence_level,
            random_seed=self.acceptance.random_seed,
        )
        result = TransferEvaluation(baseline, evolved, baseline_metrics, evolved_metrics, comparison)
        if dict(stored) != result.to_mapping():
            raise StrictSchemaError("stored transfer comparison differs from recomputation")
        return result

    def _validate_commit_transition(self, journal: Mapping[str, Any], target: Path) -> None:
        kind = journal["kind"]
        pre = journal["pre_state"]
        post = journal["post_state"]
        if kind == "baseline":
            marker = _load_json(target / "baseline.json")
            if (
                pre is not None
                or not isinstance(marker, Mapping)
                or set(marker) != {"created_ns", "kind"}
                or marker["kind"] != "baseline"
                or type(marker["created_ns"]) is not int
                or post["phase"] != "active"
                or post["next_attempt"] != 1
                or post["incumbent"] != "baseline"
            ):
                raise StrictSchemaError("baseline commit transition is invalid")
            self.editable.validate_tree(target / "scaffold")
            self._saved_evaluation(target, "evolve")
            self._saved_evaluation(target, "selection")
            return
        if pre is None:
            raise StrictSchemaError("non-baseline commit lacks a pre-state")
        if kind == "attempt":
            index = journal["attempt"]
            expected = dict(pre)
            if pre["phase"] != "active" or pre["next_attempt"] != index:
                raise StrictSchemaError("attempt pre-state is invalid")
            candidate_selection = self._saved_evaluation(target, "selection")
            incumbent_selection = self._result(str(pre["incumbent"]), "selection")
            recomputed_decision = decide_acceptance(
                incumbent_selection.outcomes,
                candidate_selection.outcomes,
                replace(self.acceptance, attempt_index=index),
            )
            decision = _load_json(target / "decision.json")
            if not isinstance(decision, Mapping) or dict(decision) != recomputed_decision.to_mapping():
                raise StrictSchemaError("attempt decision differs from recomputation")
            expected["next_attempt"] = index + 1
            if recomputed_decision.accepted:
                expected["incumbent"] = f"attempts/{index:04d}"
            if post != expected:
                raise StrictSchemaError("attempt post-state is invalid")
            self.editable.validate_tree(target / "scaffold")
            self._saved_evaluation(target, "evolve")
            return
        if kind == "freeze":
            marker = _load_json(target / "FROZEN.json")
            scaffold_hash = mapping_sha256(tree_hashes(target / "scaffold"))
            expected = {**pre, "phase": "frozen", "frozen_scaffold_sha256": scaffold_hash}
            if (
                pre["phase"] != "active"
                or post != expected
                or not isinstance(marker, Mapping)
                or set(marker) != {"schema_version", "incumbent", "scaffold_sha256", "frozen_ns"}
                or marker["schema_version"] != 1
                or marker["incumbent"] != pre["incumbent"]
                or marker["scaffold_sha256"] != scaffold_hash
                or type(marker["frozen_ns"]) is not int
            ):
                raise StrictSchemaError("freeze commit transition is invalid")
            self.editable.validate_tree(target / "scaffold")
            return
        if pre != post or pre["phase"] != "frozen":
            raise StrictSchemaError("transfer commit state is invalid")
        self._load_transfer_result(target)

    def _finish_pending_commit(self) -> dict[str, Any]:
        journal = self._load_pending_commit()
        pre = journal["pre_state"]
        post = journal["post_state"]
        current = self._load_state() if self.state_path.exists() else None
        if current != pre and current != post:
            raise RuntimeError("evolution state differs from pending commit")
        source = (self.run_dir / journal["source"]).resolve()
        target = (self.run_dir / journal["target"]).resolve()
        if source.exists() == target.exists():
            raise RuntimeError("pending commit requires exactly one source or target")
        if source.exists():
            _finish_seal(source)
            os.rename(source, target)
            _fsync_directory(target.parent)
        _verify_seal(target)
        self._validate_commit_transition(journal, target)
        if current != post:
            self._write_state(post)
        self._clear_pending_commit()
        return dict(post)

    def _recover_pending_commit(self) -> str | None:
        if not self.pending_commit_path.exists():
            return None
        kind = self._load_pending_commit()["kind"]
        self._finish_pending_commit()
        return str(kind)

    def _guard(self) -> FrozenHashGuard:
        return FrozenHashGuard.from_manifest(_load_json(self.run_dir / "frozen_hashes.json"))

    def _reference(self, name: str) -> Path:
        path = (self.run_dir / name).resolve()
        if self.run_dir not in path.parents:
            raise StrictSchemaError("incumbent reference escapes run directory")
        _verify_seal(path)
        return path

    def _validate_run_configuration(self) -> FrozenHashGuard:
        guard = self._guard()
        guard.verify()
        run_config = _load_json(self.run_dir / "run_config.json")
        if not isinstance(run_config, Mapping):
            raise StrictSchemaError("run configuration is invalid")
        if run_config.get("editable_files") != list(self.editable.allowed):
            raise RuntimeError("editable allowlist differs from initialized run")
        if run_config.get("acceptance") != self._acceptance_mapping():
            raise RuntimeError("acceptance configuration differs from initialized run")
        return guard

    def _validate_frozen_state(self, state: Mapping[str, Any]) -> Path:
        checked = self._validate_state_mapping(state)
        if checked["phase"] != "frozen":
            raise StrictSchemaError("frozen validation requires frozen state")
        target = self.run_dir / "frozen"
        if not target.is_dir():
            raise RuntimeError("frozen state has no frozen directory")
        _verify_seal(target)
        marker = _load_json(target / "FROZEN.json")
        actual = mapping_sha256(tree_hashes(target / "scaffold"))
        if (
            not isinstance(marker, Mapping)
            or set(marker) != {"schema_version", "incumbent", "scaffold_sha256", "frozen_ns"}
            or marker["schema_version"] != 1
            or marker["incumbent"] != checked["incumbent"]
            or marker["scaffold_sha256"] != actual
            or checked["frozen_scaffold_sha256"] != actual
            or type(marker["frozen_ns"]) is not int
        ):
            raise RuntimeError("frozen directory differs from frozen state")
        self.editable.validate_tree(target / "scaffold")
        return target

    def _archive_auxiliary_staging(self, staging: Path, label: str, phase: str, error: str) -> Path:
        failures = self.run_dir / "failures"
        sequence = 1
        while (
            (failures / f"{label}-failure-{sequence:04d}").exists()
            or (failures / f".{label}-failure-{sequence:04d}-staging").exists()
        ):
            sequence += 1
        wrapper = failures / f".{label}-failure-{sequence:04d}-staging"
        wrapper.mkdir()
        source_parent = staging.parent
        os.rename(staging, wrapper / "payload")
        _fsync_directory(source_parent)
        _fsync_directory(wrapper)
        _atomic_json(
            wrapper / "failure.json",
            {
                "schema_version": 1,
                "label": label,
                "phase": phase,
                "error": error,
                "archived_ns": time.time_ns(),
            },
        )
        _seal_directory(wrapper)
        target = failures / f"{label}-failure-{sequence:04d}"
        os.rename(wrapper, target)
        _fsync_directory(failures)
        return target

    def _prepare_baseline(self, guard: FrozenHashGuard) -> None:
        staging = self.run_dir / ".baseline-staging"
        if staging.exists():
            self._archive_auxiliary_staging(
                staging,
                "baseline",
                "interrupted",
                "uncommitted baseline staging found before retry",
            )
        staging.mkdir()
        try:
            shutil.copytree(self.seed_scaffold, staging / "scaffold")
            evolve = self._evaluate(staging / "scaffold", "evolve", staging / "evolve")
            selection = self._evaluate(staging / "scaffold", "selection", staging / "selection")
            _atomic_json(staging / "evolve_result.json", evolve.to_mapping())
            _atomic_json(staging / "selection_result.json", selection.to_mapping())
            _atomic_json(staging / "baseline.json", {"created_ns": time.time_ns(), "kind": "baseline"})
            guard.verify()
            post_state = {
                "schema_version": 1,
                "phase": "active",
                "next_attempt": 1,
                "incumbent": "baseline",
                "created_ns": time.time_ns(),
            }
            self._begin_commit(
                "baseline",
                staging,
                self.run_dir / "baseline",
                None,
                post_state,
            )
            self._finish_pending_commit()
        except BaseException as exc:
            if staging.exists() and not self.pending_commit_path.exists():
                self._archive_auxiliary_staging(
                    staging,
                    "baseline",
                    "evaluation",
                    f"{type(exc).__name__}: {exc}",
                )
            raise

    def initialize(self) -> None:
        if self.run_dir.exists():
            raise FileExistsError(self.run_dir)
        self.editable.validate_tree(self.seed_scaffold)
        guard = FrozenHashGuard(self.frozen_paths)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "attempts").mkdir()
        (self.run_dir / "failures").mkdir()
        _atomic_json(self.run_dir / "frozen_hashes.json", guard.to_mapping())
        _atomic_json(
            self.run_dir / "run_config.json",
            {
                "schema_version": 1,
                "editable_files": list(self.editable.allowed),
                "seed_hashes": tree_hashes(self.seed_scaffold),
                "acceptance": self._acceptance_mapping(),
            },
        )
        (self.run_dir / "run_config.json").chmod(0o444)
        _fsync_directory(self.run_dir)
        self._prepare_baseline(guard)

    def resume(self) -> None:
        guard = self._validate_run_configuration()
        self._recover_pending_commit()
        if not self.state_path.exists():
            if (self.run_dir / "baseline").exists():
                raise RuntimeError("committed baseline has no recoverable journal or state")
            self._prepare_baseline(guard)
        state = self._load_state()
        if state["phase"] == "frozen":
            self._validate_frozen_state(state)
        elif (self.run_dir / "frozen").exists():
            raise RuntimeError("active state has an unjournaled frozen directory")
        self._reference("baseline")
        for index in range(1, state["next_attempt"]):
            self._reference(f"attempts/{index:04d}")
        for failure in sorted((self.run_dir / "failures").iterdir()):
            if failure.is_dir():
                _verify_seal(failure)
        staging = self.run_dir / "attempts" / f".{state['next_attempt']:04d}-staging"
        if staging.exists():
            self._archive_failure(staging, state["next_attempt"], "interrupted", "staging found during resume")
        for staging, label in (
            (self.run_dir / ".frozen-staging", "freeze"),
            (self.run_dir / ".transfer-staging", "transfer"),
        ):
            if staging.exists():
                self._archive_auxiliary_staging(
                    staging,
                    label,
                    "interrupted",
                    f"uncommitted {label} staging found during resume",
                )

    def _evaluate(self, scaffold: Path, split: str, output: Path) -> EvaluationResult:
        result = self.evaluator.evaluate(scaffold, split, output)
        evidence_path = output / "public_evidence"
        legacy_evidence_path = output / "public_evidence.msgpack"
        if split == "evolve":
            if legacy_evidence_path.exists():
                raise StrictSchemaError("legacy full-frame public evidence is forbidden")
            evidence = PublicEvolutionEvidence.load(evidence_path)
            if evidence.bundle_sha256 != result.public_evidence_sha256:
                raise StrictSchemaError("evolve public evidence has the wrong bundle hash")
            evidence.validate_outcomes(result.outcomes)
            if len(evidence.episodes) != result.public_evidence_episodes:
                raise StrictSchemaError("evolve public evidence count mismatch")
        elif evidence_path.exists() or legacy_evidence_path.exists():
            raise StrictSchemaError("public evidence is forbidden for selection and transfer")
        return result

    def _archive_failure(self, staging: Path, attempt_index: int, phase: str, error: str) -> Path:
        if (staging / "seal.json").exists():
            return self._archive_auxiliary_staging(
                staging,
                f"attempt-{attempt_index:04d}",
                phase,
                error,
            )
        failures = self.run_dir / "failures"
        sequence = 1
        while (
            (failures / f"attempt-{attempt_index:04d}-failure-{sequence:04d}").exists()
            or (failures / f".attempt-{attempt_index:04d}-failure-{sequence:04d}-staging").exists()
        ):
            sequence += 1
        failure_staging = failures / f".attempt-{attempt_index:04d}-failure-{sequence:04d}-staging"
        source_parent = staging.parent
        os.rename(staging, failure_staging)
        _fsync_directory(source_parent)
        _fsync_directory(failures)
        _atomic_json(
            failure_staging / "failure.json",
            {
                "schema_version": 1,
                "attempt": attempt_index,
                "phase": phase,
                "error": error,
                "archived_ns": time.time_ns(),
            },
        )
        _seal_directory(failure_staging)
        target = failures / f"attempt-{attempt_index:04d}-failure-{sequence:04d}"
        os.rename(failure_staging, target)
        _fsync_directory(failures)
        return target

    def _result(self, reference: str, split: str) -> EvaluationResult:
        return EvaluationResult.load(self._reference(reference) / f"{split}_result.json")

    def _revision_material(
        self,
        state: Mapping[str, Any],
        attempt_index: int,
        public_input_path: Path,
        public_evidence_path: Path,
        scaffold_path: Path,
    ) -> tuple[str, str, PublicEvolutionEvidence]:
        incumbent_root = self._reference(str(state["incumbent"]))
        incumbent = EvaluationResult.load(incumbent_root / "evolve_result.json")
        metrics = compute_task_macro_metrics(incumbent.outcomes)
        evidence = PublicEvolutionEvidence.load(public_evidence_path)
        evidence.validate_outcomes(incumbent.outcomes)
        attempt_root = scaffold_path.parent.parent
        if public_input_path.parent != attempt_root or public_evidence_path.parent != attempt_root:
            raise StrictSchemaError("revision material paths must share one attempt root")
        public_input = (
            f"Attempt: {attempt_index}\n"
            f"Evolve task-macro success: {metrics.macro_success:.8f}\n"
            f"Evolve episodes: {metrics.n_episodes}\n"
            "Paths below are relative to the directory containing this file.\n"
            "VALIDATED PUBLIC EVOLVE EVIDENCE BUNDLE: public_evidence\n"
            f"Bundle SHA-256: {evidence.bundle_sha256}\n"
            "Index: public_evidence/index.json\n"
            "Manifest: public_evidence/manifest.json\n"
            f"Diagnostic frames: {len(evidence.manifest['frames'])}\n"
        )
        if len(public_input.encode()) > 8192:
            raise RuntimeError("public evidence locator exceeds bounded input size")
        prompt = (
            "Revise only scaffold.py. Do not create, rename, or delete files. "
            "The simulator, evaluator, policy, services, success checker, selection split, and transfer split are frozen. "
            "You may use only rendered RGB-D, calibrated cameras, robot proprioception, language, and public timing IDs. "
            "Use only the bounded post-evaluation outcomes in this bundle; do not seek live reward or success signals, simulator state, object or goal poses, simulator segmentation, or expert actions. "
            "All following paths are relative to the scaffold working directory. "
            "Read the complete validated public input at ../public_input.txt. "
            "Read ../public_evidence/manifest.json and the complete compact evidence index at ../public_evidence/index.json, then inspect only the diagnostic PNGs it references. "
            "Read the current scaffold at scaffold.py. Diagnose it and implement one focused revision."
        )
        if len(prompt.encode()) > 8192:
            raise RuntimeError("revision instruction exceeds bounded argv size")
        return prompt, public_input, evidence

    def _run_attempt(self, state: dict[str, Any]) -> dict[str, Any]:
        index = int(state["next_attempt"])
        staging = self.run_dir / "attempts" / f".{index:04d}-staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        phase = "copy"
        try:
            incumbent = self._reference(str(state["incumbent"]))
            shutil.copytree(incumbent / "scaffold", staging / "scaffold")
            (staging / "scaffold" / "scaffold.py").chmod(0o600)
            public_evidence_path = staging / "public_evidence"
            shutil.copytree(incumbent / "evolve" / "public_evidence", public_evidence_path)
            public_input_path = staging / "public_input.txt"
            scaffold_path = staging / "scaffold" / "scaffold.py"
            prompt, public_input, public_evidence = self._revision_material(
                state,
                index,
                public_input_path,
                public_evidence_path,
                scaffold_path,
            )
            public_input_path.write_text(public_input, encoding="utf-8")
            public_input_path.chmod(0o444)
            (staging / "revision_prompt.txt").write_text(prompt, encoding="utf-8")
            _atomic_json(
                staging / "revision_request.json",
                {
                    "attempt": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "public_input_sha256": file_sha256(public_input_path),
                    "public_input_bytes": len(public_input.encode()),
                    "public_evidence_sha256": public_evidence.bundle_sha256,
                    "public_evidence_payload_bytes": public_evidence.payload_bytes,
                    "public_evidence_index_bytes": public_evidence.manifest["index"]["size_bytes"],
                    "public_evidence_frame_bytes": sum(item["size_bytes"] for item in public_evidence.manifest["frames"]),
                },
            )
            phase = "revision"
            self.revision_backend.revise(prompt, staging / "scaffold", staging / "revision_logs", index)
            candidate_hashes = self.editable.validate_revision(incumbent / "scaffold", staging / "scaffold")
            self._guard().verify()
            phase = "evolve_evaluation"
            evolve = self._evaluate(staging / "scaffold", "evolve", staging / "evolve")
            phase = "selection_evaluation"
            selection = self._evaluate(staging / "scaffold", "selection", staging / "selection")
            incumbent_selection = self._result(str(state["incumbent"]), "selection")
            decision = decide_acceptance(
                incumbent_selection.outcomes,
                selection.outcomes,
                replace(self.acceptance, attempt_index=index),
            )
            _atomic_json(staging / "evolve_result.json", evolve.to_mapping())
            _atomic_json(staging / "selection_result.json", selection.to_mapping())
            _atomic_json(staging / "decision.json", decision.to_mapping())
            _atomic_json(staging / "candidate_hashes.json", candidate_hashes)
            self._guard().verify()
            target = self.run_dir / "attempts" / f"{index:04d}"
            post_state = dict(state)
            post_state["next_attempt"] = index + 1
            if decision.accepted:
                post_state["incumbent"] = f"attempts/{index:04d}"
            phase = "commit"
            self._begin_commit(
                "attempt",
                staging,
                target,
                state,
                post_state,
                attempt=index,
            )
            return self._finish_pending_commit()
        except BaseException as exc:
            if staging.exists() and not self.pending_commit_path.exists():
                self._archive_failure(staging, index, phase, f"{type(exc).__name__}: {exc}")
            raise

    def advance(self, attempts: int, *, finalize: bool = False) -> dict[str, Any]:
        if type(attempts) is not int or attempts < 0:
            raise ValueError("attempts must be a nonnegative int")
        if not self.run_dir.exists():
            self.initialize()
        else:
            self.resume()
        state = self._load_state()
        if state["phase"] != "active":
            raise RuntimeError("evolution run is already frozen")
        completed = int(state["next_attempt"]) - 1
        if completed + attempts > self.acceptance.max_candidates:
            raise RuntimeError(
                f"requested candidates exceed predeclared max_candidates={self.acceptance.max_candidates}"
            )
        for _ in range(attempts):
            state = self._run_attempt(state)
        if finalize:
            state = self.freeze()
        return state

    def advance_to(self, target_candidates: int, *, finalize: bool = False) -> dict[str, Any]:
        if type(target_candidates) is not int or target_candidates < 0:
            raise ValueError("target_candidates must be a nonnegative int")
        if target_candidates > self.acceptance.max_candidates:
            raise RuntimeError(
                f"target candidates exceed predeclared max_candidates={self.acceptance.max_candidates}"
            )
        if finalize and not self.run_dir.exists() and target_candidates != 0:
            raise RuntimeError("finalize target must equal the already-completed candidate count 0")
        if not self.run_dir.exists():
            self.initialize()
        else:
            self.resume()
        state = self._load_state()
        completed = int(state["next_attempt"]) - 1
        if state["phase"] == "frozen":
            if finalize and target_candidates == completed:
                return state
            raise RuntimeError("evolution run is already frozen")
        if finalize and target_candidates != completed:
            raise RuntimeError(
                f"finalize target must equal the already-completed candidate count {completed}"
            )
        if target_candidates < completed:
            raise RuntimeError(
                f"target candidates {target_candidates} is below completed count {completed}"
            )
        for _ in range(target_candidates - completed):
            state = self._run_attempt(state)
        if finalize:
            state = self.freeze()
        return state

    def freeze(self) -> dict[str, Any]:
        self._validate_run_configuration()
        recovered = self._recover_pending_commit()
        state = self._load_state()
        if state["phase"] != "active":
            if recovered == "freeze":
                return state
            raise RuntimeError("evolution run is already frozen")
        self._guard().verify()
        incumbent = self._reference(str(state["incumbent"]))
        staging = self.run_dir / ".frozen-staging"
        if staging.exists():
            self._archive_auxiliary_staging(
                staging,
                "freeze",
                "interrupted",
                "uncommitted freeze staging found before retry",
            )
        staging.mkdir()
        try:
            shutil.copytree(incumbent / "scaffold", staging / "scaffold")
            scaffold_hash = mapping_sha256(tree_hashes(staging / "scaffold"))
            _atomic_json(
                staging / "FROZEN.json",
                {
                    "schema_version": 1,
                    "incumbent": state["incumbent"],
                    "scaffold_sha256": scaffold_hash,
                    "frozen_ns": time.time_ns(),
                },
            )
            post_state = {**state, "phase": "frozen", "frozen_scaffold_sha256": scaffold_hash}
            self._begin_commit(
                "freeze",
                staging,
                self.run_dir / "frozen",
                state,
                post_state,
            )
            return self._finish_pending_commit()
        except BaseException as exc:
            if staging.exists() and not self.pending_commit_path.exists():
                self._archive_auxiliary_staging(
                    staging,
                    "freeze",
                    "freeze",
                    f"{type(exc).__name__}: {exc}",
                )
            raise

    def run_sealed_transfer(self) -> TransferEvaluation:
        self._validate_run_configuration()
        self._recover_pending_commit()
        state = self._load_state()
        if state["phase"] != "frozen":
            raise PermissionError("sealed transfer is unavailable before scaffold freeze")
        frozen = self._validate_frozen_state(state)
        target = self.run_dir / "transfer"
        if target.exists():
            return self._load_transfer_result(target)
        staging = self.run_dir / ".transfer-staging"
        if staging.exists():
            self._archive_auxiliary_staging(
                staging,
                "transfer",
                "interrupted",
                "uncommitted transfer staging found before retry",
            )
        staging.mkdir()
        try:
            baseline = self._evaluate(self.run_dir / "baseline" / "scaffold", "transfer", staging / "baseline")
            evolved = self._evaluate(frozen / "scaffold", "transfer", staging / "evolved")
            baseline_metrics = compute_task_macro_metrics(baseline.outcomes)
            evolved_metrics = compute_task_macro_metrics(evolved.outcomes)
            comparison = paired_hierarchical_bootstrap(
                baseline.outcomes,
                evolved.outcomes,
                bootstrap_resamples=self.acceptance.bootstrap_resamples,
                confidence_level=self.acceptance.confidence_level,
                random_seed=self.acceptance.random_seed,
            )
            result = TransferEvaluation(baseline, evolved, baseline_metrics, evolved_metrics, comparison)
            _atomic_json(staging / "transfer_comparison.json", result.to_mapping())
            self._guard().verify()
            self._begin_commit("transfer", staging, target, state, state)
            self._finish_pending_commit()
            return self._load_transfer_result(target)
        except BaseException as exc:
            if staging.exists() and not self.pending_commit_path.exists():
                self._archive_auxiliary_staging(
                    staging,
                    "transfer",
                    "transfer",
                    f"{type(exc).__name__}: {exc}",
                )
            raise
