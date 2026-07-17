from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from robot_auto_evolve.evaluation.scalars import SCALAR_METRICS, compute_benchmark_scalar
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan

from .benchmark_evidence import BenchmarkPublicEvidence
from .benchmark_models import (
    BenchmarkEvaluationData,
    BenchmarkEvaluationResult,
    BenchmarkEvaluator,
    BenchmarkTransferComparison,
    RevisionBackend,
    ScalarDecision,
)
from .hashing import (
    EditablePolicy,
    FrozenHashGuard,
    mapping_sha256,
    tree_hashes,
    verify_tree_manifest,
    write_tree_manifest,
)


PENDING_COMMIT = "pending_commit.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class BenchmarkEvolutionDriver:
    def __init__(
        self,
        *,
        seed_scaffold: Path,
        run_dir: Path,
        plan: BenchmarkPlan,
        scalar_metric: str,
        evaluator: BenchmarkEvaluator,
        revision_backend: RevisionBackend,
        candidate_budget: int,
        frozen_paths: tuple[Path, ...],
        transfer_plan: BenchmarkPlan | None = None,
        transfer_metric: str | None = None,
        transfer_evaluator: BenchmarkEvaluator | None = None,
    ) -> None:
        if not isinstance(plan, BenchmarkPlan):
            raise StrictSchemaError("benchmark evolution requires BenchmarkPlan")
        if scalar_metric not in SCALAR_METRICS:
            raise StrictSchemaError("benchmark evolution scalar metric differs")
        if type(candidate_budget) is not int or not 1 <= candidate_budget <= 10_000:
            raise StrictSchemaError("benchmark evolution candidate budget differs")
        transfer_values = (transfer_plan, transfer_evaluator)
        if any(item is None for item in transfer_values) != all(item is None for item in transfer_values):
            raise StrictSchemaError("benchmark evolution transfer plan and evaluator must be supplied together")
        if transfer_plan is not None and not isinstance(transfer_plan, BenchmarkPlan):
            raise StrictSchemaError("benchmark evolution transfer plan differs")
        if transfer_plan is not None and (
            {item.task_id for item in transfer_plan.episodes}
            & {item.task_id for item in plan.episodes}
        ):
            raise StrictSchemaError("benchmark evolution transfer tasks must be held out")
        resolved_transfer_metric = scalar_metric if transfer_metric is None else transfer_metric
        if resolved_transfer_metric not in SCALAR_METRICS:
            raise StrictSchemaError("benchmark evolution transfer metric differs")
        self.seed_scaffold = Path(seed_scaffold).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.plan = plan
        self.scalar_metric = scalar_metric
        self.evaluator = evaluator
        self.revision_backend = revision_backend
        self.candidate_budget = candidate_budget
        self.frozen_paths = tuple(Path(path).resolve() for path in frozen_paths)
        if not self.frozen_paths:
            raise StrictSchemaError("benchmark evolution requires frozen paths")
        if any(self.run_dir == path or path in self.run_dir.parents for path in self.frozen_paths):
            raise StrictSchemaError("benchmark evolution run cannot be inside a frozen path")
        self.transfer_plan = transfer_plan
        self.transfer_metric = resolved_transfer_metric
        self.transfer_evaluator = transfer_evaluator
        self.editable = EditablePolicy()

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def pending_path(self) -> Path:
        return self.run_dir / PENDING_COMMIT

    def _run_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "full_benchmark_evolution",
            "plan_sha256": self.plan.resolved_hash(),
            "plan_id": self.plan.plan_id,
            "model_route": self.plan.model_route,
            "scalar_metric": self.scalar_metric,
            "candidate_budget": self.candidate_budget,
            "seed_hashes": tree_hashes(self.seed_scaffold),
            "editable_files": list(self.editable.allowed),
            "transfer_plan_sha256": None if self.transfer_plan is None else self.transfer_plan.resolved_hash(),
            "transfer_metric": None if self.transfer_plan is None else self.transfer_metric,
        }

    @staticmethod
    def _checked_state(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise StrictSchemaError("benchmark evolution state differs")
        state = dict(value)
        common = {"schema_version", "phase", "next_candidate", "incumbent", "created_ns"}
        expected = common | ({"frozen_scaffold_sha256"} if state.get("phase") == "frozen" else set())
        if set(state) != expected or state.get("schema_version") != 1 or state.get("phase") not in {"active", "frozen"}:
            raise StrictSchemaError("benchmark evolution state fields differ")
        if type(state["next_candidate"]) is not int or state["next_candidate"] < 1:
            raise StrictSchemaError("benchmark evolution next candidate differs")
        if type(state["incumbent"]) is not str or not state["incumbent"] or type(state["created_ns"]) is not int:
            raise StrictSchemaError("benchmark evolution state values differ")
        if state["phase"] == "frozen":
            digest = state["frozen_scaffold_sha256"]
            if type(digest) is not str or len(digest) != 64:
                raise StrictSchemaError("benchmark evolution frozen hash differs")
        return state

    def _write_state(self, value: Mapping[str, Any]) -> None:
        state = self._checked_state(value)
        _atomic_json(self.state_path, {**state, "state_sha256": mapping_sha256(state)})

    def _load_state(self) -> dict[str, Any]:
        value = _load_json(self.state_path)
        if not isinstance(value, Mapping) or "state_sha256" not in value:
            raise StrictSchemaError("benchmark evolution state checksum is absent")
        state = dict(value)
        digest = state.pop("state_sha256")
        if digest != mapping_sha256(state):
            raise StrictSchemaError("benchmark evolution state checksum differs")
        return self._checked_state(state)

    def _guard(self) -> FrozenHashGuard:
        return FrozenHashGuard.from_manifest(_load_json(self.run_dir / "frozen_hashes.json"))

    def _reference(self, relative: str) -> Path:
        path = (self.run_dir / relative).resolve()
        try:
            path.relative_to(self.run_dir)
        except ValueError as exc:
            raise StrictSchemaError("benchmark evolution reference escapes run") from exc
        verify_tree_manifest(path)
        return path

    def _validate_configuration(self) -> FrozenHashGuard:
        if _load_json(self.run_dir / "run_config.json") != self._run_config():
            raise RuntimeError("benchmark evolution run configuration differs")
        guard = self._guard()
        guard.verify()
        return guard

    def _validate_result(
        self,
        root: Path,
        *,
        plan: BenchmarkPlan | None = None,
        metric: str | None = None,
        filename: str = "benchmark_result.json",
    ) -> BenchmarkEvaluationResult:
        selected_plan = self.plan if plan is None else plan
        selected_metric = self.scalar_metric if metric is None else metric
        result = BenchmarkEvaluationResult.load(root / filename)
        if result.plan_sha256 != selected_plan.resolved_hash() or result.scalar.metric != selected_metric:
            raise StrictSchemaError("benchmark evolution result identity differs")
        if tuple(item.key for item in result.outcomes) != selected_plan.episodes:
            raise StrictSchemaError("benchmark evolution result does not cover the exact plan")
        if compute_benchmark_scalar(selected_metric, result.outcomes).to_mapping() != result.scalar.to_mapping():
            raise StrictSchemaError("benchmark evolution result scalar differs")
        evidence = BenchmarkPublicEvidence.load(root / filename.removesuffix("_result.json") / "public_evidence")
        if evidence.bundle_sha256 != result.evidence_sha256 or evidence.outcomes != result.outcomes:
            raise StrictSchemaError("benchmark evolution result evidence differs")
        return result

    def _evaluate(
        self,
        scaffold: Path,
        output: Path,
        *,
        evaluator: BenchmarkEvaluator | None = None,
        plan: BenchmarkPlan | None = None,
        metric: str | None = None,
        result_path: Path | None = None,
    ) -> BenchmarkEvaluationResult:
        selected_evaluator = self.evaluator if evaluator is None else evaluator
        selected_plan = self.plan if plan is None else plan
        selected_metric = self.scalar_metric if metric is None else metric
        data = selected_evaluator.evaluate(scaffold, output)
        if not isinstance(data, BenchmarkEvaluationData):
            raise StrictSchemaError("benchmark evaluator returned the wrong type")
        if tuple(item.key for item in data.outcomes) != selected_plan.episodes:
            raise StrictSchemaError("benchmark evaluator did not execute the exact plan once")
        scalar = compute_benchmark_scalar(selected_metric, data.outcomes)
        evidence = BenchmarkPublicEvidence.create(
            output / "public_evidence",
            data.outcomes,
            scalar,
            data.diagnostics,
        )
        result = BenchmarkEvaluationResult(
            plan_sha256=selected_plan.resolved_hash(),
            scalar=scalar,
            outcomes=data.outcomes,
            metadata=data.metadata,
            evidence_sha256=evidence.bundle_sha256,
            evidence_episodes=len(data.outcomes),
        )
        target = output.parent / f"{output.name}_result.json" if result_path is None else result_path
        _atomic_json(target, result.to_mapping())
        return result

    def _begin_commit(
        self,
        kind: str,
        source: Path,
        target: Path,
        pre_state: Mapping[str, Any] | None,
        post_state: Mapping[str, Any],
        candidate: int | None = None,
    ) -> None:
        if self.pending_path.exists():
            raise RuntimeError("benchmark evolution commit is already pending")
        source_relative = source.relative_to(self.run_dir).as_posix()
        target_relative = target.relative_to(self.run_dir).as_posix()
        value = {
            "schema_version": 1,
            "kind": kind,
            "candidate": candidate,
            "source": source_relative,
            "target": target_relative,
            "pre_state": None if pre_state is None else self._checked_state(pre_state),
            "post_state": self._checked_state(post_state),
            "created_ns": time.time_ns(),
        }
        _atomic_json(self.pending_path, {**value, "journal_sha256": mapping_sha256(value)})

    def _load_pending(self) -> dict[str, Any]:
        value = _load_json(self.pending_path)
        if not isinstance(value, Mapping) or "journal_sha256" not in value:
            raise StrictSchemaError("benchmark evolution commit journal differs")
        stable = dict(value)
        digest = stable.pop("journal_sha256")
        if digest != mapping_sha256(stable) or set(stable) != {
            "schema_version",
            "kind",
            "candidate",
            "source",
            "target",
            "pre_state",
            "post_state",
            "created_ns",
        }:
            raise StrictSchemaError("benchmark evolution commit journal checksum differs")
        if stable["schema_version"] != 1 or stable["kind"] not in {"baseline", "candidate", "freeze", "transfer"}:
            raise StrictSchemaError("benchmark evolution commit journal identity differs")
        stable["pre_state"] = None if stable["pre_state"] is None else self._checked_state(stable["pre_state"])
        stable["post_state"] = self._checked_state(stable["post_state"])
        return stable

    def _validate_committed(self, journal: Mapping[str, Any], target: Path) -> None:
        kind = journal["kind"]
        pre = journal["pre_state"]
        post = journal["post_state"]
        if kind == "baseline":
            if pre is not None or post["incumbent"] != "baseline" or post["next_candidate"] != 1:
                raise StrictSchemaError("benchmark evolution baseline transition differs")
            self.editable.validate_tree(target / "scaffold")
            self._validate_result(target)
            return
        if pre is None:
            raise StrictSchemaError("benchmark evolution commit lacks pre-state")
        if kind == "candidate":
            index = journal["candidate"]
            if type(index) is not int or index != pre["next_candidate"]:
                raise StrictSchemaError("benchmark evolution candidate transition differs")
            result = self._validate_result(target)
            incumbent = self._validate_result(self._reference(pre["incumbent"]))
            decision = ScalarDecision.from_mapping(_load_json(target / "decision.json"))
            if decision.to_mapping() != ScalarDecision.create(incumbent.scalar.value, result.scalar.value).to_mapping():
                raise StrictSchemaError("benchmark evolution candidate decision differs")
            expected = dict(pre)
            expected["next_candidate"] = index + 1
            if decision.accepted:
                expected["incumbent"] = f"candidates/{index:04d}"
            if post != expected:
                raise StrictSchemaError("benchmark evolution candidate post-state differs")
            self.editable.validate_tree(target / "scaffold")
            return
        if kind == "freeze":
            marker = _load_json(target / "FROZEN.json")
            scaffold_hash = mapping_sha256(tree_hashes(target / "scaffold"))
            if post != {**pre, "phase": "frozen", "frozen_scaffold_sha256": scaffold_hash}:
                raise StrictSchemaError("benchmark evolution freeze transition differs")
            if marker != {
                "schema_version": 1,
                "incumbent": pre["incumbent"],
                "scaffold_sha256": scaffold_hash,
                "frozen_ns": marker.get("frozen_ns"),
            } or type(marker["frozen_ns"]) is not int:
                raise StrictSchemaError("benchmark evolution frozen marker differs")
            self.editable.validate_tree(target / "scaffold")
            return
        if pre != post or pre["phase"] != "frozen" or self.transfer_plan is None:
            raise StrictSchemaError("benchmark evolution transfer transition differs")
        comparison = _load_json(target / "transfer_comparison.json")
        baseline = self._validate_result(
            target,
            plan=self.transfer_plan,
            metric=self.transfer_metric,
            filename="baseline_transfer_result.json",
        )
        evolved = self._validate_result(
            target,
            plan=self.transfer_plan,
            metric=self.transfer_metric,
            filename="evolved_transfer_result.json",
        )
        expected = BenchmarkTransferComparison(baseline, evolved).to_mapping()
        if comparison != expected:
            raise StrictSchemaError("benchmark evolution transfer comparison differs")

    def _finish_pending(self) -> dict[str, Any]:
        journal = self._load_pending()
        source = self.run_dir / journal["source"]
        target = self.run_dir / journal["target"]
        if source.exists() == target.exists():
            raise RuntimeError("benchmark evolution pending commit has ambiguous filesystem state")
        if source.exists():
            if (source / "run_manifest.json").exists():
                verify_tree_manifest(source)
            else:
                write_tree_manifest(source)
            os.rename(source, target)
            _fsync_directory(target.parent)
        verify_tree_manifest(target)
        self._validate_committed(journal, target)
        current = self._load_state() if self.state_path.exists() else None
        if current is not None and current != journal["pre_state"] and current != journal["post_state"]:
            raise RuntimeError("benchmark evolution state differs from pending commit")
        if current != journal["post_state"]:
            self._write_state(journal["post_state"])
        self.pending_path.unlink()
        _fsync_directory(self.run_dir)
        return dict(journal["post_state"])

    def _archive(self, staging: Path, label: str, error: str) -> None:
        if not staging.exists():
            return
        failures = self.run_dir / "failures"
        index = 1
        while (failures / f"{label}-{index:04d}").exists():
            index += 1
        wrapper = failures / f".{label}-{index:04d}-staging"
        wrapper.mkdir()
        os.rename(staging, wrapper / "payload")
        _atomic_json(
            wrapper / "failure.json",
            {"schema_version": 1, "label": label, "error": error, "archived_ns": time.time_ns()},
        )
        write_tree_manifest(wrapper)
        os.rename(wrapper, failures / f"{label}-{index:04d}")
        _fsync_directory(failures)

    def _prepare_baseline(self, guard: FrozenHashGuard) -> None:
        staging = self.run_dir / ".baseline-staging"
        if staging.exists():
            self._archive(staging, "baseline-interrupted", "uncommitted baseline staging")
        staging.mkdir()
        try:
            shutil.copytree(self.seed_scaffold, staging / "scaffold")
            self._evaluate(staging / "scaffold", staging / "benchmark")
            guard.verify()
            post = {
                "schema_version": 1,
                "phase": "active",
                "next_candidate": 1,
                "incumbent": "baseline",
                "created_ns": time.time_ns(),
            }
            self._begin_commit("baseline", staging, self.run_dir / "baseline", None, post)
            self._finish_pending()
        except BaseException as exc:
            if staging.exists() and not self.pending_path.exists():
                self._archive(staging, "baseline-failed", f"{type(exc).__name__}: {exc}")
            raise

    def initialize(self) -> None:
        if self.run_dir.exists():
            raise FileExistsError(self.run_dir)
        self.editable.validate_tree(self.seed_scaffold)
        guard = FrozenHashGuard(self.frozen_paths)
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "candidates").mkdir()
        (self.run_dir / "failures").mkdir()
        _atomic_json(self.run_dir / "frozen_hashes.json", guard.to_mapping())
        _atomic_json(self.run_dir / "run_config.json", self._run_config(), mode=0o444)
        self._prepare_baseline(guard)

    def resume(self) -> None:
        guard = self._validate_configuration()
        if self.pending_path.exists():
            self._finish_pending()
        if not self.state_path.exists():
            if (self.run_dir / "baseline").exists() or any((self.run_dir / "candidates").iterdir()):
                raise RuntimeError("benchmark evolution baseline state is incomplete")
            self._prepare_baseline(guard)
            return
        state = self._load_state()
        self._reference("baseline")
        for index in range(1, state["next_candidate"]):
            self._reference(f"candidates/{index:04d}")
        if state["phase"] == "frozen":
            frozen = self._reference("frozen")
            marker = _load_json(frozen / "FROZEN.json")
            scaffold_hash = mapping_sha256(tree_hashes(frozen / "scaffold"))
            if (
                marker.get("scaffold_sha256") != scaffold_hash
                or state["frozen_scaffold_sha256"] != scaffold_hash
                or marker.get("incumbent") != state["incumbent"]
            ):
                raise RuntimeError("benchmark evolution frozen state differs")
        for path in sorted((self.run_dir / "failures").iterdir()):
            if path.is_dir():
                verify_tree_manifest(path)
        staging = self.run_dir / "candidates" / f".{state['next_candidate']:04d}-staging"
        if staging.exists():
            self._archive(staging, f"candidate-{state['next_candidate']:04d}-interrupted", "uncommitted candidate staging")
        for staging, label in (
            (self.run_dir / ".frozen-staging", "freeze-interrupted"),
            (self.run_dir / ".transfer-staging", "transfer-interrupted"),
        ):
            if staging.exists():
                self._archive(staging, label, "uncommitted staging")

    def _revision_material(self, state: Mapping[str, Any], index: int, staging: Path) -> str:
        incumbent = self._reference(state["incumbent"])
        incumbent_result = self._validate_result(incumbent)
        shutil.copytree(incumbent / "benchmark" / "public_evidence", staging / "incumbent_evidence")
        previous = None
        if index > 1:
            previous_root = self._reference(f"candidates/{index - 1:04d}")
            if previous_root != incumbent:
                previous = self._validate_result(previous_root)
                shutil.copytree(previous_root / "benchmark" / "public_evidence", staging / "previous_candidate_evidence")
        public_input = {
            "schema_version": 1,
            "candidate": index,
            "benchmark_plan_sha256": self.plan.resolved_hash(),
            "scalar_metric": self.scalar_metric,
            "incumbent_scalar": incumbent_result.scalar.value,
            "incumbent_full_outcomes": "incumbent_evidence/outcomes.json",
            "incumbent_diagnostics": "incumbent_evidence/diagnostics.json",
            "previous_rejected_candidate_scalar": None if previous is None else previous.scalar.value,
            "previous_rejected_candidate_full_outcomes": None if previous is None else "previous_candidate_evidence/outcomes.json",
        }
        _atomic_json(staging / "public_input.json", public_input, mode=0o444)
        return (
            "Revise only scaffold.py and make one focused change. The exact standard benchmark is intentionally in the optimization loop. "
            "Read ../public_input.json, every row in ../incumbent_evidence/outcomes.json, the bounded diagnostics, and the current scaffold. "
            "If public_input names previous_candidate_evidence, use it to avoid repeating a rejected change, but keep the incumbent scaffold as the code base. "
            "Do not seek live reward, hidden simulator state, expert actions, or files outside the provided public evidence."
        )

    def _run_candidate(self, state: dict[str, Any]) -> dict[str, Any]:
        index = state["next_candidate"]
        staging = self.run_dir / "candidates" / f".{index:04d}-staging"
        staging.mkdir()
        try:
            incumbent = self._reference(state["incumbent"])
            shutil.copytree(incumbent / "scaffold", staging / "scaffold")
            (staging / "scaffold" / "scaffold.py").chmod(0o600)
            prompt = self._revision_material(state, index, staging)
            (staging / "revision_prompt.txt").write_text(prompt, encoding="utf-8")
            self.revision_backend.revise(prompt, staging / "scaffold", staging / "revision_logs", index)
            candidate_hashes = self.editable.validate_revision(incumbent / "scaffold", staging / "scaffold")
            self._guard().verify()
            result = self._evaluate(staging / "scaffold", staging / "benchmark")
            incumbent_result = self._validate_result(incumbent)
            decision = ScalarDecision.create(incumbent_result.scalar.value, result.scalar.value)
            _atomic_json(staging / "decision.json", decision.to_mapping())
            _atomic_json(staging / "candidate_hashes.json", candidate_hashes)
            post = dict(state)
            post["next_candidate"] = index + 1
            if decision.accepted:
                post["incumbent"] = f"candidates/{index:04d}"
            self._guard().verify()
            self._begin_commit(
                "candidate",
                staging,
                self.run_dir / "candidates" / f"{index:04d}",
                state,
                post,
                index,
            )
            return self._finish_pending()
        except BaseException as exc:
            if staging.exists() and not self.pending_path.exists():
                self._archive(staging, f"candidate-{index:04d}-failed", f"{type(exc).__name__}: {exc}")
            raise

    def advance_to(self, target_candidates: int, *, finalize: bool = False) -> dict[str, Any]:
        if type(target_candidates) is not int or not 0 <= target_candidates <= self.candidate_budget:
            raise ValueError("target candidates falls outside the predeclared budget")
        if not self.run_dir.exists():
            self.initialize()
        else:
            self.resume()
        state = self._load_state()
        completed = state["next_candidate"] - 1
        if state["phase"] == "frozen":
            if finalize and completed == target_candidates:
                return state
            raise RuntimeError("benchmark evolution run is already frozen")
        if target_candidates < completed:
            raise RuntimeError("target candidates is below the completed count")
        if finalize and target_candidates != completed:
            raise RuntimeError("finalization requires an already-completed target")
        for _ in range(target_candidates - completed):
            state = self._run_candidate(state)
        return self.freeze() if finalize else state

    def freeze(self) -> dict[str, Any]:
        self._validate_configuration()
        if self.pending_path.exists():
            self._finish_pending()
        state = self._load_state()
        if state["phase"] != "active":
            raise RuntimeError("benchmark evolution run is already frozen")
        incumbent = self._reference(state["incumbent"])
        staging = self.run_dir / ".frozen-staging"
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
            post = {**state, "phase": "frozen", "frozen_scaffold_sha256": scaffold_hash}
            self._begin_commit("freeze", staging, self.run_dir / "frozen", state, post)
            return self._finish_pending()
        except BaseException as exc:
            if staging.exists() and not self.pending_path.exists():
                self._archive(staging, "freeze-failed", f"{type(exc).__name__}: {exc}")
            raise

    def run_transfer(self) -> BenchmarkTransferComparison:
        if self.transfer_plan is None or self.transfer_evaluator is None:
            raise RuntimeError("benchmark evolution study has no held-out transfer plan")
        self._validate_configuration()
        if self.pending_path.exists():
            self._finish_pending()
        state = self._load_state()
        if state["phase"] != "frozen":
            raise PermissionError("held-out transfer is available only after finalization")
        target = self.run_dir / "transfer"
        if target.exists():
            self._reference("transfer")
            baseline = self._validate_result(
                target,
                plan=self.transfer_plan,
                metric=self.transfer_metric,
                filename="baseline_transfer_result.json",
            )
            evolved = self._validate_result(
                target,
                plan=self.transfer_plan,
                metric=self.transfer_metric,
                filename="evolved_transfer_result.json",
            )
            return BenchmarkTransferComparison(baseline, evolved)
        staging = self.run_dir / ".transfer-staging"
        staging.mkdir()
        try:
            baseline = self._evaluate(
                self.run_dir / "baseline" / "scaffold",
                staging / "baseline_transfer",
                evaluator=self.transfer_evaluator,
                plan=self.transfer_plan,
                metric=self.transfer_metric,
                result_path=staging / "baseline_transfer_result.json",
            )
            evolved = self._evaluate(
                self.run_dir / "frozen" / "scaffold",
                staging / "evolved_transfer",
                evaluator=self.transfer_evaluator,
                plan=self.transfer_plan,
                metric=self.transfer_metric,
                result_path=staging / "evolved_transfer_result.json",
            )
            comparison = BenchmarkTransferComparison(baseline, evolved)
            _atomic_json(staging / "transfer_comparison.json", comparison.to_mapping())
            self._guard().verify()
            self._begin_commit("transfer", staging, target, state, state)
            self._finish_pending()
            return comparison
        except BaseException as exc:
            if staging.exists() and not self.pending_path.exists():
                self._archive(staging, "transfer-failed", f"{type(exc).__name__}: {exc}")
            raise
