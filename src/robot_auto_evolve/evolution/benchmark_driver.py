from __future__ import annotations

import json
import os
import shutil
import time
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
from .hashing import EditablePolicy


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
        self.transfer_plan = transfer_plan
        self.transfer_metric = resolved_transfer_metric
        self.transfer_evaluator = transfer_evaluator
        self.editable = EditablePolicy()

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def _run_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "full_benchmark_evolution",
            "plan_sha256": self.plan.resolved_hash(),
            "plan_id": self.plan.plan_id,
            "model_route": self.plan.model_route,
            "scalar_metric": self.scalar_metric,
            "candidate_budget": self.candidate_budget,
            "editable_files": list(self.editable.allowed),
            "transfer_plan_sha256": None if self.transfer_plan is None else self.transfer_plan.resolved_hash(),
            "transfer_metric": None if self.transfer_plan is None else self.transfer_metric,
        }

    @staticmethod
    def _checked_state(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise StrictSchemaError("benchmark evolution state differs")
        state = dict(value)
        if (
            set(state) != {"schema_version", "phase", "next_candidate", "incumbent"}
            or state.get("schema_version") != 1
            or state.get("phase") not in {"active", "frozen"}
        ):
            raise StrictSchemaError("benchmark evolution state fields differ")
        if type(state["next_candidate"]) is not int or state["next_candidate"] < 1:
            raise StrictSchemaError("benchmark evolution next candidate differs")
        if type(state["incumbent"]) is not str or not state["incumbent"]:
            raise StrictSchemaError("benchmark evolution state values differ")
        return state

    def _write_state(self, value: Mapping[str, Any]) -> None:
        _write_json(self.state_path, self._checked_state(value))

    def _load_state(self) -> dict[str, Any]:
        return self._checked_state(_load_json(self.state_path))

    def _reference(self, relative: str) -> Path:
        path = (self.run_dir / relative).resolve()
        try:
            path.relative_to(self.run_dir)
        except ValueError as exc:
            raise StrictSchemaError("benchmark evolution reference escapes run") from exc
        return path

    def _validate_configuration(self) -> None:
        if _load_json(self.run_dir / "run_config.json") != self._run_config():
            raise RuntimeError("benchmark evolution run configuration differs")

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
        _write_json(target, result.to_mapping())
        return result

    def _commit(self, staging: Path, target: Path, post_state: Mapping[str, Any]) -> dict[str, Any]:
        state = self._checked_state(post_state)
        os.replace(staging, target)
        self._write_state(state)
        return state

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
        _write_json(
            wrapper / "failure.json",
            {"schema_version": 1, "label": label, "error": error, "archived_ns": time.time_ns()},
        )
        os.rename(wrapper, failures / f"{label}-{index:04d}")

    def _prepare_baseline(self) -> None:
        staging = self.run_dir / ".baseline-staging"
        if staging.exists():
            self._archive(staging, "baseline-interrupted", "uncommitted baseline staging")
        staging.mkdir()
        try:
            shutil.copytree(self.seed_scaffold, staging / "scaffold")
            self._evaluate(staging / "scaffold", staging / "benchmark")
            post = {
                "schema_version": 1,
                "phase": "active",
                "next_candidate": 1,
                "incumbent": "baseline",
            }
            self._commit(staging, self.run_dir / "baseline", post)
        except BaseException as exc:
            if staging.exists():
                self._archive(staging, "baseline-failed", f"{type(exc).__name__}: {exc}")
            raise

    def initialize(self) -> None:
        if self.run_dir.exists():
            raise FileExistsError(self.run_dir)
        self.editable.validate_tree(self.seed_scaffold)
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "candidates").mkdir()
        (self.run_dir / "failures").mkdir()
        _write_json(self.run_dir / "run_config.json", self._run_config())
        self._prepare_baseline()

    def resume(self) -> None:
        self._validate_configuration()
        if self.state_path.exists():
            state = self._load_state()
        elif (self.run_dir / "baseline").exists():
            # Crash landed between os.replace(baseline) and _write_state: the committed
            # baseline dir is on disk but state.json was never written. Synthesize it.
            state = {"schema_version": 1, "phase": "active", "next_candidate": 1, "incumbent": "baseline"}
        else:
            # Baseline was never committed; archive any partial staging and redo it.
            self._archive(self.run_dir / ".baseline-staging", "baseline-interrupted", "uncommitted baseline staging")
            self._prepare_baseline()
            return
        # Reconcile the loaded/synthesized state with the committed dirs on disk, in case a
        # crash landed in the tiny window between os.replace and _write_state.
        while (self.run_dir / "candidates" / f"{state['next_candidate']:04d}").exists():
            candidate = self.run_dir / "candidates" / f"{state['next_candidate']:04d}"
            decision = ScalarDecision.from_mapping(_load_json(candidate / "decision.json"))
            if decision.accepted:
                state["incumbent"] = f"candidates/{state['next_candidate']:04d}"
            state["next_candidate"] += 1
        if (self.run_dir / "frozen").exists():
            state["phase"] = "frozen"
        self._write_state(state)
        # Archive any leftover interrupted staging (redo semantics: the interrupted
        # candidate is redone next -- same behavior as before).
        self._archive(self.run_dir / ".baseline-staging", "baseline-interrupted", "uncommitted baseline staging")
        self._archive(
            self.run_dir / "candidates" / f".{state['next_candidate']:04d}-staging",
            f"candidate-{state['next_candidate']:04d}-interrupted",
            "uncommitted candidate staging",
        )
        self._archive(self.run_dir / ".frozen-staging", "freeze-interrupted", "uncommitted staging")
        self._archive(self.run_dir / ".transfer-staging", "transfer-interrupted", "uncommitted staging")

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
        _write_json(staging / "public_input.json", public_input)
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
            result = self._evaluate(staging / "scaffold", staging / "benchmark")
            incumbent_result = self._validate_result(incumbent)
            decision = ScalarDecision.create(incumbent_result.scalar.value, result.scalar.value)
            _write_json(staging / "decision.json", decision.to_mapping())
            _write_json(staging / "candidate_hashes.json", candidate_hashes)
            post = dict(state)
            post["next_candidate"] = index + 1
            if decision.accepted:
                post["incumbent"] = f"candidates/{index:04d}"
            return self._commit(staging, self.run_dir / "candidates" / f"{index:04d}", post)
        except (KeyboardInterrupt, SystemExit):
            # Operator/process-level interrupts are never a candidate rejection: archive and re-raise.
            if staging.exists():
                self._archive(staging, f"candidate-{index:04d}-interrupted", "run interrupted")
            raise
        except Exception as exc:
            # Reject-and-continue: a broken candidate revision (invalid agent_event, fairness
            # violation, non-compiling scaffold, episode errors, revision timeout, ...) is an
            # EXPECTED occasional event across a multi-candidate run. Archive it under failures/
            # for inspection, count it as a REJECTED attempt (incumbent unchanged), advance
            # next_candidate, and let the loop proceed instead of crashing the whole run.
            if staging.exists():
                self._archive(staging, f"candidate-{index:04d}-failed", f"{type(exc).__name__}: {exc}")
            post = dict(state)
            post["next_candidate"] = index + 1
            self._write_state(post)
            return post

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
        # A single call may both run the remaining candidates AND finalize (freeze +,
        # via run_transfer, the held-out comparison): the loop below advances to the
        # target first, then freeze() runs. The earlier "finalization requires an
        # already-completed target" guard forced a redundant second invocation (and a
        # full service reload); resume() still reconciles a separate finalize call, so
        # both the one-call and two-call forms work.
        for _ in range(target_candidates - completed):
            state = self._run_candidate(state)
        return self.freeze() if finalize else state

    def freeze(self) -> dict[str, Any]:
        self._validate_configuration()
        state = self._load_state()
        if state["phase"] != "active":
            raise RuntimeError("benchmark evolution run is already frozen")
        incumbent = self._reference(state["incumbent"])
        staging = self.run_dir / ".frozen-staging"
        staging.mkdir()
        try:
            shutil.copytree(incumbent / "scaffold", staging / "scaffold")
            _write_json(
                staging / "FROZEN.json",
                {
                    "schema_version": 1,
                    "incumbent": state["incumbent"],
                    "frozen_ns": time.time_ns(),
                },
            )
            post = {**state, "phase": "frozen"}
            return self._commit(staging, self.run_dir / "frozen", post)
        except BaseException as exc:
            if staging.exists():
                self._archive(staging, "freeze-failed", f"{type(exc).__name__}: {exc}")
            raise

    def run_transfer(self) -> BenchmarkTransferComparison:
        if self.transfer_plan is None or self.transfer_evaluator is None:
            raise RuntimeError("benchmark evolution study has no held-out transfer plan")
        self._validate_configuration()
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
            _write_json(staging / "transfer_comparison.json", comparison.to_mapping())
            self._commit(staging, target, state)
            return comparison
        except BaseException as exc:
            if staging.exists():
                self._archive(staging, "transfer-failed", f"{type(exc).__name__}: {exc}")
            raise
