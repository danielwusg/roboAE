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


def _resumable(staging: Path) -> bool:
    """True iff an interrupted staging dir can be resumed in place: its scaffold is present
    and its evaluation output tree exists. On resume the already-committed episodes under
    staging/benchmark/canonical/episodes are reused (the evaluator's pending-set skips them)
    and only the unfinished + not-yet-started episodes run. A staging that never reached the
    evaluation stage (e.g. interrupted mid-coding-agent-revision) is NOT resumable -> redo."""
    return (staging / "benchmark" / "canonical").is_dir() and (staging / "scaffold" / "scaffold.py").is_file()


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
        route_notes: str = "",
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
        # Revision 1 (§2.4): per-route facts the coding agent would otherwise have to discover the
        # hard way -- which tools are actually served here, whether the cameras carry 3D, what the
        # action channels are. Built by study_runner from the live runtime profile and pasted into
        # the revision prompt verbatim. Empty string = no route section (fixtures/tests).
        if type(route_notes) is not str:
            raise StrictSchemaError("benchmark evolution route notes differ")
        self.route_notes = route_notes
        # Revision 1 (§2.3): state the coding agent's real budget as a FACT in the prompt. Read off
        # the backend that will actually enforce it, so the prompt can never quote a stale number.
        self.revision_max_turns = int(getattr(revision_backend, "max_turns", 0) or 0)
        self.revision_timeout_s = float(getattr(revision_backend, "timeout_s", 0.0) or 0.0)
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

    def _progress(self, message: str) -> None:
        """Append one human-readable, timestamped line to <run_dir>/progress.log so a
        run can be followed live with `tail -f`. Best-effort: logging must never break
        a run, so any I/O error is swallowed."""
        try:
            with open(self.run_dir / "progress.log", "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
        except OSError:
            pass

    def _metrics_row(self, phase: str, candidate: Any, score: Any, incumbent: Any, accepted: Any, episodes: Any) -> None:
        """Append one row to <run_dir>/metrics.csv (writing the header on first use)."""
        try:
            path = self.run_dir / "metrics.csv"
            header = not path.exists()
            with open(path, "a", encoding="utf-8") as handle:
                if header:
                    handle.write("phase,candidate,score,incumbent_score,accepted,n_episodes\n")
                handle.write(f"{phase},{candidate},{score},{incumbent},{accepted},{episodes}\n")
        except OSError:
            pass

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
        result = BenchmarkEvaluationResult(
            plan_sha256=selected_plan.resolved_hash(),
            scalar=scalar,
            outcomes=data.outcomes,
            metadata=data.metadata,
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
        # RESUME: if a previous attempt was interrupted mid-evaluation, its finished episodes
        # are already committed under staging/benchmark and the evaluator's pending-set will
        # skip them, so we re-enter that staging rather than redo baseline from episode zero.
        resuming = _resumable(staging)
        if staging.exists() and not resuming:
            self._archive(staging, "baseline-interrupted", "unusable partial baseline staging")
        if not resuming:
            staging.mkdir()
            shutil.copytree(self.seed_scaffold, staging / "scaffold")
        try:
            self._progress(
                "baseline: " + ("RESUMING evaluation (skipping finished episodes)" if resuming else "evaluating the seed scaffold") + " ..."
            )
            result = self._evaluate(staging / "scaffold", staging / "benchmark")
            post = {
                "schema_version": 1,
                "phase": "active",
                "next_candidate": 1,
                "incumbent": "baseline",
            }
            self._commit(staging, self.run_dir / "baseline", post)
            self._progress(
                f"baseline: DONE  {self.scalar_metric}={result.scalar.value:.4f}  ({len(result.outcomes)} episodes)"
            )
            self._metrics_row("baseline", 0, f"{result.scalar.value:.6f}", "-", "-", len(result.outcomes))
        except (KeyboardInterrupt, SystemExit):
            # Interrupt: leave the partial staging in place so the next run resumes it. (SIGTERM
            # arrives as RunInterrupted, a BaseException, and already bypasses these handlers.)
            raise
        except Exception as exc:
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
            # Baseline was never committed: resume its partial staging in place (or, if there is
            # no resumable partial, _prepare_baseline archives the unusable remnant and starts fresh).
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
        # Leftover baseline/candidate/transfer staging is NOT archived here: _prepare_baseline,
        # _run_candidate and run_transfer each resume a partial evaluation in place (skipping finished
        # episodes via the evaluator's pending-set), or redo it if it never reached the evaluation stage.
        # This matters for transfer, whose held-out set can be large (e.g. 2x400 episodes) -- archive+redo
        # would waste that whole set on any interrupt (s17 fix; the s16 "transfer is cheap to redo" note
        # was wrong at scale). Only the freeze staging is still archived: freeze is a single scaffold copy
        # + a small JSON, genuinely cheap, and has no episodes to skip.
        self._archive(self.run_dir / ".frozen-staging", "freeze-interrupted", "uncommitted staging")

    def _revision_material(self, state: Mapping[str, Any], index: int, staging: Path) -> str:
        incumbent = self._reference(state["incumbent"])
        incumbent_result = self._validate_result(incumbent)
        incumbent_episodes = incumbent / "benchmark" / "canonical" / "episodes"
        previous = None
        previous_episodes = None
        if index > 1:
            previous_root = self._reference(f"candidates/{index - 1:04d}")
            # A broken previous candidate is archived under failures/ and leaves no candidates/NNNN
            # dir, so guard on is_dir() (else reading a rejected candidate would cascade-fail this one).
            if previous_root != incumbent and previous_root.is_dir():
                previous = self._validate_result(previous_root)
                previous_episodes = previous_root / "benchmark" / "canonical" / "episodes"
        public_input = {
            "schema_version": 1,
            "candidate": index,
            "scalar_metric": self.scalar_metric,
            "incumbent_scalar": incumbent_result.scalar.value,
            "incumbent_episode_traces_dir": str(incumbent_episodes),
            "previous_rejected_candidate_scalar": None if previous is None else previous.scalar.value,
            "previous_rejected_candidate_episode_traces_dir": None if previous_episodes is None else str(previous_episodes),
        }
        _write_json(staging / "public_input.json", public_input)
        route_section = f"\n## Facts about THIS route\n{self.route_notes.strip()}\n" if self.route_notes.strip() else ""
        return (
            "You are a senior robotics-AI research engineer improving an AGENT SCAFFOLD. The scaffold (scaffold.py) "
            "wraps a FROZEN robot policy (weights frozen, served) and may collaborate with other FROZEN "
            "foundation-model TOOLS (a vision model, a language model, a detector, a segmenter, a pointer). You do "
            "NOT train or change any model. You improve the SCAFFOLD: everything that happens between the observation "
            "arriving and the action leaving.\n\n"
            f"## Objective\n"
            f"Maximize the route metric `{self.scalar_metric}` on the FIXED evaluation set (the exact same tasks and "
            f"seeds every iteration, so scores are directly comparable). The current incumbent scaffold scores "
            f"{incumbent_result.scalar.value:.4f}. The exact benchmark is intentionally in the optimization loop; "
            "acceptance is a strict improvement over the incumbent, and a disjoint held-out task set is scored "
            "separately after the run is frozen. The held-out score is the number that decides whether your change was "
            "real, so prefer changes that would work on a task you have never seen.\n\n"
            "How much to change in one revision is your call. There is no rule that it must be one edit; a coherent "
            "set of changes aimed at one diagnosis is fine, and so is a rewrite if the evidence supports it. Whatever "
            "you do, finish your run by stating plainly what you changed and why, so the change can be attributed "
            "later.\n\n"
            "## What you may edit, and what survives\n"
            "ONLY scaffold.py, in your current directory, edited IN PLACE. Keep create_scaffold(), the SCAFFOLD_CONFIG "
            "dict, and the act(request, tools) signature intact so the harness can still load and run it. EVERYTHING "
            "ELSE you write into this directory is deleted the moment you stop: notes, extra modules, caches, data "
            "files. The whole agent has to fit in that one file, and nothing you write persists to the next episode, "
            "the next candidate, or the next run. Scratch files elsewhere (e.g. /tmp) are fine while you work; they "
            "just do not travel.\n\n"
            "## The scaffold interface (read scaffold.py first)\n"
            "act(request, tools) is called once per environment step and must return exactly ONE CanonicalActionChunk. "
            "request.observation is a privilege-stripped FairObservation with: .instruction (the task text, straight "
            "from the benchmark), .cameras (name -> CameraObservation), .proprioception (the robot's own joint / "
            "end-effector state, as RobotStateVector entries you can look up by name), and .step_index. Each "
            "CameraObservation has .rgb (uint8 HxWx3) and, when the route supplies 3D, .depth_m (float32 HxW, metres, "
            "pixel-aligned with .rgb), .depth_valid, .intrinsics (3x3) and .camera_to_world (4x4). The exact "
            "dataclasses are in robot_auto_evolve/agent/api.py; the toolbox interface is in "
            "robot_auto_evolve/agent/tools.py; the observation schema is in "
            "robot_auto_evolve/protocol/observation.py. Read them rather than guessing.\n\n"
            "## Tools you may call (frozen infra -- call them, do NOT edit them)\n"
            "- tools.vla(VLARequest(...)) -> the FROZEN robot policy. Always served. It returns a CanonicalActionChunk "
            "whose .spec describes the action channels and .values holds the numbers.\n"
            "  - `instruction=` is the text the policy is conditioned on. It is the main lever you have over it.\n"
            "  - `refresh=True` CLEARS the policy's own internal action cache. Most of these policies emit a chunk of "
            "future actions and hand them out one per step; refresh throws the remaining chunk away and re-runs the "
            "model on the current observation. That also resets whatever internal state the chunk carried (for "
            "instance a held gripper command), so a refresh is a real behaviour change, not just a recompute.\n"
            "  - `context=` is a tuple of strings shipped to the policy service. It is a DEAD CHANNEL: no policy "
            "adapter reads it, with exactly one exception -- X-VLA reads a single key, `policy_resample_index=<n>`, "
            "and uses it to re-draw its action with a different sampling seed for the same observation "
            "(robot_auto_evolve/policies/xvla.py, `_resample_index`). Anything else you put in `context` is "
            "transmitted and then ignored. If you want the policy to know something, it has to be in the instruction "
            "text or in the observation.\n"
            "- Optional perception/language tools, each guarded with tools.has(<capability>) because a route serves "
            "only some. Return shapes, exactly as declared in agent/api.py:\n"
            "  - language -> tools.language(LanguageRequest(instruction, context, max_tokens, temperature)) -> "
            "TextResult(.text)\n"
            "  - vision   -> tools.vision(VisionRequest(instruction, images={name: rgb}, context, max_tokens)) -> "
            "TextResult(.text)\n"
            "  - detection -> tools.detect(DetectionRequest(image, query, threshold)) -> "
            "DetectionResult(.detections), each Detection(.label, .score, .box_xyxy)\n"
            "  - pointing -> tools.point(PointingRequest(image, instruction)) -> "
            "PointingResult(.points_xy, .confidence) in pixel coordinates\n"
            "  - segmentation -> tools.segment(SegmentationRequest(image, boxes_xyxy=..., points_xy=..., labels=...)) "
            "-> SegmentationResult(.masks bool [N,H,W], .scores)\n"
            "  - grasp -> tools.grasp(GraspRequest(rgb, depth_m, intrinsics, camera_to_world, mask)) -> "
            "GraspResult(.candidates), each GraspCandidate(.pose_world 4x4, .score, .width_m). Needs 3D.\n"
            "  A tool that is not served on this route raises ToolUnavailableError, so guard every call.\n\n"
            "## Harness helpers you may import\n"
            "The scaffold runs with the project on its import path, so `from robot_auto_evolve...` works. Two modules "
            "exist specifically for you, and both are plain arithmetic with no hidden state:\n"
            "- robot_auto_evolve.agent.geometry -- turns a pixel plus its depth into a 3D point in the robot's own "
            "frame, and back. `pixel_to_world(camera, u, v)`, `depth_at(camera, u, v, radius)`, "
            "`world_to_pixel(camera, point)`. It handles the per-route camera convention for you, so you do not have "
            "to work out which way is up.\n"
            "- robot_auto_evolve.agent.motion -- a small fixed set of movement commands you can issue INSTEAD OF, or "
            "alongside, the policy's action: `controller = make_controller(spec)` from any action spec (take it from a "
            "policy chunk's `.spec`), then `controller.move_to(observation, target_xyz)`, "
            "`controller.set_gripper(observation, closed=True/False)`, `controller.hold(observation)`, "
            "`controller.nudge(observation, delta_xyz)`. Each returns a float32 array of action values you wrap in a "
            "CanonicalActionChunk yourself (see the module docstring for a five-line example). "
            "`make_controller` returns None if this route's action layout is not supported -- check for that.\n"
            "Read both modules before using them.\n\n"
            "## One rule about calling the policy that WILL bite you if you skip it\n"
            "Ask the policy EVERY step, even on a step whose action you intend to throw away. The policy "
            "service remembers which step it last produced an action for and rejects a request that skips "
            "ahead (`policy_act: previous action is not observed as executed`) -- that check is what makes "
            "\"the frozen policy produced the action\" verifiable. So there are exactly two safe shapes: "
            "call tools.vla every step and sometimes return something else instead of its action, or never "
            "call it at all for the whole episode. MIXING -- calling it, skipping some steps, then calling it "
            "again -- fails every episode of your candidate. Discarding an action you asked for costs only "
            "the inference, and the policy's own state then advances exactly as it otherwise would.\n\n"
            "## Learn from the FULL raw trajectories (your primary evidence)\n"
            "Read ../public_input.json for the scores and trace paths. The incumbent's raw per-episode traces are in "
            "the directory named by incumbent_episode_traces_dir -- read them DIRECTLY with Bash/Grep/Read and analyze "
            "them with python; there is NO pre-made summary, so build your own understanding from the raw logs. Each "
            "episode folder holds:\n"
            "  - trace.jsonl : the full step-by-step decision trace. Line 0 is the episode header carrying the "
            "GROUND-TRUTH success + termination; each later line is one step with the instruction fed to the policy, "
            "the action values, and every tool call with its result.\n"
            "  - episode.json : the episode outcome (ground-truth success) + the exact task/seed/scenario key.\n"
            "  - private_metrics.json : extra ground-truth metrics for that episode (e.g. progress_score) -- present "
            "ONLY for routes that report such metrics; many routes have just episode.json's success, so do not rely on "
            "this file existing.\n"
            "  - frame-<step>.png : the first and last rendered camera frames.\n"
            "If public_input names a previous rejected candidate, its traces are there too, and they are worth reading "
            "-- a rejected change still tells you what the robot did.\n\n"
            "## Hard constraints (these define a VALID result)\n"
            "- The scaffold you write runs live inside a fresh episode where it ONLY ever receives the stripped "
            "FairObservation. You MAY study every recorded ground-truth outcome above to guide your design, but the "
            "scaffold itself must NOT read privileged ground-truth DURING a rollout: no live simulator state, no true "
            "object/goal poses, no success predicate, no _check_success, no sim.data / body_xpos / site_xpos / "
            "geom_xpos, no BDDL goal predicates, no expert actions. Such reads are statically rejected. Do not smuggle "
            "the answer into the scaffold.\n"
            "- Do NOT hardcode a task-specific solution. This is the rule that matters most, and it gets easier to "
            "break the more freedom you have. A general RULE derived from evidence -- split a compound instruction, "
            "strip a phrase the scene layout has invalidated, debounce the gripper, move toward wherever the detector "
            "says the target is -- carries over to a task you have never seen. A LOOKUP -- a table keyed by task name "
            "or task id, a coordinate you read off a trace and typed in, a per-episode special case -- does not, and "
            "will show up as a rising in-loop score with a flat held-out score. In particular, any spatial target you "
            "move toward must be COMPUTED from this episode's own observation (perception, depth, proprioception), "
            "never a numeric position written into the source.\n"
            "- Do NOT change the action space, the success check, or the evaluation. The action chunk you return must "
            "carry the route's own action spec and stay within the horizon and execution-count limits the harness "
            "enforces.\n\n"
            "## How this run is scored, and your budget\n"
            "When you stop, the harness re-evaluates your scaffold on the same fixed episode set and compares it with "
            "the incumbent; you do not have to score it and there is no way for you to influence that measurement. "
            f"You have {self.revision_max_turns} turns and {self.revision_timeout_s:.0f} seconds of wall clock. That "
            "is a large budget and past runs have used well under a fifth of it. Spend it: read more traces, write "
            "throwaway analysis scripts, check a hypothesis against the numbers before you commit to it. If you want "
            "to look at the environment or the policy code first-hand, you may -- nothing is off limits to read.\n"
            + route_section
        )

    def _run_candidate(self, state: dict[str, Any]) -> dict[str, Any]:
        index = state["next_candidate"]
        staging = self.run_dir / "candidates" / f".{index:04d}-staging"
        # RESUME: if this candidate was interrupted AFTER the coding agent finished and its
        # evaluation had begun, keep the already-revised scaffold and re-enter the evaluation
        # (its finished episodes are skipped). If it was interrupted mid-revision (no evaluation
        # output yet), it is NOT resumable -> archive the partial and redo from scratch (re-copy
        # the incumbent, re-invoke the coding agent).
        resuming = _resumable(staging)
        if staging.exists() and not resuming:
            self._archive(staging, f"candidate-{index:04d}-interrupted", "restarting candidate (no evaluation to resume)")
        if not resuming:
            staging.mkdir()
        try:
            incumbent = self._reference(state["incumbent"])
            if resuming:
                self._progress(
                    f"candidate {index:04d}: RESUMING evaluation of the already-revised scaffold "
                    "(skipping finished episodes) ..."
                )
            else:
                shutil.copytree(incumbent / "scaffold", staging / "scaffold")
                (staging / "scaffold" / "scaffold.py").chmod(0o600)
                prompt = self._revision_material(state, index, staging)
                (staging / "revision_prompt.txt").write_text(prompt, encoding="utf-8")
                self._progress(f"candidate {index:04d}: invoking coding agent (incumbent={state['incumbent']}) ...")
                self.revision_backend.revise(prompt, staging / "scaffold", staging / "revision_logs", index)
                self._progress(f"candidate {index:04d}: coding agent returned a revised scaffold; evaluating ...")
            candidate_hashes = self.editable.validate_revision(incumbent / "scaffold", staging / "scaffold")
            result = self._evaluate(staging / "scaffold", staging / "benchmark")
            incumbent_result = self._validate_result(incumbent)
            decision = ScalarDecision.create(incumbent_result.scalar.value, result.scalar.value)
            _write_json(staging / "decision.json", decision.to_mapping())
            _write_json(staging / "candidate_hashes.json", candidate_hashes)
            self._progress(
                f"candidate {index:04d}: {self.scalar_metric}={result.scalar.value:.4f} "
                f"vs incumbent {incumbent_result.scalar.value:.4f} "
                f"-> {'ACCEPTED (new incumbent)' if decision.accepted else 'rejected (incumbent kept)'}"
            )
            self._metrics_row(
                "candidate", index, f"{result.scalar.value:.6f}",
                f"{incumbent_result.scalar.value:.6f}", decision.accepted, len(result.outcomes),
            )
            post = dict(state)
            post["next_candidate"] = index + 1
            if decision.accepted:
                post["incumbent"] = f"candidates/{index:04d}"
            return self._commit(staging, self.run_dir / "candidates" / f"{index:04d}", post)
        except (KeyboardInterrupt, SystemExit):
            # Interrupt: never a candidate rejection. Leave the partial staging in place so the
            # next run resumes it (if its evaluation had started) or redoes it (if interrupted
            # mid-revision). SIGTERM = RunInterrupted (a BaseException) likewise leaves it intact.
            raise
        except Exception as exc:
            # Reject-and-continue: a broken candidate revision (invalid agent_event, fairness
            # violation, non-compiling scaffold, episode errors, revision timeout, ...) is an
            # EXPECTED occasional event across a multi-candidate run. Archive it under failures/
            # for inspection, count it as a REJECTED attempt (incumbent unchanged), advance
            # next_candidate, and let the loop proceed instead of crashing the whole run.
            if staging.exists():
                self._archive(staging, f"candidate-{index:04d}-failed", f"{type(exc).__name__}: {exc}")
            self._progress(
                f"candidate {index:04d}: FAILED ({type(exc).__name__}: {exc}) "
                f"-> archived to failures/, incumbent kept, continuing"
            )
            self._metrics_row("candidate", index, "broken", "-", "-", "-")
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
            frozen = self._commit(staging, self.run_dir / "frozen", post)
            self._progress(f"FROZEN: incumbent {state['incumbent']} is the final scaffold")
            return frozen
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
        # RESUME: exist_ok=True re-enters a partial transfer left by an interrupt. Each _evaluate below
        # runs through the evaluator's pending-set skip, so a completed baseline_transfer (its final.json
        # present) is skipped and a partially-evaluated one resumes at its first unfinished episode --
        # the held-out set (up to 2xN episodes) is never redone from scratch.
        staging.mkdir(exist_ok=True)
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
            self._progress(
                f"transfer (held-out): baseline {baseline.scalar.value:.4f} vs frozen "
                f"{evolved.scalar.value:.4f}  (reported only; never affects acceptance)"
            )
            return comparison
        except (KeyboardInterrupt, SystemExit):
            # Interrupt: leave the partial transfer staging in place so the next run resumes it
            # (skipping finished held-out episodes). SIGTERM = RunInterrupted (a BaseException, not
            # Exception) likewise bypasses both handlers and propagates without archiving.
            raise
        except Exception as exc:
            # A real transfer error (not an interrupt) is archived for inspection, then re-raised.
            if staging.exists():
                self._archive(staging, "transfer-failed", f"{type(exc).__name__}: {exc}")
            raise
