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
from .editable import EditablePolicy


REVISION_PROMPT = (
    "You are a robotics engineer. Your job is to improve one Python program, `scaffold.py`, which is the robot's control agent: every step of an episode it receives what the robot can see and returns the numbers that move the robot. An episode is one attempt at one task, from reset until the robot succeeds or runs out of steps.\n"
    '\n'
    'Several frozen models are available to `scaffold.py`. One is a robot policy, trained to turn a camera picture plus a task sentence into robot motion; the others see or read. You cannot change any of them and you must not train or fine-tune anything. How you use them, and whether you use them at all, is yours to decide: the numbers you return can come from a model, from arithmetic you wrote, or from any mixture.\n'
    '\n'
    'Everything between the observation arriving and the action leaving is yours to change.\n'
    '\n'
    '\n'
    '## What to aim at\n'
    'Make the robot succeed on more episodes. Work out what is going wrong first, from the recordings described below, and only then change the code. Say what the failure is before you fix it.\n'
    '\n'
    'Whatever you change must be a general improvement to the robot agent. It has to keep working on tasks and scenes you have never seen, not just the ones in these recordings.\n'
    '\n'
    'How much you change is your call -- one line or a rewrite. Write a short note of what you changed and why into `../agent_workspace/CHANGES.md`: what you saw in the recordings, what you concluded, and what you did about it. Finish by saying the same thing in your last message.\n'
    '\n'
    '## The one file that runs, and where everything else goes\n'
    '`scaffold.py` in your current directory is the whole robot agent, and it is the only file that runs. Edit it in place. Keep the agent in that one file only.\n'
    '\n'
    'Keep `create_scaffold()`, the `SCAFFOLD_CONFIG` dict and the `act(request, tools)` signature. `SCAFFOLD_CONFIG` keeps its two lists, and no name may be in both. A name in `required_capabilities` is loaded every time. A name in `optional_capabilities` is loaded only when your file mentions it. A name in neither is never loaded and calling it fails. `"vla"` is the robot policy and follows the same rule, so move it to `optional_capabilities` if you stop calling it.\n'
    '\n'
    'Everything else you produce -- notes, analysis scripts, plots, extracted data, rendered pictures -- goes in `../agent_workspace/`. That folder is yours, put scratch files there rather than in `/tmp`.\n'
    '\n'
    'Your scaffold cannot keep a memory file: anything it writes to disk while an episode is running is thrown away. Python values stored on the scaffold object may or may not survive into the next episode. Clear anything episode-specific in `reset(session_id)`, and do not rely on carrying state from one episode to the next.\n'
    '\n'
    '\n'
    '## The interface (read `scaffold.py` first)\n'
    "`act(request, tools)` is called once per step and must return exactly one `CanonicalActionChunk`. Its `spec` must be this setup's own action spec. Its `start_step` must equal `request.observation.step_index`. Its values must obey this setup's limits, which the last section lists. The class is defined in `robot_auto_evolve/protocol/action.py`.\n"
    '\n'
    '`request.observation` is everything the robot can see this step:\n'
    '  - `.instruction` -- the task sentence, word for word as the task gives it.\n'
    '  - `.cameras` -- a mapping from camera name to a camera picture.\n'
    "  - `.proprioception` -- the robot's own readings, typically its gripper pose and how open the gripper is. Which entries exist depends on the setup. Look one up by name with `.by_name(...)`, or read `.vectors`, where each entry has a `.spec` describing what its numbers mean and `.values` holding them.\n"
    '  - `.step_index` -- which step of the episode this is.\n'
    'There is nothing else in it. No object positions, no goal, no score.\n'
    '\n'
    "Each camera has `.rgb` (uint8, height x width x 3). Where this setup has depth sensing it also has `.depth_m` (float32, height x width, in metres, one value per colour pixel), `.depth_valid`, `.intrinsics` (3x3) and `.camera_to_world` (4x4). Those four together let you turn a pixel into a point in the robot's own coordinates.\n"
    '\n'
    'The exact definitions are in `robot_auto_evolve/agent/api.py` (the request and result types), `robot_auto_evolve/agent/tools.py` (the `tools` object) and `robot_auto_evolve/protocol/observation.py` (the observation). Read them rather than guessing.\n'
    '\n'
    '\n'
    '## The models you can call\n'
    'Every call goes through the `tools` object. `robot_auto_evolve/agent/tools.py` is the implementation and `robot_auto_evolve/agent/api.py` holds every request and result type, with the exact fields and the checks each one applies. Read both before writing a call.\n'
    '\n'
    '- `tools.vla(...)` -- the robot policy. You can build the whole agent around it, use it on some steps only, or never call it at all. Three of its arguments are worth knowing about before you read the file:\n'
    '  - `instruction=` is the sentence the policy is given. The policy service feeds it to the model only on a step where it actually runs the model, so changing the text on any other step does nothing until then. See `refresh` next, and read `robot_auto_evolve/policies/` for the service this setup uses.\n'
    "  - `refresh=True` throws away whatever the policy service had left over and makes it run the model again on this step's picture. The service runs the model once, gets several future actions back, and hands them out one per step; that is why a new instruction can sit unused for several steps. A refresh is a change in behaviour, not just a recomputation: anything the discarded actions were carrying is gone with them.\n"
    "  - `context=` is a tuple of strings sent to the policy service. It is ignored unless the last section says this setup's policy reads it.\n"
    '  - If you want the policy to know something, it has to be in the instruction text or in the observation.\n'
    '\n'
    '- `tools.vision(...)` -- a text model that can see pictures. Describe a scene, compare two views, judge whether something is done.\n'
    '\n'
    '- `tools.language(...)` -- a text model. It cannot see pictures.\n'
    '\n'
    '- `tools.detect(...)` -- an object detector. Give it a picture and words; it returns boxes with labels and scores.\n'
    '\n'
    '- `tools.point(...)` -- a pointer. Give it a picture and an instruction; it returns pixel coordinates.\n'
    '\n'
    '- `tools.segment(...)` -- a segmenter. Give it a picture and at least one box or point; it returns outlines.\n'
    '\n'
    'Check with `tools.has("detection")` and so on before calling; the last section lists which models are available here. Calling one that is not running raises `ToolUnavailableError`, and so does a call that fails for any other reason, such as a timeout or a malformed request. Any exception that escapes `act` ends that episode and it counts as a failure, so guard your calls.\n'
    '\n'
    "`tools.record(event_type, status, detail, capability, result)` writes a line into this episode's recording. Every model call is recorded for you, answer included, so you do not have to log those yourself -- use `record` for your own decisions: what you concluded, what you chose to do, and why.\n"
    '\n'
    '\n'
    '## Two ready-made modules you can import\n'
    'Your scaffold runs with this project on its import path, so `from robot_auto_evolve...` works. numpy and PIL are available; no simulator package and no PyYAML are, so `robot_auto_evolve.benchmarks.*` will not import. Two modules are there for you to use, and both are plain arithmetic with no hidden state.\n'
    '\n'
    "- `robot_auto_evolve.agent.geometry` -- turns a pixel into a point in the robot's own coordinates, and back. Call `has_3d(camera)` first: it is False where this setup has no depth. Then `depth_at`, `pixel_to_world`, `world_to_pixel`, `point_cloud`. `pixel_to_world` returns `None` rather than raising when there is no usable depth at that pixel, so check the result. The module already accounts for which way up this setup stores its picture.\n"
    '\n'
    "- `robot_auto_evolve.agent.motion` -- ready-made movement commands you can return instead of a model's answer. `make_controller(spec)` gives you `move_to`, `nudge`, `hold` and `set_gripper`; each returns a float32 array of action numbers for one step, which you wrap in a `CanonicalActionChunk` yourself. `make_controller` returns `None` where the action layout is not one it can drive, so check for that; the last section says whether it can drive this setup. On a setup whose action is a change from the current pose these commands leave the wrist rotation alone; on a setup whose action is a target pose they reuse the rotation from the last chunk you passed to `controller.note(chunk)`.\n"
    '\n'
    "These commands are a convenience, not a limit. You may also compute the action numbers yourself and return them, as long as the chunk carries this setup's own spec and its values obey this setup's limits.\n"
    'Read both modules before using them.\n'
    '\n'
    '\n'
    '## If you call the policy, call it on every step\n'
    'The policy service remembers which step it last produced an action for and refuses a request that jumps ahead, with `policy_act: previous action is not observed as executed`. Do not try to remove that check.\n'
    '\n'
    'So there are exactly two shapes that work: call `tools.vla` on every step, and on any step return something else instead of its action if you want to -- or never call it at all for the whole episode. Mixing the two, by calling it, skipping some steps and calling it again, fails every episode. Throwing away an action you asked for costs only the time it took to compute.\n'
    '\n'
    'None of this applies to the other models. You can call or skip those freely, on any step.\n'
    '\n'
    '\n'
    '## The recordings are your evidence\n'
    '`../public_input.json` gives you the folders that hold them.\n'
    '  - `incumbent_episode_traces_dir` -- the recordings of the `scaffold.py` you have now, running the whole episode set.\n'
    '  - `previous_rejected_candidate_episode_traces_dir` -- present only sometimes. It is a change that was already tried and did not do better, with its own full recordings in the same shape.\n'
    '\n'
    'Read them yourself with Bash, Grep and Read, and analyse them with python. Each episode folder holds:\n'
    "  - `trace.jsonl` : line 0 names the task, says whether the episode succeeded and how it ended, and lists every camera with its size and the picture file that holds it. Every later line is one step, with `step`, `instruction` (what the policy was given), `action` (the numbers that were returned), `state` (the robot's own readings that step: gripper pose, and where the setup reports it, how far apart the fingers are), `depth` (per camera: how much of the depth picture was usable, and its smallest, middle and largest distance) and `events`.\n"
    "  - Each event is one thing that happened that step. A model call carries `result`, which holds what that model actually answered -- the detector's boxes with labels and scores, the pointer's pixels, a summary of each mask, the text a text model wrote. Events your scaffold wrote with `tools.record(...)` appear the same way.\n"
    '  - `camera-<name>.mp4` : every rendered frame of that camera for the whole episode. Pull any step out with ffmpeg, for example `ffmpeg -i camera-main.mp4 -vf "select=eq(n\\\\,63)" -vsync 0 -vframes 1 ../agent_workspace/step63.png`, and then open the PNG as an image. Line 0 of the trace names the file for each camera. These are compressed, so a pixel can be off by a shade.\n'
    '  - `episode.json` : whether that episode succeeded, how many steps it took, and exactly which task, scene and seeds it used.\n'
    '  - `private_metrics.json` : extra measurements for that episode, on the setups that produce any. Many produce none, so do not assume this file exists.\n'
    '\n'
    '\n'
    '## Running the simulator yourself\n'
    'You can re-run any recorded episode in the simulator and look at whatever you like. It reads the task, scene and seeds from `episode.json` and the actions from `trace.jsonl`, so it repeats the episode that was recorded. The last section gives the exact command for this setup.\n'
    '\n'
    '\n'
    '## Rules your scaffold has to follow\n'
    "- While you work you may read anything you need: the recordings, the simulator source, the policy code. Keep to this project and the paths `runtime_paths.json` names; nothing else on this machine is part of your task. But the scaffold you ship must not read the simulator's own answers while an episode is running -- no live simulator state, no true object or goal positions, no success check, no expert actions. In practice that means none of `_check_success`, `sim.data`, `body_xpos`, `site_xpos`, `geom_xpos`. Do not smuggle the answer into the scaffold.\n"
    '\n'
    '- Do not write a solution for the particular tasks in these recordings. A general rule names no particular task and no particular object, and would behave the same way on a task you have never seen. A lookup does not: a table keyed by task name or id, a coordinate you read off a recording and typed in, a special case for one episode. In particular, a place you move the gripper to must be worked out from this episode\'s own observation -- from what the camera sees, from depth, from the robot\'s own readings -- and never typed in as a number. A constant offset is a different thing and is fine: `nudge(observation, (0, 0, 0.03))` means "lift three centimetres from wherever the gripper is now", which behaves the same on any task.\n'
    '\n'
    '- Do not change how success is decided or how the result is counted. That is not in your file, and reaching for it will not work.\n'
    '\n'
    '\n'
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resumable(staging: Path) -> bool:
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
        if type(route_notes) is not str:
            raise StrictSchemaError("benchmark evolution route notes differ")
        self.route_notes = route_notes
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
            "plan_id": self.plan.plan_id,
            "model_route": self.plan.model_route,
            "n_evolve_episodes": len(self.plan.episodes),
            "scalar_metric": self.scalar_metric,
            "candidate_budget": self.candidate_budget,
            "editable_files": list(self.editable.allowed),
            "transfer_plan_id": None if self.transfer_plan is None else self.transfer_plan.plan_id,
            "n_transfer_episodes": 0 if self.transfer_plan is None else len(self.transfer_plan.episodes),
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
        try:
            with open(self.run_dir / "progress.log", "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
        except OSError:
            pass

    def _metrics_row(
        self,
        phase: str,
        candidate: Any,
        score: Any,
        incumbent: Any,
        accepted: Any,
        episodes: Any,
        elapsed_s: Any = "-",
        errored: Any = "-",
    ) -> None:
        try:
            path = self.run_dir / "metrics.csv"
            header = not path.exists()
            with open(path, "a", encoding="utf-8") as handle:
                if header:
                    handle.write(
                        "phase,candidate,score,incumbent_score,accepted,n_episodes,n_errored,eval_seconds\n"
                    )
                handle.write(
                    f"{phase},{candidate},{score},{incumbent},{accepted},{episodes},{errored},{elapsed_s}\n"
                )
        except OSError:
            pass

    @staticmethod
    def _errored(result: BenchmarkEvaluationResult) -> int:
        try:
            return int(result.metadata.get("n_errored") or 0)
        except (AttributeError, TypeError, ValueError):
            return 0

    def _errored_warning(self, result: BenchmarkEvaluationResult) -> str:
        errored = self._errored(result)
        if not errored:
            return ""
        return (
            f"  WARNING: {errored} of {len(result.outcomes)} episodes ERRORED. Each was already run "
            "again automatically and died again. They are counted as failures in the score above, and "
            "this evaluation is now finished, so re-running the same command will NOT try them once "
            "more. Look at benchmark/canonical/episodes/*/episode.json for the error text before "
            "trusting this number"
        )

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
        if result.plan_id != selected_plan.plan_id or result.scalar.metric != selected_metric:
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
            plan_id=selected_plan.plan_id,
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
            started = time.monotonic()
            result = self._evaluate(staging / "scaffold", staging / "benchmark")
            elapsed = time.monotonic() - started
            post = {
                "schema_version": 1,
                "phase": "active",
                "next_candidate": 1,
                "incumbent": "baseline",
            }
            self._commit(staging, self.run_dir / "baseline", post)
            self._progress(
                f"baseline: DONE  {self.scalar_metric}={result.scalar.value:.4f}  "
                f"({len(result.outcomes)} episodes, {elapsed:.0f}s)" + self._errored_warning(result)
            )
            self._metrics_row(
                "baseline", 0, f"{result.scalar.value:.6f}", "-", "-", len(result.outcomes),
                f"{elapsed:.0f}", self._errored(result),
            )
        except (KeyboardInterrupt, SystemExit):
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
            state = {"schema_version": 1, "phase": "active", "next_candidate": 1, "incumbent": "baseline"}
        else:
            self._prepare_baseline()
            return
        while (self.run_dir / "candidates" / f"{state['next_candidate']:04d}").exists():
            candidate = self.run_dir / "candidates" / f"{state['next_candidate']:04d}"
            decision = ScalarDecision.from_mapping(_load_json(candidate / "decision.json"))
            if decision.accepted:
                state["incumbent"] = f"candidates/{state['next_candidate']:04d}"
            state["next_candidate"] += 1
        if (self.run_dir / "frozen").exists():
            state["phase"] = "frozen"
        self._write_state(state)
        self._archive(self.run_dir / ".frozen-staging", "freeze-interrupted", "uncommitted staging")

    def _retain_agent_workspace(self, staging: Path) -> int:
        scaffold = staging / "scaffold"
        workspace = staging / "agent_workspace"
        if not scaffold.is_dir():
            return 0
        moved = 0
        for path in sorted(scaffold.iterdir()):
            if path.name == "scaffold.py":
                continue
            workspace.mkdir(exist_ok=True)
            target = workspace / path.name
            suffix = 1
            while target.exists():
                suffix += 1
                target = workspace / f"{path.name}.{suffix}"
            os.rename(path, target)
            moved += 1
        return moved

    def _revision_material(self, state: Mapping[str, Any], index: int, staging: Path) -> str:
        incumbent = self._reference(state["incumbent"])
        incumbent_result = self._validate_result(incumbent)
        incumbent_episodes = incumbent / "benchmark" / "canonical" / "episodes"
        previous = None
        previous_episodes = None
        if index > 1:
            previous_root = self._reference(f"candidates/{index - 1:04d}")
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
        route_section = f"## Facts about THIS setup\n{self.route_notes.strip()}\n" if self.route_notes.strip() else ""
        return REVISION_PROMPT + route_section

    def _run_candidate(self, state: dict[str, Any]) -> dict[str, Any]:
        index = state["next_candidate"]
        staging = self.run_dir / "candidates" / f".{index:04d}-staging"
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
                (staging / "agent_workspace").mkdir(exist_ok=True)
                self._progress(f"candidate {index:04d}: invoking coding agent (incumbent={state['incumbent']}) ...")
                try:
                    self.revision_backend.revise(prompt, staging / "scaffold", staging / "revision_logs", index)
                finally:
                    kept = self._retain_agent_workspace(staging)
                self._progress(
                    f"candidate {index:04d}: coding agent returned a revised scaffold "
                    f"({kept} scratch item(s) kept in agent_workspace/); evaluating ..."
                )
            candidate_files = self.editable.validate_revision(incumbent / "scaffold", staging / "scaffold")
            started = time.monotonic()
            result = self._evaluate(staging / "scaffold", staging / "benchmark")
            elapsed = time.monotonic() - started
            incumbent_result = self._validate_result(incumbent)
            decision = ScalarDecision.create(incumbent_result.scalar.value, result.scalar.value)
            _write_json(staging / "decision.json", decision.to_mapping())
            _write_json(staging / "candidate_files.json", candidate_files)
            self._progress(
                f"candidate {index:04d}: {self.scalar_metric}={result.scalar.value:.4f} "
                f"vs incumbent {incumbent_result.scalar.value:.4f} "
                f"-> {'ACCEPTED (new incumbent)' if decision.accepted else 'rejected (incumbent kept)'}"
                f"  ({elapsed:.0f}s to evaluate)" + self._errored_warning(result)
            )
            self._metrics_row(
                "candidate", index, f"{result.scalar.value:.6f}",
                f"{incumbent_result.scalar.value:.6f}", decision.accepted, len(result.outcomes),
                f"{elapsed:.0f}", self._errored(result),
            )
            post = dict(state)
            post["next_candidate"] = index + 1
            if decision.accepted:
                post["incumbent"] = f"candidates/{index:04d}"
            return self._commit(staging, self.run_dir / "candidates" / f"{index:04d}", post)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
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

    def _reopen_frozen(self, state: Mapping[str, Any], target_candidates: int, completed: int) -> dict[str, Any]:
        index = 1
        history_root = self.run_dir / "superseded"
        history_root.mkdir(exist_ok=True)
        while (history_root / f"{index:04d}").exists():
            index += 1
        history = history_root / f"{index:04d}"
        history.mkdir()
        _write_json(
            history / "superseded.json",
            {
                "schema_version": 1,
                "reason": "reopened for more candidates",
                "completed_candidates": completed,
                "requested_candidates": target_candidates,
                "incumbent_at_freeze": state["incumbent"],
                "superseded_ns": time.time_ns(),
            },
        )
        frozen = self.run_dir / "frozen"
        if frozen.exists():
            os.rename(frozen, history / "frozen")
        transfer = self.run_dir / "transfer"
        staging = self.run_dir / ".transfer-staging"
        reused_baseline_half = False
        if transfer.is_dir():
            if staging.exists():
                self._archive(staging, "transfer-superseded", "replaced by the committed transfer on reopen")
            os.rename(transfer, staging)
            for name in ("evolved_transfer", "evolved_transfer_result.json", "transfer_comparison.json"):
                source = staging / name
                if source.exists():
                    os.rename(source, history / name)
            reused_baseline_half = (staging / "baseline_transfer" / "canonical" / "final.json").is_file()
        post = {**dict(state), "phase": "active"}
        self._write_state(post)
        self._progress(
            f"REOPENED: this run was frozen after {completed} candidate(s) and has been asked for "
            f"{target_candidates}. The old frozen scaffold and the held-out half scored against it are kept in "
            f"superseded/{index:04d}/. The evolve baseline is reused as-is; the held-out BASELINE half is "
            + ("reused, so it will not be scored again." if reused_baseline_half
               else "NOT available to reuse, so it will be scored again.")
        )
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
            if target_candidates > completed:
                state = self._reopen_frozen(state, target_candidates, completed)
            else:
                raise RuntimeError("benchmark evolution run is already frozen")
        if target_candidates < completed:
            raise RuntimeError("target candidates is below the completed count")
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
        resuming = staging.is_dir()
        staging.mkdir(exist_ok=True)
        self._progress(
            "transfer (held-out): "
            + (
                "RESUMING the held-out evaluation (skipping finished episodes) ..."
                if resuming
                else "scoring the baseline scaffold and the frozen scaffold on the held-out set ..."
            )
        )
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
                + self._errored_warning(baseline)
                + self._errored_warning(evolved)
            )
            return comparison
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if staging.exists():
                self._archive(staging, "transfer-failed", f"{type(exc).__name__}: {exc}")
            raise
