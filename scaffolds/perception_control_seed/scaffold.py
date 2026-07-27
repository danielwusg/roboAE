from __future__ import annotations

import numpy as np

from robot_auto_evolve.agent import (
    AgentRequest,
    DetectionRequest,
    SegmentationRequest,
    ToolboxProtocol,
    ToolUnavailableError,
    VLARequest,
)
from robot_auto_evolve.agent.geometry import has_3d, pixel_to_world
from robot_auto_evolve.agent.motion import make_controller
from robot_auto_evolve.protocol import CanonicalActionChunk


# A structural starting agent: perception decides WHEN to act, it never writes a sentence.
#
# The older planner seed (scaffolds/volo_harness_seed) ran a language model every 32 steps and
# replaced the benchmark's own instruction with its output. Measured over eight finished runs
# that made the frozen policy worse on four routes out of five, because the rewritten sentence
# was off-distribution for a policy trained on the benchmark's own phrasing. This seed keeps the
# useful half of that design -- watch whether the robot is actually making progress, look at the
# scene now and then, and intervene when it is stuck -- and drops the half that hurt: there is no
# language model anywhere in this file and the instruction handed to the policy is always the
# benchmark's own, unchanged.
#
# What it does, one step at a time:
#   1. The free check, every step. Compare this camera frame with the last and this joint vector
#      with the last. That answers "is anything moving at all" for no model calls.
#   2. The real look, occasionally. Ask the detector where the thing named in the task is. The
#      task text comes straight from the benchmark and the detector takes open-ended text, so no
#      language model is needed to form the query. With 3D on, turn the answer into a point in
#      the robot's own frame.
#   3. Judge: running normally, stalled, or the target cannot be found.
#   4. Act.
#        normal  -> ask the policy and pass its action through untouched. Deliberately identical
#                   to the bare-policy seed, so on a healthy episode this agent IS the policy.
#        stalled -> intervene, mildest first. First just tell the policy to start fresh
#                   (refresh=True clears its cached action chunk). If it is still stuck after
#                   that, and perception has a point and this route supports movement commands,
#                   drive the gripper toward the point for a bounded burst of steps.
#        lost    -> keep asking the policy, but look again sooner.
#   5. Record what it decided, so the coding agent can read it back out of the trace.
#
# A note on one thing that is deliberately absent. Two earlier runs independently discovered that
# the old planner seed's recovery re-fires every single step once stagnation latches, so a stuck
# episode spends its whole horizon replanning, and both patched a cooldown on top. That is a
# defect in a seed we wrote, not a finding about robots, so it is designed out here rather than
# shipped and patched: every intervention runs for a fixed number of steps and then hands control
# back, and re-triggering needs fresh evidence of stalling.
#
# Also deliberately absent: gripper latching or debouncing, action smoothing, instruction
# rewriting or scrubbing, and any per-task text handling. Those were things the coding agent
# found for itself in earlier runs; putting them in the starting point would hand it its own
# answers and leave nothing to measure.
SCAFFOLD_CONFIG = {
    "required_capabilities": ("vla",),
    "optional_capabilities": ("language", "vision", "detection", "segmentation", "pointing", "grasp"),
}

# How many consecutive still checks count as stalled.
STALL_CHECKS = 6
# Fraction of full-scale pixel change / radians of joint change below which a step counts as still.
IMAGE_STILL = 0.002
JOINT_STILL = 0.002
# How often to spend a detector call, in steps.
LOOK_EVERY = 16
# How long one intervention lasts before control goes back to the policy.
REFRESH_BURST = 1
MOVE_BURST = 8
# How far one movement step may command the gripper to travel, in metres.
MOVE_STEP_M = 0.03


class _Session:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.joints: np.ndarray | None = None
        self.step_index = -1
        self.still_steps = 0
        self.last_look = -(10**9)
        self.target: np.ndarray | None = None
        self.target_step = -(10**9)
        self.target_pixel: tuple[float, float] | None = None
        self.move_until = -1
        self.tried_refresh_at = -(10**9)
        self.controller = None


class PerceptionControlSeed:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # -- step 1: the free check -----------------------------------------------------------

    def _still(self, session: _Session, request: AgentRequest) -> bool:
        observation = request.observation
        camera = observation.cameras[sorted(observation.cameras)[0]]
        image = camera.rgb[::16, ::16].copy()
        joints = np.concatenate([vector.values for vector in observation.proprioception.vectors])
        if session.step_index >= observation.step_index:
            session.image = None
            session.joints = None
            session.still_steps = 0
        still = False
        if session.image is not None and session.joints is not None:
            image_delta = float(np.abs(image.astype(np.int16) - session.image.astype(np.int16)).mean()) / 255.0
            joint_delta = float(np.linalg.norm(joints - session.joints))
            still = image_delta < IMAGE_STILL and joint_delta < JOINT_STILL
        session.still_steps = session.still_steps + 1 if still else 0
        session.image = image
        session.joints = joints
        session.step_index = observation.step_index
        return session.still_steps >= STALL_CHECKS

    # -- step 2: the real look ------------------------------------------------------------

    def _look(self, session: _Session, request: AgentRequest, tools: ToolboxProtocol) -> None:
        observation = request.observation
        step = observation.step_index
        if step - session.last_look < LOOK_EVERY:
            return
        session.last_look = step
        if not tools.has("detection"):
            tools.record("perception", "skipped", "no detector is served on this route", "detection")
            return
        name = sorted(observation.cameras)[0]
        camera = observation.cameras[name]
        try:
            found = tools.detect(DetectionRequest(camera.rgb, observation.instruction))
        except ToolUnavailableError as error:
            tools.record("perception", "optional_error", f"detector call failed: {error}", "detection")
            return
        if not found.detections:
            tools.record("perception", "ok", "detector found nothing above threshold", "detection")
            session.target = None
            session.target_pixel = None
            return
        best = max(found.detections, key=lambda item: item.score)
        x0, y0, x1, y1 = best.box_xyxy
        centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        session.target_pixel = centre
        # A segmentation mask gives a better centre than a box centre when the object is not
        # box-shaped, so use it when the segmenter is served.
        if tools.has("segmentation"):
            try:
                masks = tools.segment(SegmentationRequest(camera.rgb, boxes_xyxy=(best.box_xyxy,)))
                if masks.masks.shape[0]:
                    mask = masks.masks[int(masks.scores.argmax())]
                    rows, columns = np.nonzero(mask)
                    if columns.size:
                        centre = (float(columns.mean()), float(rows.mean()))
                        session.target_pixel = centre
            except ToolUnavailableError as error:
                tools.record("perception", "optional_error", f"segmenter call failed: {error}", "segmentation")
        point = None
        if has_3d(camera):
            point = pixel_to_world(camera, centre[0], centre[1])
        session.target = None if point is None else np.asarray(point, dtype=np.float64)
        session.target_step = step
        if session.target is None:
            tools.record(
                "perception",
                "ok",
                f"target '{best.label}' at pixel ({centre[0]:.1f},{centre[1]:.1f}) score={best.score:.3f}; "
                "no metric 3D on this route, so it stays a pixel",
                "detection",
            )
        else:
            tools.record(
                "perception",
                "ok",
                f"target '{best.label}' at pixel ({centre[0]:.1f},{centre[1]:.1f}) score={best.score:.3f} "
                f"-> robot frame ({session.target[0]:.4f},{session.target[1]:.4f},{session.target[2]:.4f})",
                "detection",
            )

    # -- steps 3-5 -------------------------------------------------------------------------

    def act(self, request: AgentRequest, tools: ToolboxProtocol) -> CanonicalActionChunk:
        session = self._sessions.setdefault(request.session_id, _Session())
        observation = request.observation
        step = observation.step_index
        stalled = self._still(session, request)
        self._look(session, request, tools)

        in_burst = session.controller is not None and step < session.move_until and session.target is not None
        start_burst = False
        refresh = step == 0
        if stalled and not in_burst:
            if step - session.tried_refresh_at > STALL_CHECKS + REFRESH_BURST:
                # Mildest intervention first: make the policy think again from this frame.
                session.tried_refresh_at = step
                session.still_steps = 0
                refresh = True
                tools.record("recovery", "triggered", "nothing moved for six checks; refreshing the policy")
            elif session.controller is not None and session.target is not None:
                # The refresh did not help. Take over for a bounded burst.
                session.move_until = step + MOVE_BURST
                session.still_steps = 0
                start_burst = True
                tools.record(
                    "recovery",
                    "triggered",
                    f"still stuck after a refresh; taking over for {MOVE_BURST} steps toward the "
                    "detected target",
                )
            else:
                # Nothing better available: look again sooner and let the policy carry on.
                session.last_look = -(10**9)
                session.still_steps = 0
                tools.record(
                    "decision",
                    "skipped",
                    "stuck, but there is no usable target or no movement support on this route; "
                    "looking again and continuing with the policy",
                )

        # ASK THE POLICY EVERY SINGLE STEP, even on a step whose action we are going to discard.
        # This is not optional. The policy services track which step they last produced an action
        # for and reject a request that skips ahead ("policy_act: previous action is not observed
        # as executed"), because that check is what makes "the frozen policy produced the action"
        # verifiable. Calling it and throwing the answer away keeps its internal state advancing
        # exactly as it otherwise would, and costs only the inference. The alternative -- never
        # calling it at all for a whole episode -- is also fine; it is MIXING that breaks.
        action = tools.vla(
            VLARequest(
                request_id=request.request_id,
                session_id=request.session_id,
                observation=observation,
                instruction=observation.instruction,
                context=(),
                refresh=refresh,
            )
        )
        if action.horizon != 1 or action.execution_count != 1:
            raise RuntimeError("native policy service must return exactly one executable action per step")
        if session.controller is None:
            session.controller = make_controller(action.spec)
        if session.controller is not None:
            session.controller.note(action)

        if (in_burst or start_burst) and session.controller is not None and session.target is not None:
            values = session.controller.move_to(observation, session.target, max_step_m=MOVE_STEP_M)
            if in_burst:
                tools.record(
                    "recovery",
                    "triggered",
                    f"moving the gripper toward the detected target; burst ends at step {session.move_until}",
                )
            return CanonicalActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                start_step=step,
                spec=session.controller.spec,
                values=np.asarray(values, dtype=np.float32)[None, :],
                execution_count=1,
            )

        # Normal: the policy drives, with the benchmark's own instruction, unchanged.
        return action


def create_scaffold() -> PerceptionControlSeed:
    return PerceptionControlSeed()
