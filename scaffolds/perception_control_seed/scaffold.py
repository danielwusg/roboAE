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


# Each step: check whether anything is moving, look with the detector on a fixed cadence, and
# decide whether to pass the policy's action through or intervene.
SCAFFOLD_CONFIG = {
    "required_capabilities": ("vla",),
    "optional_capabilities": ("language", "vision", "detection", "segmentation", "pointing", "grasp"),
}

# Consecutive still checks that count as stalled.
STALL_CHECKS = 6
# Fraction of full-scale pixel change / radians of joint change below which a step counts as still.
IMAGE_STILL = 0.002
JOINT_STILL = 0.002
# Steps between detector calls.
LOOK_EVERY = 16
# Steps one intervention lasts before control returns to the policy.
REFRESH_BURST = 1
MOVE_BURST = 8
# Metres one movement step may command.
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

    def _still(self, session: _Session, request: AgentRequest) -> bool:
        """True once nothing has moved for STALL_CHECKS consecutive steps."""
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

    def _look(self, session: _Session, request: AgentRequest, tools: ToolboxProtocol) -> None:
        """Every LOOK_EVERY steps: locate the task's target and, where the route has 3D, turn it
        into a point in the frame the end effector is reported in."""
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
        # A mask centroid is a better centre than a box centre for a non-box-shaped object.
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
                session.tried_refresh_at = step
                session.still_steps = 0
                refresh = True
                tools.record("recovery", "triggered", "nothing moved for six checks; refreshing the policy")
            elif session.controller is not None and session.target is not None:
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
                session.still_steps = 0
                tools.record(
                    "decision",
                    "skipped",
                    "stuck, but there is no usable target or no movement support on this route; "
                    "continuing with the policy",
                )

        # Ask the policy on EVERY step, including steps whose action is discarded. The policy
        # service tracks which step it last produced an action for and rejects a request that
        # skips ahead ("policy_act: previous action is not observed as executed"). Calling it and
        # discarding the answer keeps its internal state advancing; never calling it at all for a
        # whole episode is also fine. Mixing the two fails every episode.
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

        return action


def create_scaffold() -> PerceptionControlSeed:
    return PerceptionControlSeed()
