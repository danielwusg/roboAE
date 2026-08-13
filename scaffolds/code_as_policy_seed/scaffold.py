from __future__ import annotations

import numpy as np

from robot_auto_evolve.agent import (
    AgentRequest,
    DetectionRequest,
    PointingRequest,
    SegmentationRequest,
    ToolboxProtocol,
    ToolUnavailableError,
)
from robot_auto_evolve.agent.geometry import has_3d, pixel_to_world
from robot_auto_evolve.agent.motion import make_controller
from robot_auto_evolve.protocol import CanonicalActionChunk


SCAFFOLD_CONFIG = {
    "required_capabilities": (),
    "optional_capabilities": ("language", "vision", "detection", "segmentation", "pointing", "grasp", "vla"),
}

LOOK_EVERY = 12
APPROACH_HEIGHT_M = 0.10
GRASP_DROP_M = 0.005
LIFT_M = 0.12
RELEASE_DROP_M = 0.04
STEP_M = 0.03
ARRIVED_M = 0.015
CLOSE_STEPS = 6
OPEN_STEPS = 4
PHASE_LIMIT = 25

_STOP_WORDS = (
    "pick up the",
    "pick up",
    "place it on top of the",
    "place it on the",
    "place it in the",
    "place it on",
    "place it in",
    "put the",
    "put it on the",
    "put it in the",
    "move the",
    "open the",
    "close the",
    "the",
    "a ",
    "an ",
)

_DESTINATION_MARKERS = (
    " on top of ",
    " into the ",
    " inside the ",
    " in the ",
    " on the ",
    " onto ",
    " into ",
    " inside ",
    " near ",
    " to the ",
)


_TRAILING = (
    "and place it",
    "and put it",
    "and place",
    "and put",
    "then place it",
    "then put it",
    "and",
    "then",
)


def _clean(text: str) -> str:
    value = " ".join(str(text).lower().replace("_", " ").split())
    for word in _STOP_WORDS:
        if value.startswith(word):
            value = value[len(word):].strip()
            break
    changed = True
    while changed:
        changed = False
        for word in _TRAILING:
            if value.endswith(" " + word) or value == word:
                value = value[: len(value) - len(word)].strip()
                changed = True
                break
    return value.strip(" .,")


def _phrases(instruction: str) -> tuple[str, str | None]:
    text = " ".join(str(instruction).lower().replace("_", " ").split())
    best = None
    for marker in _DESTINATION_MARKERS:
        index = text.find(marker)
        if index > 0 and (best is None or index < best[0]):
            best = (index, marker)
    if best is None:
        return (_clean(text) or text, None)
    index, marker = best
    target = _clean(text[:index])
    destination = _clean(text[index + len(marker):])
    return (target or text, destination or None)


class _Session:
    def __init__(self) -> None:
        self.phase = "find_target"
        self.phase_step = 0
        self.target = None
        self.destination = None
        self.target_phrase = ""
        self.destination_phrase = None
        self.controller = None
        self.last_look = -(10 ** 9)
        self.grip_z = None


class CodeAsPolicySeed:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @staticmethod
    def _camera(observation):
        for name in sorted(observation.cameras):
            camera = observation.cameras[name]
            if has_3d(camera):
                return name, camera
        name = sorted(observation.cameras)[0]
        return name, observation.cameras[name]

    def _locate(self, phrase, camera, tools, session, label):
        if not phrase or not has_3d(camera):
            return None
        pixel = None
        if tools.has("pointing"):
            try:
                found = tools.point(PointingRequest(camera.rgb, phrase))
            except ToolUnavailableError as error:
                tools.record("perception", "optional_error", f"pointer failed: {error}", "pointing")
                found = None
            if found is not None and found.points_xy:
                height, width = camera.rgb.shape[:2]
                inside = [
                    index
                    for index, (x, y) in enumerate(found.points_xy)
                    if 0.0 <= x < float(width) and 0.0 <= y < float(height)
                ]
                if inside:
                    pixel = found.points_xy[max(inside, key=found.confidence.__getitem__)]
        if pixel is None and tools.has("detection"):
            try:
                boxes = tools.detect(DetectionRequest(camera.rgb, phrase))
            except ToolUnavailableError as error:
                tools.record("perception", "optional_error", f"detector failed: {error}", "detection")
                boxes = None
            if boxes is not None and boxes.detections:
                best = max(boxes.detections, key=lambda item: item.score)
                x0, y0, x1, y1 = best.box_xyxy
                pixel = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                if tools.has("segmentation"):
                    try:
                        masks = tools.segment(SegmentationRequest(camera.rgb, boxes_xyxy=(best.box_xyxy,)))
                        if masks.masks.shape[0]:
                            mask = masks.masks[int(masks.scores.argmax())]
                            rows, columns = np.nonzero(mask)
                            if columns.size:
                                pixel = (float(columns.mean()), float(rows.mean()))
                    except ToolUnavailableError as error:
                        tools.record("perception", "optional_error", f"segmenter failed: {error}", "segmentation")
        if pixel is None:
            tools.record("perception", "ok", f"nothing found for '{phrase}' ({label})", "pointing")
            return None
        point = pixel_to_world(camera, pixel[0], pixel[1])
        if point is None:
            tools.record("perception", "ok", f"'{phrase}' has no usable depth at that pixel ({label})", "pointing")
            return None
        value = np.asarray(point, dtype=np.float64)
        tools.record(
            "perception",
            "ok",
            f"{label} '{phrase}' at pixel ({pixel[0]:.1f},{pixel[1]:.1f}) "
            f"-> ({value[0]:.4f},{value[1]:.4f},{value[2]:.4f})",
            "pointing",
        )
        return value

    def _chunk(self, request, values, session):
        return CanonicalActionChunk(
            request_id=request.request_id,
            session_id=request.session_id,
            start_step=request.observation.step_index,
            spec=request.action_spec,
            values=np.asarray(values, dtype=np.float32).reshape(1, -1),
            execution_count=1,
        )

    def _hold(self, request, session, tools, reason):
        controller = session.controller
        if controller is None:
            values = np.zeros(len(request.action_spec.channel_names), dtype=np.float32)
            tools.record("decision", "skipped", reason)
            return self._chunk(request, values, session)
        try:
            values = controller.hold(request.observation)
        except ValueError as error:
            tools.record("decision", "skipped", f"{reason}; holding still is not possible either: {error}")
            values = np.zeros(len(request.action_spec.channel_names), dtype=np.float32)
        return self._chunk(request, values, session)

    def _advance(self, session, phase):
        session.phase = phase
        session.phase_step = 0

    def act(self, request: AgentRequest, tools: ToolboxProtocol) -> CanonicalActionChunk:
        observation = request.observation
        session = self._sessions.setdefault(request.session_id, _Session())
        step = observation.step_index
        if session.controller is None:
            session.controller = make_controller(request.action_spec)
        if session.controller is None:
            return self._hold(request, session, tools, "this setup has no movement commands in agent.motion")
        if not session.target_phrase:
            target_phrase, destination_phrase = _phrases(observation.instruction)
            session.target_phrase = target_phrase
            session.destination_phrase = destination_phrase
            tools.record(
                "decision",
                "ok",
                f"target '{target_phrase}' destination '{destination_phrase}'",
            )
        camera_name, camera = self._camera(observation)
        if not has_3d(camera):
            return self._hold(request, session, tools, "this setup reports no depth, so no target can be reached")

        session.phase_step += 1
        here = session.controller.position(observation)
        if here is None:
            return self._hold(request, session, tools, "this setup reports no gripper position")
        try:
            return self._plan_step(request, session, tools, camera, here, step)
        except ValueError as error:
            return self._hold(request, session, tools, f"a movement command was refused: {error}")

    def _plan_step(self, request, session, tools, camera, here, step):
        observation = request.observation

        if session.phase == "find_target":
            if step - session.last_look >= LOOK_EVERY or session.target is None:
                session.last_look = step
                session.target = self._locate(session.target_phrase, camera, tools, session, "target")
            if session.target is None:
                return self._hold(request, session, tools, "target not located yet")
            self._advance(session, "above_target")

        if session.phase == "above_target":
            goal = np.array([session.target[0], session.target[1], session.target[2] + APPROACH_HEIGHT_M])
            if session.controller.distance_to(observation, goal) < ARRIVED_M or session.phase_step > PHASE_LIMIT:
                self._advance(session, "descend")
            else:
                values = session.controller.move_to(observation, goal, closed=False, max_step_m=STEP_M)
                return self._chunk(request, values, session)

        if session.phase == "descend":
            goal = np.array([session.target[0], session.target[1], session.target[2] - GRASP_DROP_M])
            if session.controller.distance_to(observation, goal) < ARRIVED_M or session.phase_step > PHASE_LIMIT:
                session.grip_z = float(here[2])
                self._advance(session, "close")
            else:
                values = session.controller.move_to(observation, goal, closed=False, max_step_m=STEP_M)
                return self._chunk(request, values, session)

        if session.phase == "close":
            if session.phase_step > CLOSE_STEPS:
                self._advance(session, "lift")
            else:
                values = session.controller.set_gripper(observation, closed=True)
                return self._chunk(request, values, session)

        if session.phase == "lift":
            goal = np.array([here[0], here[1], (session.grip_z or here[2]) + LIFT_M])
            if session.phase_step > PHASE_LIMIT or here[2] >= goal[2] - ARRIVED_M:
                self._advance(session, "find_destination")
            else:
                values = session.controller.nudge(observation, (0.0, 0.0, STEP_M), closed=True, max_step_m=STEP_M)
                return self._chunk(request, values, session)

        if session.phase == "find_destination":
            if session.destination_phrase is None:
                self._advance(session, "done")
            else:
                if session.destination is None:
                    session.destination = self._locate(
                        session.destination_phrase, camera, tools, session, "destination"
                    )
                if session.destination is None and session.phase_step > LOOK_EVERY:
                    self._advance(session, "done")
                elif session.destination is None:
                    values = session.controller.hold(observation, closed=True)
                    return self._chunk(request, values, session)
                else:
                    self._advance(session, "above_destination")

        if session.phase == "above_destination":
            goal = np.array(
                [session.destination[0], session.destination[1], session.destination[2] + APPROACH_HEIGHT_M]
            )
            if session.controller.distance_to(observation, goal) < ARRIVED_M or session.phase_step > PHASE_LIMIT:
                self._advance(session, "lower")
            else:
                values = session.controller.move_to(observation, goal, closed=True, max_step_m=STEP_M)
                return self._chunk(request, values, session)

        if session.phase == "lower":
            goal = np.array(
                [session.destination[0], session.destination[1], session.destination[2] + RELEASE_DROP_M]
            )
            if session.controller.distance_to(observation, goal) < ARRIVED_M or session.phase_step > PHASE_LIMIT:
                self._advance(session, "release")
            else:
                values = session.controller.move_to(observation, goal, closed=True, max_step_m=STEP_M)
                return self._chunk(request, values, session)

        if session.phase == "release":
            if session.phase_step > OPEN_STEPS:
                self._advance(session, "done")
            else:
                values = session.controller.set_gripper(observation, closed=False)
                return self._chunk(request, values, session)

        return self._hold(request, session, tools, "the planned sequence is finished")


def create_scaffold() -> CodeAsPolicySeed:
    return CodeAsPolicySeed()
