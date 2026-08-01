from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np

from robot_auto_evolve.agent import (
    AgentRequest,
    DetectionRequest,
    GraspRequest,
    LanguageRequest,
    PointingRequest,
    SegmentationRequest,
    ToolboxProtocol,
    ToolUnavailableError,
    VLARequest,
    VisionRequest,
)
from robot_auto_evolve.protocol import CanonicalActionChunk


SCAFFOLD_CONFIG = {
    "required_capabilities": ("vla",),
    "optional_capabilities": ("language", "vision", "detection", "segmentation", "pointing", "grasp"),
}


@dataclass
class _Plan:
    instruction: str
    subgoal: str
    context: tuple[str, ...]
    planned_step: int


@dataclass
class _Monitor:
    image: np.ndarray | None = None
    joint_position: np.ndarray | None = None
    step_index: int = -1
    stagnant_steps: int = 0
    last_vision_check: int = -1


T = TypeVar("T")


class SeedScaffold:
    def __init__(self) -> None:
        self._plans: dict[str, _Plan] = {}
        self._monitors: dict[str, _Monitor] = {}

    def reset(self, session_id: str) -> None:
        self._plans.pop(session_id, None)
        self._monitors.pop(session_id, None)

    def _optional(
        self,
        capability: str,
        tools: ToolboxProtocol,
        call: Callable[[], T],
    ) -> T | None:
        if not tools.has(capability):
            if tools.required(capability):
                raise ToolUnavailableError(f"required {capability} is unavailable")
            tools.record("decision", "skipped", "optional capability unavailable", capability)
            return None
        try:
            return call()
        except ToolUnavailableError:
            if tools.required(capability):
                raise
            tools.record("decision", "optional_error", "continuing without optional capability", capability)
            return None

    @staticmethod
    def _clip_box(
        box: tuple[float, float, float, float],
        image: np.ndarray,
    ) -> tuple[float, float, float, float] | None:
        height, width = image.shape[:2]
        x0, y0, x1, y1 = box
        clipped = (
            min(float(width), max(0.0, x0)),
            min(float(height), max(0.0, y0)),
            min(float(width), max(0.0, x1)),
            min(float(height), max(0.0, y1)),
        )
        return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None

    def _observe_motion(self, request: AgentRequest) -> bool:
        observation = request.observation
        camera = next(iter(observation.cameras.values()))
        image = camera.rgb[::16, ::16].copy()
        robot_state = np.concatenate([vector.values for vector in observation.proprioception.vectors])
        monitor = self._monitors.setdefault(request.session_id, _Monitor())
        if monitor.step_index >= observation.step_index:
            monitor.image = None
            monitor.joint_position = None
            monitor.stagnant_steps = 0
        stagnant = False
        if monitor.image is not None and monitor.joint_position is not None:
            image_delta = float(np.abs(image.astype(np.int16) - monitor.image.astype(np.int16)).mean()) / 255.0
            joint_delta = float(np.linalg.norm(robot_state - monitor.joint_position))
            stagnant = image_delta < 0.002 and joint_delta < 0.002
        monitor.stagnant_steps = monitor.stagnant_steps + 1 if stagnant else 0
        monitor.image = image
        monitor.joint_position = robot_state
        monitor.step_index = observation.step_index
        return monitor.stagnant_steps >= 6

    def _vision_replan_reason(self, request: AgentRequest, tools: ToolboxProtocol) -> str | None:
        monitor = self._monitors[request.session_id]
        step = request.observation.step_index
        plan = self._plans.get(request.session_id)
        if plan is None or step < 8:
            return None
        if monitor.last_vision_check >= 0 and step - monitor.last_vision_check < 8:
            return None
        monitor.last_vision_check = step
        result = self._optional(
            "vision",
            tools,
            lambda: tools.vision(
                VisionRequest(
                    instruction=(
                        "Use first line COMPLETE or CONTINUE, then one short reason. Judge whether the active "
                        "subgoal is visibly complete. Object-location phrases may identify the starting object; "
                        f"do not treat them as desired final relations. Task: {request.observation.instruction}"
                    ),
                    images={name: camera.rgb for name, camera in request.observation.cameras.items()},
                    context=(f"active subgoal: {plan.subgoal}",),
                    max_tokens=64,
                )
            ),
        )
        if result is None:
            return None
        text = result.text.strip()
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        match = re.match(r"^[^A-Za-z0-9]*(COMPLETE|CONTINUE)\b", first_line, flags=re.IGNORECASE)
        label = match.group(1).upper() if match else ""
        if label not in {"COMPLETE", "CONTINUE"}:
            tools.record("monitor", "skipped", f"unparsed progress response: {first_line[:80]}", "vision")
            return None
        tools.record("monitor", "ok", text[:300], "vision")
        if label == "COMPLETE":
            return "vision monitor reported active subgoal complete"
        return None

    def _recovery_reason(self, request: AgentRequest, tools: ToolboxProtocol) -> str | None:
        stagnant = self._observe_motion(request)
        vision_reason = self._vision_replan_reason(request, tools)
        if stagnant:
            return "fair image and joint observations were stagnant for six checks"
        return vision_reason

    def _needs_plan(self, request: AgentRequest) -> bool:
        plan = self._plans.get(request.session_id)
        observation = request.observation
        return (
            plan is None
            or plan.instruction != observation.instruction
            or observation.step_index < plan.planned_step
            or observation.step_index - plan.planned_step >= 32
        )

    def _plan(self, request: AgentRequest, tools: ToolboxProtocol) -> _Plan:
        observation = request.observation
        camera_name, camera = next(iter(observation.cameras.items()))
        context: list[str] = []
        scene = self._optional(
            "vision",
            tools,
            lambda: tools.vision(
                VisionRequest(
                    instruction="Describe task-relevant objects, free space, and robot-object relations.",
                    images={name: value.rgb for name, value in observation.cameras.items()},
                    context=(observation.instruction,),
                )
            ),
        )
        if scene is not None:
            context.append(f"scene: {scene.text[:1200]}")
            tools.record("perception", "ok", scene.text[:500], "vision")
        detection = self._optional(
            "detection",
            tools,
            lambda: tools.detect(DetectionRequest(camera.rgb, observation.instruction)),
        )
        best_box: tuple[float, float, float, float] | None = None
        if detection is not None and detection.detections:
            best = max(detection.detections, key=lambda item: item.score)
            best_box = self._clip_box(best.box_xyxy, camera.rgb)
            detail = f"detected {best.label} at xyxy={best.box_xyxy} score={best.score:.3f}"
            if best_box != best.box_xyxy:
                detail += f" clipped_xyxy={best_box}"
            context.append(detail)
            tools.record(
                "perception",
                "ok",
                detail,
                "detection",
            )
        elif detection is not None:
            tools.record("perception", "ok", "no detection above threshold", "detection")
        points = self._optional(
            "pointing",
            tools,
            lambda: tools.point(PointingRequest(camera.rgb, observation.instruction)),
        )
        best_point: tuple[float, float] | None = None
        if points is not None and points.points_xy:
            height, width = camera.rgb.shape[:2]
            valid = [
                index
                for index, (x, y) in enumerate(points.points_xy)
                if 0.0 <= x < float(width) and 0.0 <= y < float(height)
            ]
            if valid:
                index = max(valid, key=points.confidence.__getitem__)
                best_point = points.points_xy[index]
                context.append(f"target point xy={best_point} confidence={points.confidence[index]:.3f}")
            else:
                tools.record("perception", "skipped", "pointing returned no in-bounds target point", "pointing")
        mask = None
        segmentation_request = None
        if best_box is not None:
            segmentation_request = SegmentationRequest(camera.rgb, boxes_xyxy=(best_box,))
        elif best_point is not None:
            segmentation_request = SegmentationRequest(camera.rgb, points_xy=(best_point,), labels=(1,))
        if segmentation_request is not None:
            segmented = self._optional(
                "segmentation",
                tools,
                lambda: tools.segment(segmentation_request),
            )
            if segmented is not None and segmented.masks.shape[0]:
                index = int(segmented.scores.argmax())
                mask = segmented.masks[index]
                y, x = np.nonzero(mask)
                if x.size:
                    summary = (
                        f"target mask bbox_xyxy=({int(x.min())},{int(y.min())},{int(x.max()) + 1},{int(y.max()) + 1}) "
                        f"centroid_xy=({float(x.mean()):.1f},{float(y.mean()):.1f}) "
                        f"coverage={float(mask.mean()):.4f} score={float(segmented.scores[index]):.3f}"
                    )
                    context.append(summary)
                    tools.record("perception", "ok", summary, "segmentation")
                else:
                    tools.record("perception", "ok", "best segmentation mask is empty", "segmentation")
        else:
            tools.record(
                "perception",
                "skipped",
                "segmentation needs a detector box or in-bounds target point",
                "segmentation",
            )
        calibrated = camera.depth_m is not None and camera.intrinsics is not None and camera.camera_to_world is not None
        if calibrated:
            grasps = self._optional(
                "grasp",
                tools,
                lambda: tools.grasp(
                    GraspRequest(
                        rgb=camera.rgb,
                        depth_m=camera.depth_m,
                        intrinsics=camera.intrinsics,
                        camera_to_world=camera.camera_to_world,
                        optical_convention=camera.optical_convention,
                        mask=mask,
                    )
                ),
            )
            if grasps is not None and grasps.candidates:
                best_grasp = max(grasps.candidates, key=lambda item: item.score)
                xyz = best_grasp.pose_world[:3, 3]
                context.append(
                    f"grasp xyz=({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}) score={best_grasp.score:.3f}"
                )
        else:
            tools.record("decision", "skipped", "grasp needs calibrated depth", "grasp")
        context.append(f"primary_camera={camera_name}")
        subgoal = observation.instruction
        planned = self._optional(
            "language",
            tools,
            lambda: tools.language(
                LanguageRequest(
                    instruction=(
                        "Return one short robot-executable subgoal grounded in the supplied observations. "
                        f"Task: {observation.instruction}."
                    ),
                    context=tuple(context),
                    max_tokens=96,
                )
            ),
        )
        if planned is not None and planned.text.strip():
            subgoal = planned.text.strip()[:500]
        tools.record("decision", "ok", f"selected subgoal: {subgoal[:300]}")
        return _Plan(observation.instruction, subgoal, tuple(context), observation.step_index)

    def act(self, request: AgentRequest, tools: ToolboxProtocol) -> CanonicalActionChunk:
        recovery = self._recovery_reason(request, tools)
        interrupted = recovery is not None
        if interrupted:
            self._plans.pop(request.session_id, None)
            tools.record("recovery", "triggered", recovery)
        if self._needs_plan(request):
            self._plans[request.session_id] = self._plan(request, tools)
            interrupted = True
        plan = self._plans[request.session_id]
        instruction = plan.subgoal or request.observation.instruction
        action = tools.vla(
            VLARequest(
                request_id=request.request_id,
                session_id=request.session_id,
                observation=request.observation,
                instruction=instruction,
                context=plan.context,
                refresh=interrupted,
            )
        )
        if action.horizon != 1 or action.execution_count != 1:
            raise RuntimeError("native policy service must return exactly one executable action per agent step")
        if interrupted:
            tools.record("execution", "triggered", "refreshed native policy cache before this action")
        return action


def create_scaffold() -> SeedScaffold:
    return SeedScaffold()
