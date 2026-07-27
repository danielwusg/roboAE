"""Movement commands the scaffold may return instead of, or alongside, the policy's action.

WHAT IT IS
    Arithmetic on numbers the scaffold already has: where the gripper is now (from
    `observation.proprioception`) and where you want it (a point you computed, normally from
    perception plus `robot_auto_evolve.agent.geometry`). It returns the action values for ONE step
    in the route's own action space. No inverse-kinematics library and no motion planner: for a
    delta-controlled arm a move is the clipped difference between here and there, repeated each
    step; for an absolute-pose arm it is the target pose itself.

    It reads nothing privileged. It cannot see the simulator, object poses, or the success check.
    A target must come from this episode's observation. A spatial target written into the source
    as a numeric literal is rejected as a fairness violation.

FRAMES
    Targets are in the SAME frame the benchmark reports the end effector in -- the
    `reference_frame` of the `eef_pose` entry in `observation.proprioception`, which is also the
    frame `geometry.pixel_to_world` returns points in. So this works:

        point = pixel_to_world(camera, u, v)
        values = controller.move_to(observation, point)

USING IT
    from robot_auto_evolve.agent.motion import make_controller
    from robot_auto_evolve.protocol import CanonicalActionChunk

    chunk = tools.vla(VLARequest(...))
    controller = make_controller(chunk.spec)        # None if this route is unsupported
    controller.note(chunk)                          # remember its rotation + gripper
    values = controller.move_to(request.observation, point, max_step_m=0.03)
    return CanonicalActionChunk(
        request_id=request.request_id,
        session_id=request.session_id,
        start_step=request.observation.step_index,
        spec=chunk.spec,
        values=values[None, :],                     # one step -> shape [1, n_channels]
        execution_count=1,
    )

    `make_controller` also accepts the spec on its own, so a scaffold that never calls the policy
    can still build one. `supported_layouts()` lists what it can drive.

THE COMMANDS
    move_to(observation, target_xyz, ...)   step the gripper toward a point
    nudge(observation, delta_xyz, ...)      step by an offset from where it is now
    hold(observation, ...)                  stay put
    set_gripper(observation, closed=...)    stay put and open/close the gripper
    Each returns a float32 array of action values for ONE step in the route's channel order.
    None of them apply the action; the scaffold returns it as usual.

ASK THE POLICY EVERY STEP
    Every policy service tracks which step it last produced an action for and rejects a request
    that skips ahead ("policy_act: previous action is not observed as executed"). Some services
    are stricter still and require observations contiguous from step zero. So there are exactly two
    safe shapes -- call `tools.vla(...)` every step and sometimes return a movement command instead
    of its action, or never call it at all for the whole episode. Mixing them fails every episode.
    Discarding an action you asked for costs only the inference; the policy's internal state then
    advances as it otherwise would.

ROTATION
    On a delta-controlled arm the rotation channels are set to zero, which means "do not turn".
    On an absolute-pose arm the action IS the target pose, so the rotation channels are copied
    from the last action passed to `note()`, or derived from the gripper's current orientation if
    there is none. Passing the policy's own chunk to `note()` is the reliable route.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionSpec


__all__ = ["MotionController", "end_effector_position", "make_controller", "supported_layouts"]


# The three rotation conversions this module needs, written out locally: importing
# robot_auto_evolve.benchmarks.transforms would execute that package's __init__, which pulls in
# the simulator adapters and PyYAML, neither of which the scaffold's Python environment has.


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("cannot normalize a near-zero quaternion")
    x, y, z, w = q / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    y = float(np.arcsin(np.clip(-m[2, 0], -1.0, 1.0)))
    if abs(np.cos(y)) > 1e-7:
        x = float(np.arctan2(m[2, 1], m[2, 2]))
        z = float(np.arctan2(m[1, 0], m[0, 0]))
    else:
        x = 0.0
        z = float(np.arctan2(-m[0, 1], m[1, 1]))
    return np.array((x, y, z), dtype=np.float64)


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    candidates = np.array(
        (1 + m[0, 0] - m[1, 1] - m[2, 2], 1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2], 1 + trace)
    )
    index = int(np.argmax(candidates))
    q = np.zeros(4, dtype=np.float64)
    q[index] = 0.5 * np.sqrt(max(float(candidates[index]), 0.0))
    scale = 0.25 / max(float(q[index]), 1e-12)
    if index == 0:
        q[1:] = ((m[0, 1] + m[1, 0]) * scale, (m[0, 2] + m[2, 0]) * scale, (m[2, 1] - m[1, 2]) * scale)
    elif index == 1:
        q[0], q[2], q[3] = (m[0, 1] + m[1, 0]) * scale, (m[1, 2] + m[2, 1]) * scale, (m[0, 2] - m[2, 0]) * scale
    elif index == 2:
        q[0], q[1], q[3] = (m[0, 2] + m[2, 0]) * scale, (m[1, 2] + m[2, 1]) * scale, (m[1, 0] - m[0, 1]) * scale
    else:
        q[:3] = ((m[2, 1] - m[1, 2]) * scale, (m[0, 2] - m[2, 0]) * scale, (m[1, 0] - m[0, 1]) * scale)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    xyz = q[:3]
    w = float(np.clip(q[3], -1.0, 1.0))
    length = float(np.linalg.norm(xyz))
    angle = 2 * float(np.arctan2(length, w))
    factor = angle / length if length > 1e-8 else 2.0
    return np.asarray(xyz * factor, dtype=np.float64)


# One layout per action space this module can drive, classified from the route's own
# CanonicalActionSpec.
DELTA_EE7 = "delta_ee7"
ABSOLUTE_EE7 = "absolute_ee7"
ABSOLUTE_EE8_TWO_FINGERS = "absolute_ee8_two_fingers"
DELTA_ARM_WITH_BASE12 = "delta_arm_with_base12"

# Default cap on how far one step may command the gripper to travel, in metres. A short step
# repeated each step is a smooth move; a long step is a lunge the low-level controller may not
# track.
DEFAULT_MAX_STEP_M = 0.03


def supported_layouts() -> tuple[str, ...]:
    """The action layouts `make_controller` can drive."""
    return (DELTA_EE7, ABSOLUTE_EE7, ABSOLUTE_EE8_TWO_FINGERS, DELTA_ARM_WITH_BASE12)


def end_effector_position(observation: Any) -> np.ndarray | None:
    """Where the gripper is now, as (x, y, z), or None if the route does not report it.

    Looks up the first `end_effector_pose` entry in `observation.proprioception` whose first three
    components are named x, y, z. Routes name that entry differently, so it is found by meaning
    rather than by name.
    """
    proprioception = getattr(observation, "proprioception", None)
    if proprioception is None:
        return None
    for vector in proprioception.vectors:
        spec = vector.spec
        if spec.quantity != "end_effector_pose":
            continue
        if tuple(spec.component_names[:3]) != ("x", "y", "z"):
            continue
        return np.asarray(vector.values[:3], dtype=np.float64)
    return None


def end_effector_frame(observation: Any) -> str | None:
    """The name of the frame `end_effector_position` is reported in (e.g. "world")."""
    proprioception = getattr(observation, "proprioception", None)
    if proprioception is None:
        return None
    for vector in proprioception.vectors:
        spec = vector.spec
        if spec.quantity == "end_effector_pose" and tuple(spec.component_names[:3]) == ("x", "y", "z"):
            return spec.reference_frame
    return None


def _end_effector_rotation_matrix(observation: Any) -> np.ndarray | None:
    """The gripper's current orientation as a 3x3 matrix, from whatever the route reports."""
    proprioception = getattr(observation, "proprioception", None)
    if proprioception is None:
        return None
    for vector in proprioception.vectors:
        spec = vector.spec
        if spec.quantity != "end_effector_pose":
            continue
        values = np.asarray(vector.values, dtype=np.float64)
        if spec.representation == "xyz_quaternion" and values.shape == (7,):
            quaternion = values[3:] if spec.quaternion_order == "xyzw" else values[[4, 5, 6, 3]]
            return _quaternion_xyzw_to_matrix(quaternion)
        if spec.representation == "quaternion" and values.shape == (4,):
            quaternion = values if spec.quaternion_order == "xyzw" else values[[1, 2, 3, 0]]
            return _quaternion_xyzw_to_matrix(quaternion)
    return None


def _classify(spec: CanonicalActionSpec) -> str | None:
    channels = tuple(spec.channel_names)
    semantics = tuple(spec.channel_semantics)
    if len(channels) == 7 and semantics[:6] == ("delta",) * 6:
        return DELTA_EE7
    if len(channels) == 7 and semantics[:6] == ("absolute",) * 6:
        return ABSOLUTE_EE7
    if (
        len(channels) == 8
        and semantics == ("absolute",) * 8
        and channels[6:] == ("left_finger", "right_finger")
    ):
        return ABSOLUTE_EE8_TWO_FINGERS
    if len(channels) == 12 and channels[:7] == (
        "right_arm_base_dx",
        "right_arm_base_dy",
        "right_arm_base_dz",
        "right_arm_base_drx",
        "right_arm_base_dry",
        "right_arm_base_drz",
        "right_gripper_close",
    ):
        return DELTA_ARM_WITH_BASE12
    return None


class MotionController:
    """Movement commands for one route's action space. Build it with `make_controller`."""

    def __init__(self, spec: CanonicalActionSpec, layout: str) -> None:
        self.spec = spec
        self.layout = layout
        self.channel_names = tuple(spec.channel_names)
        self.n_channels = len(self.channel_names)
        self.is_delta = layout in (DELTA_EE7, DELTA_ARM_WITH_BASE12)
        self.gripper_index: int | None = None
        if layout in (DELTA_EE7, ABSOLUTE_EE7):
            self.gripper_index = 6
        elif layout == DELTA_ARM_WITH_BASE12:
            self.gripper_index = 6
        # Metres of real travel per unit of the translation channels: the route's own
        # controller_output_scale when the values are normalized, one metre per unit when they
        # are physical.
        if spec.value_encoding == "normalized_controller" and spec.controller_output_scale:
            self.metres_per_unit = float(spec.controller_output_scale[0])
        else:
            self.metres_per_unit = 1.0
        # Absolute layouts command a pose, so "one unit" is already a metre of position.
        if not self.is_delta:
            self.metres_per_unit = 1.0
        self._last: np.ndarray | None = None
        self._gripper = -1.0 if spec.gripper_convention in ("closed_positive", "binary_closed_one") else 1.0
        self._fingers = np.array((0.04, 0.04), dtype=np.float64)

    # -- state ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the last action. Call this from the scaffold's own reset(session_id)."""
        self._last = None
        self._gripper = -1.0 if self.spec.gripper_convention in ("closed_positive", "binary_closed_one") else 1.0
        self._fingers = np.array((0.04, 0.04), dtype=np.float64)

    def note(self, action: Any) -> None:
        """Remember an action (a CanonicalActionChunk or a raw value array).

        Pass the policy's own chunk here on every step you use it. The controller keeps the
        rotation channels, the gripper command and, on a 12-channel layout, the base/torso/mode
        channels, so a later `move_to` changes only where the gripper goes.
        """
        values = getattr(action, "values", action)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2:
            count = int(getattr(action, "execution_count", array.shape[0]) or array.shape[0])
            array = array[min(count, array.shape[0]) - 1]
        if array.shape != (self.n_channels,):
            raise ValueError(f"action has {array.shape} values, expected ({self.n_channels},)")
        self._last = array.copy()
        if self.gripper_index is not None:
            self._gripper = float(array[self.gripper_index])
        if self.layout == ABSOLUTE_EE8_TWO_FINGERS:
            self._fingers = array[6:8].copy()

    @property
    def last_action(self) -> np.ndarray | None:
        """The last action passed to `note`, or None."""
        return None if self._last is None else self._last.copy()

    # -- helpers -------------------------------------------------------------------------

    def position(self, observation: Any) -> np.ndarray | None:
        """Where the gripper is now, in the frame targets are given in."""
        return end_effector_position(observation)

    def distance_to(self, observation: Any, target_xyz: Any) -> float | None:
        """Straight-line metres from the gripper to a target, or None if unknown."""
        here = end_effector_position(observation)
        if here is None:
            return None
        return float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64).reshape(3) - here))

    def _gripper_value(self, closed: bool | None) -> float:
        if closed is None:
            return float(self._gripper)
        if self.spec.gripper_convention in ("closed_positive", "binary_closed_one"):
            return 1.0 if closed else -1.0
        return -1.0 if closed else 1.0

    def _base(self, reference: Any | None) -> np.ndarray:
        """The action to start from before overwriting the translation channels."""
        if reference is not None:
            values = getattr(reference, "values", reference)
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 2:
                array = array[0]
            if array.shape != (self.n_channels,):
                raise ValueError(f"reference action has {array.shape} values, expected ({self.n_channels},)")
            return array.copy()
        if self._last is not None:
            return self._last.copy()
        return np.zeros(self.n_channels, dtype=np.float64)

    def _rotation_channels(self, observation: Any, base: np.ndarray) -> np.ndarray:
        if self.is_delta:
            # A delta layout expresses "do not turn" exactly: zero.
            return np.zeros(3, dtype=np.float64)
        if self._last is not None or np.any(base[3:6]):
            return base[3:6].copy()
        matrix = _end_effector_rotation_matrix(observation)
        if matrix is None:
            raise ValueError(
                "this route commands an absolute pose, so a rotation is required: call note() "
                "with the policy's chunk first, or pass reference="
            )
        if self.spec.rotation_representation == "euler_xyz":
            return _matrix_to_euler_xyz(matrix)
        return _matrix_to_axis_angle(matrix)

    def _finish(self, values: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float32)
        if self.spec.value_encoding == "normalized_controller":
            result = np.clip(result, -1.0, 1.0).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("motion command produced non-finite values")
        return result

    # -- commands ------------------------------------------------------------------------

    def hold(self, observation: Any, *, closed: bool | None = None, reference: Any | None = None) -> np.ndarray:
        """Stay where you are for one step. Useful while you look at something."""
        return self.nudge(observation, (0.0, 0.0, 0.0), closed=closed, reference=reference, max_step_m=None)

    def set_gripper(self, observation: Any, *, closed: bool, reference: Any | None = None) -> np.ndarray:
        """Stay put and open (closed=False) or close (closed=True) the gripper."""
        return self.nudge(observation, (0.0, 0.0, 0.0), closed=closed, reference=reference, max_step_m=None)

    def nudge(
        self,
        observation: Any,
        delta_xyz: Any,
        *,
        closed: bool | None = None,
        reference: Any | None = None,
        max_step_m: float | None = DEFAULT_MAX_STEP_M,
    ) -> np.ndarray:
        """Move by an offset from where the gripper is now, for one step."""
        offset = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        if max_step_m is not None and max_step_m > 0.0:
            length = float(np.linalg.norm(offset))
            if length > max_step_m:
                offset = offset * (max_step_m / length)
        base = self._base(reference)
        values = base.copy()
        values[3:6] = self._rotation_channels(observation, base)
        if self.is_delta:
            values[:3] = offset / max(self.metres_per_unit, 1e-9)
        else:
            here = end_effector_position(observation)
            if here is None:
                raise ValueError("this route does not report an end-effector position, so nudge cannot work")
            values[:3] = here + offset
        if self.gripper_index is not None:
            values[self.gripper_index] = self._gripper_value(closed)
        if self.layout == ABSOLUTE_EE8_TWO_FINGERS:
            if closed is None:
                fingers = self._fingers
            else:
                fingers = np.array((0.0, 0.0)) if closed else np.array((0.04, 0.04))
            values[6:8] = fingers
        result = self._finish(values)
        self.note(result)
        return result

    def move_to(
        self,
        observation: Any,
        target_xyz: Any,
        *,
        closed: bool | None = None,
        reference: Any | None = None,
        max_step_m: float | None = DEFAULT_MAX_STEP_M,
        offset_xyz: Any | None = None,
    ) -> np.ndarray:
        """Step the gripper toward `target_xyz` (in the end-effector's own frame).

        Call it again each step and the gripper walks to the point. `max_step_m` caps how far
        one step commands; None removes the cap (on an absolute-pose route that means jumping
        straight to the target, which the low-level controller may or may not track).
        `offset_xyz` is added to the target -- handy for approaching from above before closing.
        """
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        if offset_xyz is not None:
            target = target + np.asarray(offset_xyz, dtype=np.float64).reshape(3)
        here = end_effector_position(observation)
        if here is None:
            raise ValueError("this route does not report an end-effector position, so move_to cannot work")
        return self.nudge(
            observation,
            target - here,
            closed=closed,
            reference=reference,
            max_step_m=max_step_m,
        )


def make_controller(spec: Any) -> MotionController | None:
    """Build a controller for a route's action space, or None if the layout is not supported.

    `spec` is a CanonicalActionSpec -- take it from any policy chunk's `.spec`. Returns None for
    layouts this module cannot drive, for example a two-armed robot. Always check for None.
    """
    if not isinstance(spec, CanonicalActionSpec):
        spec = getattr(spec, "spec", None)
    if not isinstance(spec, CanonicalActionSpec):
        return None
    layout = _classify(spec)
    if layout is None:
        return None
    return MotionController(spec, layout)
