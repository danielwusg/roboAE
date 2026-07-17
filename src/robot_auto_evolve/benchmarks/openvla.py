from __future__ import annotations

from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionSpec, StrictSchemaError

from .contracts import action_spec, policy_actions
from .transforms import euler_xyz_to_matrix, matrix_to_axis_angle


OPENVLA_GOOGLE_TASKS = frozenset(
    {
        "google_robot_pick_coke_can",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_apple_in_closed_top_drawer",
    }
)

OPENVLA_GOOGLE_ACTION_SPEC: CanonicalActionSpec = action_spec(
    arms=("arm",),
    channels=("x", "y", "z", "rx", "ry", "rz", "gripper"),
    semantics=("delta",) * 7,
    frame="robot_base",
    rotation="axis_angle",
    gripper="closed_positive",
    period_s=1.0 / 3.0,
)


def decode_openvla_google_action(value: Any, gripper_action: float) -> np.ndarray:
    raw = policy_actions(value, 7)
    if raw.shape != (1, 7):
        raise StrictSchemaError("OpenVLA base must return exactly one 7D action")
    result = np.empty(7, dtype=np.float32)
    result[:3] = raw[0, :3]
    result[3:6] = matrix_to_axis_angle(euler_xyz_to_matrix(raw[0, 3:6])).astype(np.float32)
    result[6] = np.float32(gripper_action)
    if not np.isfinite(result).all():
        raise StrictSchemaError("OpenVLA converted action is non-finite")
    return np.ascontiguousarray(result)


class OpenVLAGoogleAdapter:
    action_spec = OPENVLA_GOOGLE_ACTION_SPEC

    @staticmethod
    def decode(value: Any, gripper_action: float) -> np.ndarray:
        return decode_openvla_google_action(value, gripper_action)
