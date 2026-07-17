from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation

from .contracts import AdapterError, action_chunk, action_spec, camera, observation, policy_actions, state
MODEL_HORIZON = 50
EXECUTION_HORIZON = 10

PI05_LIBERO_ACTION_SPEC = action_spec(
    arms=("arm",),
    channels=("dx", "dy", "dz", "drx", "dry", "drz", "gripper"),
    semantics=("delta", "delta", "delta", "delta", "delta", "delta", "binary"),
    frame="world",
    rotation="axis_angle",
    gripper="closed_positive",
    period_s=1 / 20,
    value_encoding="normalized_controller",
    controller_output_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0),
)


def quaternion_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise AdapterError("quaternion must be finite xyzw with width 4")
    w = float(np.clip(value[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return np.ascontiguousarray(value[:3] * (2.0 * np.arccos(w) / denominator), dtype=np.float32)


class Pi05LiberoAdapter:
    action_spec: CanonicalActionSpec = PI05_LIBERO_ACTION_SPEC
    execution_count = EXECUTION_HORIZON

    def __init__(self) -> None:
        self._episode_id: str | None = None

    def reset(self, value: FairObservation | Mapping[str, Any]) -> None:
        self._episode_id = observation(value).episode_id

    def encode(self, value: FairObservation | Mapping[str, Any]) -> dict[str, Any]:
        obs = observation(value)
        if obs.episode_id != self._episode_id:
            raise AdapterError("adapter.reset must run once at episode start")
        pose = state(
            obs,
            "eef_pose",
            7,
            quantity="end_effector_pose",
            representation="xyz_quaternion",
            quaternion_order="xyzw",
        )
        gripper = state(obs, "gripper_position", 2, quantity="gripper_position", representation="vector")
        axis_angle = quaternion_xyzw_to_axis_angle(pose[3:])
        return {
            "observation.images.image": np.ascontiguousarray(camera(obs, "main").rgb[::-1, ::-1]),
            "observation.images.image2": np.ascontiguousarray(camera(obs, "wrist").rgb[::-1, ::-1]),
            "observation.state": np.ascontiguousarray(np.concatenate((pose[:3], axis_angle, gripper)), dtype=np.float32),
            "task": obs.instruction,
        }

    def select_native(self, value: Any) -> np.ndarray:
        actions = policy_actions(value, 7)
        if actions.shape[0] != MODEL_HORIZON:
            raise AdapterError("pi0.5 LIBERO must return 50 actions")
        return np.ascontiguousarray(np.clip(actions[:EXECUTION_HORIZON], -1.0, 1.0), dtype=np.float32)

    def decode_selected(
        self,
        selected: np.ndarray,
        *,
        request_id: str,
        session_id: str,
        start_step: int,
    ) -> CanonicalActionChunk:
        actions = policy_actions(selected, 7)
        if actions.shape[0] != EXECUTION_HORIZON:
            raise AdapterError("pi0.5 LIBERO execution horizon must be 10")
        return action_chunk(
            actions,
            spec=self.action_spec,
            execution_count=EXECUTION_HORIZON,
            request_id=request_id,
            session_id=session_id,
            start_step=start_step,
        )

    def decode(self, value: Any, *, request_id: str, session_id: str, start_step: int) -> CanonicalActionChunk:
        return self.decode_selected(
            self.select_native(value),
            request_id=request_id,
            session_id=session_id,
            start_step=start_step,
        )

    def commit(self, native_action: Any) -> None:
        action = np.asarray(native_action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise AdapterError("committed pi0.5 action must have shape [7]")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise AdapterError("committed pi0.5 action must be within [-1, 1]")
