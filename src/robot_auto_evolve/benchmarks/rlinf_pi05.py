from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation

from .contracts import AdapterError, action_chunk, action_spec, camera, observation, policy_actions, state
from .pi05 import quaternion_xyzw_to_axis_angle


MODEL_HORIZON = 10
EXECUTION_HORIZON = 5

RLINF_PI05_LIBERO_ACTION_SPEC = action_spec(
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


class RLinfPi05LiberoAdapter:
    action_spec: CanonicalActionSpec = RLINF_PI05_LIBERO_ACTION_SPEC
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
            "observation/image": np.ascontiguousarray(camera(obs, "main").rgb[::-1, ::-1]),
            "observation/wrist_image": np.ascontiguousarray(camera(obs, "wrist").rgb[::-1, ::-1]),
            "observation/state": np.ascontiguousarray(
                np.concatenate((pose[:3], axis_angle, gripper)), dtype=np.float32
            ),
            "prompt": obs.instruction,
        }

    def select_native(self, value: Any) -> np.ndarray:
        actions = policy_actions(value, 7)
        if actions.shape[0] != MODEL_HORIZON:
            raise AdapterError("RLinf pi0.5 LIBERO must return 10 actions")
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
            raise AdapterError("RLinf pi0.5 LIBERO execution horizon must be 5")
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
            raise AdapterError("committed RLinf pi0.5 action must have shape [7]")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise AdapterError("committed RLinf pi0.5 action must be within [-1, 1]")
