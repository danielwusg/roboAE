from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation

from .contracts import AdapterError, action_chunk, action_spec, camera, observation, policy_actions, state


DROID_ACTION_SPEC = action_spec(
    arms=("arm",),
    channels=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"),
    semantics=("absolute", "absolute", "absolute", "absolute", "absolute", "absolute", "absolute", "binary"),
    frame="franka_joint_space",
    rotation="joint_position",
    gripper="binary_closed_one",
    period_s=1 / 15,
)

MOLMOBOT_GRIPPER_STATE_CLOSED = np.float32(0.824033)


class DroidJointPositionAdapter:
    action_spec: CanonicalActionSpec = DROID_ACTION_SPEC

    def __init__(
        self,
        action_horizon: int,
        *,
        camera_order: tuple[str, ...],
        gripper_threshold: float,
    ) -> None:
        if type(action_horizon) is not int or action_horizon < 1:
            raise AdapterError("DROID action horizon must be positive")
        if not camera_order or any(name not in {"external", "wrist"} for name in camera_order):
            raise AdapterError("DROID camera order is invalid")
        if type(gripper_threshold) not in (int, float) or not np.isfinite(gripper_threshold):
            raise AdapterError("DROID gripper threshold must be finite")
        self.action_horizon = action_horizon
        self.execution_count = action_horizon
        self.camera_order = camera_order
        self.gripper_threshold = float(gripper_threshold)
        self._episode_id: str | None = None

    def reset(self, value: FairObservation | Mapping[str, Any]) -> None:
        self._episode_id = observation(value).episode_id

    def encode(self, value: FairObservation | Mapping[str, Any]) -> dict[str, Any]:
        obs = observation(value)
        if obs.episode_id != self._episode_id:
            raise AdapterError("adapter.reset must run once at episode start")
        arm = state(
            obs,
            "arm_joint_position",
            7,
            quantity="joint_position",
            representation="vector",
        )
        gripper = state(
            obs,
            "gripper_position",
            1,
            quantity="gripper_position",
            representation="vector",
        )
        images = {
            name: camera(obs, name).rgb
            for name in {"external", "wrist"}
        }
        return {
            "images": tuple(np.ascontiguousarray(images[name]).copy() for name in self.camera_order),
            "state": np.ascontiguousarray(np.concatenate((arm, gripper)), dtype=np.float32),
            "task": obs.instruction,
        }

    def select_native(self, value: Any) -> np.ndarray:
        actions = policy_actions(value, 8)
        if actions.shape != (self.action_horizon, 8):
            raise AdapterError(f"DROID policy must return {self.action_horizon} actions")
        selected = actions.copy()
        selected[:, 7] = (selected[:, 7] > self.gripper_threshold).astype(np.float32)
        return np.ascontiguousarray(selected, dtype=np.float32)

    def decode_selected(
        self,
        selected: np.ndarray,
        *,
        request_id: str,
        session_id: str,
        start_step: int,
    ) -> CanonicalActionChunk:
        actions = policy_actions(selected, 8)
        if actions.shape != (self.action_horizon, 8):
            raise AdapterError(f"DROID execution horizon must be {self.action_horizon}")
        if not np.isin(actions[:, 7], (0.0, 1.0)).all():
            raise AdapterError("DROID gripper actions must be binary")
        return action_chunk(
            actions,
            spec=self.action_spec,
            execution_count=self.action_horizon,
            request_id=request_id,
            session_id=session_id,
            start_step=start_step,
        )

    def commit(self, native_action: Any) -> None:
        action = np.asarray(native_action)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise AdapterError("committed DROID action must have shape [8]")
        if action[7] not in (0.0, 1.0):
            raise AdapterError("committed DROID gripper action must be binary")


class MolmoAct2DroidAdapter(DroidJointPositionAdapter):
    def __init__(self) -> None:
        super().__init__(15, camera_order=("external", "external", "wrist"), gripper_threshold=0.5)


class MolmoBotDroidAdapter(DroidJointPositionAdapter):
    def __init__(self) -> None:
        super().__init__(16, camera_order=("external", "wrist"), gripper_threshold=128.0)
        self.execution_count = 8

    def encode(self, value: FairObservation | Mapping[str, Any]) -> dict[str, Any]:
        encoded = super().encode(value)
        model_state = np.asarray(encoded["state"], dtype=np.float32).copy()
        if not 0.0 <= model_state[7] <= 1.0:
            raise AdapterError("MolmoBot DROID gripper state must be in [0, 1]")
        model_state[7] *= MOLMOBOT_GRIPPER_STATE_CLOSED
        encoded["state"] = np.ascontiguousarray(model_state, dtype=np.float32)
        return encoded


class OpenPiDroidJointPositionAdapter(DroidJointPositionAdapter):
    def __init__(self, action_horizon: int) -> None:
        super().__init__(
            action_horizon,
            camera_order=("external", "wrist"),
            gripper_threshold=0.5,
        )
