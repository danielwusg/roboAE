from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation

from .contracts import AdapterError, action_chunk, action_spec, camera, observation, state


MODEL_HORIZON = 16
EXECUTION_HORIZON = 8
VIDEO_DELTA_INDICES = (-6, -4, -2, 0)
CAMERA_KEYS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
STATE_KEYS = (
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
    "base_position",
    "base_rotation",
)
ACTION_KEYS = (
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
    ("base_motion", 4),
    ("control_mode", 1),
)
TARGET_TASK_HORIZONS = {
    "CloseBlenderLid": 600,
    "CloseFridge": 600,
    "CloseToasterOvenDoor": 300,
    "CoffeeSetupMug": 400,
    "NavigateKitchen": 300,
    "OpenCabinet": 700,
    "OpenDrawer": 500,
    "OpenStandMixerHead": 300,
    "PickPlaceCounterToCabinet": 500,
    "PickPlaceCounterToStove": 400,
    "PickPlaceDrawerToCounter": 500,
    "PickPlaceSinkToCounter": 600,
    "PickPlaceToasterToCounter": 400,
    "SlideDishwasherRack": 300,
    "TurnOffStove": 500,
    "TurnOnElectricKettle": 300,
    "TurnOnMicrowave": 300,
    "TurnOnSinkFaucet": 400,
    "DeliverStraw": 1700,
    "GetToastedBread": 2000,
    "KettleBoiling": 1000,
    "LoadDishwasher": 1200,
    "PackIdenticalLunches": 2600,
    "PreSoakPan": 1600,
    "PrepareCoffee": 1200,
    "RinseSinkBasin": 900,
    "ScrubCuttingBoard": 800,
    "SearingMeat": 2900,
    "SetUpCuttingStation": 1600,
    "StackBowlsCabinet": 1400,
    "SteamInMicrowave": 1400,
    "StirVegetables": 1600,
    "StoreLeftoversInBowl": 1700,
    "WashLettuce": 1100,
    "ArrangeBreadBasket": 2900,
    "ArrangeTea": 1500,
    "BreadSelection": 1300,
    "CategorizeCondiments": 1100,
    "CuttingToolSelection": 800,
    "GarnishPancake": 1800,
    "GatherTableware": 1500,
    "HeatKebabSandwich": 1800,
    "MakeIceLemonade": 2000,
    "PanTransfer": 1200,
    "PortionHotDogs": 1500,
    "RecycleBottlesByType": 1900,
    "SeparateFreezerRack": 1600,
    "WaffleReheat": 2700,
    "WashFruitColander": 2100,
    "WeighIngredients": 2000,
}
TARGET_TASK_GROUPS = {
    "atomic_seen": (
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    ),
    "composite_seen": (
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "PackIdenticalLunches",
        "PreSoakPan",
        "PrepareCoffee",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    ),
    "composite_unseen": (
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    ),
}

RENDER_DARK_THRESHOLD = 8
RENDER_MAX_DARK_FRACTION = 0.75
RENDER_STATIC_SPATIAL_DELTA = 25.0

RLDX_ROBOCASA365_ACTION_SPEC = action_spec(
    arms=("right_arm", "mobile_base", "torso"),
    channels=(
        "right_arm_base_dx",
        "right_arm_base_dy",
        "right_arm_base_dz",
        "right_arm_base_drx",
        "right_arm_base_dry",
        "right_arm_base_drz",
        "right_gripper_close",
        "mobile_base_vx",
        "mobile_base_vy",
        "mobile_base_yaw_velocity",
        "torso_joint_delta",
        "base_control_mode",
    ),
    semantics=(
        "delta",
        "delta",
        "delta",
        "delta",
        "delta",
        "delta",
        "binary",
        "velocity",
        "velocity",
        "velocity",
        "delta",
        "categorical",
    ),
    frame="per_channel_robot_base_frames",
    rotation="axis_angle",
    gripper="closed_positive",
    period_s=1 / 20,
    value_encoding="normalized_controller",
    controller_output_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.05, 1.0),
)


def _action_array(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (MODEL_HORIZON, width) or not np.isfinite(array).all():
        raise AdapterError(f"RLDX action {name!r} must have shape [1,16,{width}] or [16,{width}]")
    return np.ascontiguousarray(array)


def validate_robocasa365_rgb(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if (
        image.dtype != np.dtype("uint8")
        or image.ndim != 3
        or image.shape[0] < 2
        or image.shape[1] < 2
        or image.shape[2] != 3
    ):
        raise AdapterError(f"RoboCasa365 camera {name!r} must be uint8 HWC RGB")
    pixels = image.astype(np.int16)
    dark_fraction = float(np.mean(pixels < RENDER_DARK_THRESHOLD))
    horizontal_delta = float(np.mean(np.abs(pixels[:, 1:] - pixels[:, :-1])))
    vertical_delta = float(np.mean(np.abs(pixels[1:] - pixels[:-1])))
    value_range = int(pixels.max()) - int(pixels.min())
    if (
        value_range == 0
        or dark_fraction >= RENDER_MAX_DARK_FRACTION
        or (
            horizontal_delta >= RENDER_STATIC_SPATIAL_DELTA
            and vertical_delta >= RENDER_STATIC_SPATIAL_DELTA
        )
    ):
        raise AdapterError(
            f"RoboCasa365 camera {name!r} failed render integrity: range={value_range}, "
            f"dark_fraction={dark_fraction:.6f}, horizontal_delta={horizontal_delta:.6f}, "
            f"vertical_delta={vertical_delta:.6f}"
        )
    return np.ascontiguousarray(image)


class RLDXRoboCasa365Adapter:
    action_spec: CanonicalActionSpec = RLDX_ROBOCASA365_ACTION_SPEC
    execution_count = EXECUTION_HORIZON

    def __init__(self) -> None:
        self._episode_id: str | None = None

    def reset(self, value: FairObservation | Mapping[str, Any]) -> None:
        self._episode_id = observation(value).episode_id

    def encode_current(self, value: FairObservation | Mapping[str, Any]) -> dict[str, Any]:
        obs = observation(value)
        if obs.episode_id != self._episode_id:
            raise AdapterError("adapter.reset must run once at episode start")
        return {
            "video": {
                key: np.ascontiguousarray(camera(obs, key).rgb)
                for key in CAMERA_KEYS
            },
            "state": {
                "end_effector_position_relative": state(
                    obs,
                    "end_effector_position_relative",
                    3,
                    quantity="end_effector_pose",
                    representation="vector",
                ),
                "end_effector_rotation_relative": state(
                    obs,
                    "end_effector_rotation_relative",
                    4,
                    quantity="end_effector_pose",
                    representation="quaternion",
                    quaternion_order="xyzw",
                ),
                "gripper_qpos": state(
                    obs,
                    "gripper_qpos",
                    2,
                    quantity="gripper_position",
                    representation="vector",
                ),
                "base_position": state(
                    obs,
                    "base_position",
                    3,
                    quantity="base_pose",
                    representation="vector",
                ),
                "base_rotation": state(
                    obs,
                    "base_rotation",
                    4,
                    quantity="base_pose",
                    representation="quaternion",
                    quaternion_order="xyzw",
                ),
            },
            "language": obs.instruction,
            "step_index": obs.step_index,
        }

    def temporal_batch(self, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not history:
            raise AdapterError("RLDX observation history is empty")
        latest = history[-1]
        latest_step = latest.get("step_index")
        if type(latest_step) is not int:
            raise AdapterError("RLDX observation history has no integer step index")
        expected_steps = list(range(latest_step - len(history) + 1, latest_step + 1))
        if [item.get("step_index") for item in history] != expected_steps:
            raise AdapterError("RLDX observation history must be contiguous")
        indices = [max(0, len(history) - 1 + delta) for delta in VIDEO_DELTA_INDICES]
        result: dict[str, Any] = {}
        for key in CAMERA_KEYS:
            frames = [np.asarray(history[index]["video"][key], dtype=np.uint8) for index in indices]
            result[f"video.{key}"] = np.ascontiguousarray(np.stack(frames, axis=0)[None])
        for key in STATE_KEYS:
            value = np.asarray(latest["state"][key], dtype=np.float32)
            result[f"state.{key}"] = np.ascontiguousarray(value[None, None])
        result["annotation.human.task_description"] = [str(latest["language"])]
        return result

    def select_native(self, value: Any) -> np.ndarray:
        if not isinstance(value, Mapping):
            raise AdapterError("RLDX policy response must be a mapping")
        expected = {f"action.{name}" for name, _ in ACTION_KEYS}
        if set(value) != expected:
            raise AdapterError("RLDX policy response action keys differ")
        arrays = [_action_array(value[f"action.{name}"], width, name) for name, width in ACTION_KEYS]
        actions = np.concatenate(arrays, axis=-1)
        return np.ascontiguousarray(np.clip(actions[:EXECUTION_HORIZON], -1.0, 1.0), dtype=np.float32)

    def decode_selected(
        self,
        selected: np.ndarray,
        *,
        request_id: str,
        session_id: str,
        start_step: int,
    ) -> CanonicalActionChunk:
        actions = np.asarray(selected, dtype=np.float32)
        if actions.shape != (EXECUTION_HORIZON, 12) or not np.isfinite(actions).all():
            raise AdapterError("RLDX execution chunk must have shape [8,12]")
        return action_chunk(
            np.ascontiguousarray(np.clip(actions, -1.0, 1.0)),
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
        if action.shape != (12,) or not np.isfinite(action).all():
            raise AdapterError("committed RLDX action must have shape [12]")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise AdapterError("committed RLDX action must be within [-1,1]")


def native_action_dict(value: Any) -> dict[str, np.ndarray]:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (12,) or not np.isfinite(action).all():
        raise AdapterError("RoboCasa native action must have shape [12]")
    action = np.clip(action, -1.0, 1.0)
    return {
        "action.end_effector_position": np.ascontiguousarray(action[0:3]),
        "action.end_effector_rotation": np.ascontiguousarray(action[3:6]),
        "action.gripper_close": np.ascontiguousarray(action[6:7]),
        "action.base_motion": np.ascontiguousarray(action[7:11]),
        "action.control_mode": np.ascontiguousarray(action[11:12]),
    }
