from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation

from .contracts import AdapterError, action_chunk, action_spec, camera, observation, policy_actions, state
from .transforms import (
    euler_xyz_to_matrix,
    matrix_to_rotation6d,
    quaternion_xyzw_to_rotation6d,
    rotation6d_columns_to_axis_angle,
    rotation6d_to_euler_xyz,
    rotation6d_to_quaternion_xyzw,
)


MODEL_HORIZON = 30
DENOISE_STEPS = 10

WIDOWX_GRIPPER_THRESHOLDS = {
    "widowx_spoon_on_towel": 0.70,
    "widowx_carrot_on_plate": 0.95,
    "widowx_stack_cube": 0.91,
    "widowx_put_eggplant_in_basket": 0.80,
}

GOOGLE_VA_RULES = {
    "google_robot_pick_coke_can": (0.25, 10),
    "google_robot_move_near": (0.25, 10),
    "google_robot_open_drawer": (0.25, 10),
    "google_robot_close_drawer": (0.25, 10),
    "google_robot_place_apple_in_closed_top_drawer": (0.30, 10),
}

GOOGLE_VM_RULES = {
    "google_robot_pick_coke_can": (0.25, 10),
    "google_robot_move_near": (0.25, 10),
    "google_robot_open_drawer": (0.35, 10),
    "google_robot_close_drawer": (0.35, 10),
    "google_robot_place_apple_in_closed_top_drawer": (0.28, 6),
}

LIBERO_TASKS = frozenset({
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
})

CALVIN_TASKS = frozenset({
    "rotate_red_block_right", "rotate_red_block_left", "rotate_blue_block_right", "rotate_blue_block_left",
    "rotate_pink_block_right", "rotate_pink_block_left", "push_red_block_right", "push_red_block_left",
    "push_blue_block_right", "push_blue_block_left", "push_pink_block_right", "push_pink_block_left",
    "move_slider_left", "move_slider_right", "open_drawer", "close_drawer", "lift_red_block_table",
    "lift_red_block_slider", "lift_red_block_drawer", "lift_blue_block_table", "lift_blue_block_slider",
    "lift_blue_block_drawer", "lift_pink_block_table", "lift_pink_block_slider", "lift_pink_block_drawer",
    "place_in_slider", "place_in_drawer", "stack_block", "unstack_block", "turn_on_lightbulb",
    "turn_off_lightbulb", "turn_on_led", "turn_off_led", "push_into_drawer",
})

ROBOTWIN_TASKS = frozenset({
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size", "click_alarmclock",
    "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block", "handover_mic", "hanging_mug",
    "lift_pot", "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left",
    "place_a2b_right", "place_bread_basket", "place_bread_skillet", "place_burger_fries",
    "place_can_basket", "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two", "stamp_seal",
    "turn_switch",
})

VLABENCH_TASKS = frozenset({
    "add_condiment", "insert_flower", "select_book", "select_chemistry_tube", "select_drink", "select_fruit",
    "select_mahjong", "select_nth_largest_poker", "select_painting", "select_poker", "select_toy",
    "select_unique_type_mahjong",
})


def _single_channels(rotation: str) -> tuple[str, ...]:
    suffix = ("rx", "ry", "rz") if rotation != "quaternion" else ("qx", "qy", "qz", "qw")
    return ("x", "y", "z") + suffix + ("gripper",)


def _single_spec(rotation: str, period_s: float | None, gripper: str, frame: str) -> CanonicalActionSpec:
    channels = _single_channels(rotation)
    return action_spec(
        arms=("arm",),
        channels=channels,
        semantics=("absolute",) * (len(channels) - 1) + ("binary",),
        frame=frame,
        rotation=rotation,
        quaternion_order="xyzw" if rotation == "quaternion" else "none",
        gripper=gripper,
        period_s=period_s,
    )


LIBERO_ACTION_SPEC = _single_spec("axis_angle", 1 / 20, "closed_positive", "world")
CALVIN_ACTION_SPEC = _single_spec("quaternion", 1 / 30, "open_positive", "world")
SIMPLER_GOOGLE_ACTION_SPEC = _single_spec("euler_xyz", 1 / 3, "closed_positive", "robot_base")
SIMPLER_WIDOWX_ACTION_SPEC = _single_spec("euler_xyz", 1 / 5, "open_positive", "robot_base")
ROBOTWIN_CHANNELS = tuple(
    f"{arm}_{channel}"
    for arm in ("left", "right")
    for channel in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
)
ROBOTWIN_ACTION_SPEC = action_spec(
    arms=("left", "right"),
    channels=ROBOTWIN_CHANNELS,
    semantics=tuple("binary" if name.endswith("gripper") else "absolute" for name in ROBOTWIN_CHANNELS),
    frame="world",
    rotation="quaternion",
    quaternion_order="wxyz",
    gripper="open_positive",
    period_s=None,
)
VLABENCH_ACTION_SPEC = action_spec(
    arms=("arm",),
    channels=("x", "y", "z", "rx", "ry", "rz", "left_finger", "right_finger"),
    semantics=("absolute",) * 8,
    frame="world",
    rotation="euler_xyz",
    gripper="none",
    period_s=None,
)
VLABENCH_BASE_ENV_STEP_S = 0.1


def _rotation6d(value: FairObservation, prefix: str = "") -> np.ndarray:
    names = {vector.spec.name for vector in value.proprioception.vectors}
    rotation_name = f"{prefix}eef_rotation6d"
    if rotation_name in names:
        return state(value, rotation_name, 6, quantity="end_effector_pose", representation="vector")
    euler_name = f"{prefix}eef_euler_xyz"
    if euler_name in names:
        euler = state(value, euler_name, 3, quantity="end_effector_pose", representation="vector")
        return matrix_to_rotation6d(euler_xyz_to_matrix(euler))
    pose = state(
        value,
        f"{prefix}eef_pose",
        7,
        quantity="end_effector_pose",
        representation="xyz_quaternion",
        quaternion_order="xyzw",
    )
    return quaternion_xyzw_to_rotation6d(pose[3:])


def _position(value: FairObservation, prefix: str = "") -> np.ndarray:
    names = {vector.spec.name for vector in value.proprioception.vectors}
    name = f"{prefix}eef_position"
    if name in names:
        return state(value, name, 3, quantity="end_effector_pose", representation="vector")
    return state(
        value,
        f"{prefix}eef_pose",
        7,
        quantity="end_effector_pose",
        representation="xyz_quaternion",
        quaternion_order="xyzw",
    )[:3]


def _gripper(value: FairObservation, prefix: str = "") -> np.ndarray:
    return state(value, f"{prefix}gripper_position", 1, quantity="gripper_position", representation="vector")


def _upstream_wxyz_pose(value: FairObservation, prefix: str = "") -> np.ndarray:
    return state(
        value,
        f"{prefix}eef_pose",
        7,
        quantity="end_effector_pose",
        representation="xyz_quaternion",
        quaternion_order="wxyz",
    )


def _upstream_wxyz_as_xyzw_rotation6d(pose: np.ndarray) -> np.ndarray:
    return quaternion_xyzw_to_rotation6d(pose[3:])


def _upstream_xyzw_numbers_consumed_as_wxyz(rotation6d: np.ndarray) -> np.ndarray:
    return rotation6d_to_quaternion_xyzw(rotation6d)


class _XVLAAdapter:
    domain_id: int
    image_names: tuple[str, ...]
    include_steps = True
    action_spec: CanonicalActionSpec
    execution_count: int

    def __init__(self) -> None:
        self._episode_id: str | None = None

    def reset(self, value: FairObservation | Mapping[str, Any]) -> None:
        obs = observation(value)
        self._episode_id = obs.episode_id
        self._reset_state(obs)

    def _reset_state(self, value: FairObservation) -> None:
        return None

    def _ensure_episode(self, value: FairObservation) -> None:
        if self._episode_id != value.episode_id:
            raise AdapterError("adapter.reset must run once at episode start")

    def _proprio(self, value: FairObservation) -> np.ndarray:
        raise NotImplementedError

    def encode(self, value: FairObservation | Mapping[str, Any]) -> dict[str, Any]:
        obs = observation(value)
        self._ensure_episode(obs)
        payload: dict[str, Any] = {
            "language_instruction": obs.instruction,
            "proprio": np.ascontiguousarray(self._proprio(obs), dtype=np.float32),
            "domain_id": self.domain_id,
        }
        if self.include_steps:
            payload["steps"] = DENOISE_STEPS
        for index, name in enumerate(self.image_names):
            payload[f"image{index}"] = camera(obs, name).rgb
        return payload

    def _select(self, actions: np.ndarray) -> np.ndarray:
        return actions

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _update_state(self, selected: np.ndarray) -> None:
        return None

    def select_native(self, value: Any) -> np.ndarray:
        actions = policy_actions(value, 20)
        if actions.shape[0] != MODEL_HORIZON:
            raise AdapterError(f"X-VLA must return {MODEL_HORIZON} actions")
        selected = np.ascontiguousarray(self._select(actions))
        if selected.shape[0] != self.execution_count:
            raise AdapterError("frozen execution count mismatch")
        return selected

    def commit(self, native_action: Any) -> None:
        action = np.asarray(native_action, dtype=np.float32)
        if action.shape != (20,) or not np.isfinite(action).all():
            raise AdapterError("committed X-VLA action must have shape [20]")
        self._update_state(action[None])

    def decode_selected(
        self,
        selected: np.ndarray,
        *,
        request_id: str,
        session_id: str,
        start_step: int,
    ) -> CanonicalActionChunk:
        selected = policy_actions(selected, 20)
        if selected.shape[0] != self.execution_count:
            raise AdapterError("frozen execution count mismatch")
        decoded = self._decode_values(selected)
        return action_chunk(
            decoded,
            spec=self.action_spec,
            execution_count=self.execution_count,
            request_id=request_id,
            session_id=session_id,
            start_step=start_step,
        )

    def decode(
        self,
        value: Any,
        *,
        request_id: str,
        session_id: str,
        start_step: int,
    ) -> CanonicalActionChunk:
        selected = self.select_native(value)
        return self.decode_selected(
            selected,
            request_id=request_id,
            session_id=session_id,
            start_step=start_step,
        )


class XVLALiberoAdapter(_XVLAAdapter):
    domain_id = 3
    image_names = ("main", "wrist")
    action_spec = LIBERO_ACTION_SPEC
    execution_count = 30

    def _reset_state(self, value: FairObservation) -> None:
        rotation = _rotation6d(value)
        rotation_columns = np.concatenate((rotation[0:5:2], rotation[1:6:2]))
        current = np.concatenate((_position(value), rotation_columns, np.zeros(1, dtype=np.float32)))
        self._memory = np.concatenate((current, np.zeros_like(current))).astype(np.float32)

    def _proprio(self, value: FairObservation) -> np.ndarray:
        return self._memory

    def encode(self, value: FairObservation | Mapping[str, Any]) -> dict[str, Any]:
        payload = super().encode(value)
        payload["image0"] = np.ascontiguousarray(payload["image0"][::-1, ::-1])
        return payload

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                actions[:, :3],
                rotation6d_columns_to_axis_angle(actions[:, 3:9]),
                np.where(actions[:, 9:10] > 0.5, 1.0, -1.0),
            ),
            axis=-1,
        )

    def _update_state(self, selected: np.ndarray) -> None:
        self._memory[:9] = selected[-1, :9]


class XVLACalvinAdapter(_XVLAAdapter):
    domain_id = 2
    image_names = ("static", "wrist")
    action_spec = CALVIN_ACTION_SPEC
    execution_count = 20

    def _reset_state(self, value: FairObservation) -> None:
        gripper = state(
            value,
            "gripper_action",
            1,
            quantity="base_control_state",
            representation="vector",
        )
        current = np.concatenate((_position(value), _rotation6d(value), (gripper > 0).astype(np.float32)))
        self._memory = np.concatenate((current, np.zeros_like(current))).astype(np.float32)

    def _proprio(self, value: FairObservation) -> np.ndarray:
        return self._memory

    def _select(self, actions: np.ndarray) -> np.ndarray:
        return actions[:20]

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                actions[:, :3],
                rotation6d_to_quaternion_xyzw(actions[:, 3:9]),
                np.where(actions[:, 9:10] < 0.8, 1.0, -1.0),
            ),
            axis=-1,
        )

    def _update_state(self, selected: np.ndarray) -> None:
        self._memory[:10] = selected[-1, :10]


class XVLAWidowXAdapter(_XVLAAdapter):
    domain_id = 0
    image_names = ("main",)
    action_spec = SIMPLER_WIDOWX_ACTION_SPEC
    execution_count = 30

    def __init__(self, task_id: str) -> None:
        if task_id not in WIDOWX_GRIPPER_THRESHOLDS:
            raise AdapterError(f"unsupported WidowX task {task_id!r}")
        super().__init__()
        self.task_id = task_id
        self.gripper_threshold = WIDOWX_GRIPPER_THRESHOLDS[task_id]

    def _reset_state(self, value: FairObservation) -> None:
        current = np.concatenate(
            (
                _position(value),
                np.array((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            )
        )
        self._memory = np.concatenate((current, np.zeros_like(current))).astype(np.float32)

    def _proprio(self, value: FairObservation) -> np.ndarray:
        return self._memory

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        euler = rotation6d_to_euler_xyz(actions[:, 3:9]) + np.array((0.0, np.pi / 2, 0.0))
        return np.concatenate(
            (actions[:, :3], euler, np.where(actions[:, 9:10] < self.gripper_threshold, 1.0, -1.0)),
            axis=-1,
        )

    def _update_state(self, selected: np.ndarray) -> None:
        self._memory[:10] = selected[-1, :10]


class XVLAGoogleAdapter(_XVLAAdapter):
    domain_id = 1
    image_names = ("main",)
    action_spec = SIMPLER_GOOGLE_ACTION_SPEC

    def __init__(self, task_id: str, rules: Mapping[str, tuple[float, int]] = GOOGLE_VA_RULES) -> None:
        if task_id not in rules:
            raise AdapterError(f"unsupported Google task {task_id!r}")
        super().__init__()
        self.task_id = task_id
        self.gripper_threshold, self.execution_count = rules[task_id]

    def _reset_state(self, value: FairObservation) -> None:
        self._memory = np.zeros(20, dtype=np.float32)
        self._current_xyz = _position(value).astype(np.float32).copy()

    def _proprio(self, value: FairObservation) -> np.ndarray:
        return self._memory

    def _select(self, actions: np.ndarray) -> np.ndarray:
        return actions[::2][: self.execution_count]

    def select_native(self, value: Any) -> np.ndarray:
        self._chunk_origin = self._current_xyz.copy()
        return super().select_native(value)

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        absolute = actions[:, :3] + self._current_xyz
        return np.concatenate(
            (
                absolute,
                rotation6d_to_euler_xyz(actions[:, 3:9]),
                np.where(actions[:, 9:10] > self.gripper_threshold, 1.0, -1.0),
            ),
            axis=-1,
        )

    def _update_state(self, selected: np.ndarray) -> None:
        self._current_xyz = self._current_xyz + selected[-1, :3]

    def commit(self, native_action: Any) -> None:
        action = np.asarray(native_action, dtype=np.float32)
        if action.shape != (20,) or not np.isfinite(action).all():
            raise AdapterError("committed X-VLA action must have shape [20]")
        self._current_xyz = self._chunk_origin + action[:3]


class XVLARoboTwinAdapter(_XVLAAdapter):
    domain_id = 6
    image_names = ("head", "left_wrist", "right_wrist")
    include_steps = False
    action_spec = ROBOTWIN_ACTION_SPEC
    execution_count = 30

    def _proprio(self, value: FairObservation) -> np.ndarray:
        left_pose = _upstream_wxyz_pose(value, "left_")
        right_pose = _upstream_wxyz_pose(value, "right_")
        left = np.concatenate(
            (
                left_pose[:3],
                _upstream_wxyz_as_xyzw_rotation6d(left_pose),
                1 - 2 * _gripper(value, "left_"),
            )
        )
        right = np.concatenate(
            (
                right_pose[:3],
                _upstream_wxyz_as_xyzw_rotation6d(right_pose),
                1 - 2 * _gripper(value, "right_"),
            )
        )
        return np.concatenate((left, right))

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        left = np.concatenate(
            (
                actions[:, :3],
                _upstream_xyzw_numbers_consumed_as_wxyz(actions[:, 3:9]),
                1 - 2 * (actions[:, 9:10] > 0.7),
            ),
            axis=-1,
        )
        right = np.concatenate(
            (
                actions[:, 10:13],
                _upstream_xyzw_numbers_consumed_as_wxyz(actions[:, 13:19]),
                1 - 2 * (actions[:, 19:20] > 0.7),
            ),
            axis=-1,
        )
        return np.concatenate((left, right), axis=-1)


class XVLAVLABenchAdapter(_XVLAAdapter):
    domain_id = 8
    image_names = ("main", "front", "wrist")
    action_spec = VLABENCH_ACTION_SPEC
    execution_count = 30
    base_offset = np.array((0.0, -0.4, 0.78), dtype=np.float32)

    def _proprio(self, value: FairObservation) -> np.ndarray:
        pose = _upstream_wxyz_pose(value)
        current = np.concatenate(
            (
                pose[:3] - self.base_offset,
                _upstream_wxyz_as_xyzw_rotation6d(pose),
                _gripper(value),
            )
        )
        return np.concatenate((current, np.zeros_like(current)))

    def _decode_values(self, actions: np.ndarray) -> np.ndarray:
        width = np.where(actions[:, 9:10] <= 0.5, 0.04, 0.0)
        return np.concatenate(
            (actions[:, :3] + self.base_offset, rotation6d_to_euler_xyz(actions[:, 3:9]), width, width),
            axis=-1,
        )
