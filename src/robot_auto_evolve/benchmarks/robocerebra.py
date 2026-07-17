from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import CanonicalActionSpec, FairObservation, StrictSchemaError

from .contracts import AdapterError, action_spec, camera, observation, policy_actions, state
from .pi05 import quaternion_xyzw_to_axis_angle


CONDITIONS = (
    "Ideal",
    "Memory_Execution",
    "Memory_Exploration",
    "Mix",
    "Observation_Mismatching",
    "Random_Disturbance",
)
MODEL_HORIZON = 50
EXECUTION_HORIZON = 8
SETTLE_STEPS = 15
SEGMENT_STEPS = 150
PUBLIC_PROTOCOL = "robocerebra_released_anchor_resume_smolvla_v1"
FULL_STACK_SMOKE_PROTOCOL = "robocerebra_full_stack_smoke_v1"
FULL_STACK_SMOKE_HORIZON = 9
PAPER_TRIALS_PER_CASE = 10
RELEASED_DEFAULT_TRIALS_PER_CASE = 5

SMOLVLA_ROBOCEREBRA_ACTION_SPEC = action_spec(
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


@dataclass(frozen=True)
class RoboCerebraCase:
    task_id: str
    condition: str
    case_id: str
    effective_condition: str
    instruction: str
    steps: tuple[str, ...]
    num_steps: int
    bddl_path: str
    demo_path: str
    goal_path: str
    description_path: str
    description_json_path: str
    init_path: str | None

    @property
    def horizon(self) -> int:
        return SEGMENT_STEPS * self.num_steps - SETTLE_STEPS


def task_id(condition: str, case_id: str) -> str:
    if condition not in CONDITIONS or re.fullmatch(r"case(?:[1-9]|10)", case_id) is None:
        raise StrictSchemaError("RoboCerebra condition or case ID differs")
    return f"robocerebra_public60::{condition}::{case_id}"


def parse_task_id(value: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"robocerebra_public60::(Ideal|Memory_Execution|Memory_Exploration|Mix|Observation_Mismatching|Random_Disturbance)::(case(?:[1-9]|10))",
        value,
    )
    if match is None:
        raise StrictSchemaError("RoboCerebra task ID differs")
    return match.group(1), match.group(2)


def load_case_catalog(path: str | Path) -> tuple[RoboCerebraCase, ...]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "dataset", "protocol", "cases"}:
        raise StrictSchemaError("RoboCerebra case catalog fields differ")
    if value["schema_version"] != 1:
        raise StrictSchemaError("RoboCerebra case catalog version differs")
    rows = []
    expected_fields = {
        "task_id",
        "condition",
        "case_id",
        "effective_condition",
        "instruction",
        "steps",
        "num_steps",
        "bddl_path",
        "demo_path",
        "goal_path",
        "description_path",
        "description_json_path",
        "init_path",
    }
    for item in value["cases"]:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise StrictSchemaError("RoboCerebra case catalog row differs")
        condition, case_id = parse_task_id(item["task_id"])
        if item["condition"] != condition or item["case_id"] != case_id:
            raise StrictSchemaError("RoboCerebra case catalog identity differs")
        steps = tuple(item["steps"])
        if not steps or len(steps) != item["num_steps"] or any(type(step) is not str or not step for step in steps):
            raise StrictSchemaError("RoboCerebra case catalog steps differ")
        row = RoboCerebraCase(
            task_id=item["task_id"],
            condition=condition,
            case_id=case_id,
            effective_condition=item["effective_condition"],
            instruction=item["instruction"],
            steps=steps,
            num_steps=item["num_steps"],
            bddl_path=item["bddl_path"],
            demo_path=item["demo_path"],
            goal_path=item["goal_path"],
            description_path=item["description_path"],
            description_json_path=item["description_json_path"],
            init_path=item["init_path"],
        )
        if row.horizon < 1:
            raise StrictSchemaError("RoboCerebra case horizon differs")
        rows.append(row)
    result = tuple(sorted(rows, key=lambda item: (CONDITIONS.index(item.condition), int(item.case_id[4:]))))
    if len(result) != 60 or {item.condition for item in result} != set(CONDITIONS):
        raise StrictSchemaError("RoboCerebra case catalog must contain 60 cases")
    if any(sum(item.condition == condition for item in result) != 10 for condition in CONDITIONS):
        raise StrictSchemaError("RoboCerebra condition must contain 10 cases")
    if len({item.task_id for item in result}) != len(result):
        raise StrictSchemaError("RoboCerebra case catalog has duplicate task IDs")
    return result


def checkpoint_state_contract(config_path: str | Path, normalizer_path: str | Path) -> dict[str, int]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    declared = config["input_features"]["observation.state"]["shape"]
    if declared != [6]:
        raise StrictSchemaError("SmolVLA RoboCerebra declared state width changed")
    source = Path(normalizer_path)
    with source.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        if header_size < 2 or header_size > source.stat().st_size - 8:
            raise StrictSchemaError("SmolVLA RoboCerebra normalizer header differs")
        header = json.loads(handle.read(header_size))
    tensor = header.get("observation.state.mean")
    if not isinstance(tensor, dict) or tensor.get("shape") != [8]:
        raise StrictSchemaError("SmolVLA RoboCerebra normalizer state tensor differs")
    normalized = int(tensor["shape"][0])
    if normalized != 8:
        raise StrictSchemaError("SmolVLA RoboCerebra normalizer state width changed")
    return {"declared_width": 6, "normalizer_width": 8, "runtime_width": 8}


class SmolVLARoboCerebraAdapter:
    action_spec: CanonicalActionSpec = SMOLVLA_ROBOCEREBRA_ACTION_SPEC
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
        runtime_state = np.concatenate((pose[:3], quaternion_xyzw_to_axis_angle(pose[3:]), gripper))
        if runtime_state.shape != (8,):
            raise AdapterError("SmolVLA RoboCerebra runtime state must have width 8")
        return {
            "observation.images.image": np.ascontiguousarray(camera(obs, "main").rgb[::-1, ::-1]),
            "observation.images.wrist_image": np.ascontiguousarray(camera(obs, "wrist").rgb[::-1, ::-1]),
            "observation.state": np.ascontiguousarray(runtime_state, dtype=np.float32),
            "task": obs.instruction,
        }

    def select_native(self, value: Any) -> np.ndarray:
        actions = policy_actions(value, 7)
        if actions.shape[0] != MODEL_HORIZON:
            raise AdapterError("SmolVLA RoboCerebra must return 50 actions")
        selected = np.ascontiguousarray(actions[:EXECUTION_HORIZON], dtype=np.float32)
        selected[:, :6] = np.clip(selected[:, :6], -1.0, 1.0)
        selected[:, 6] = np.where(selected[:, 6] < 0.5, 1.0, -1.0)
        return selected

    def commit(self, native_action: Any) -> None:
        action = np.asarray(native_action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise AdapterError("committed SmolVLA RoboCerebra action must have shape [7]")
        if np.any(action < -1.0) or np.any(action > 1.0) or action[6] not in {-1.0, 1.0}:
            raise AdapterError("committed SmolVLA RoboCerebra action differs")
