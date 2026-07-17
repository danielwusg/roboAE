from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import StrictSchemaError


CALVIN_SOURCE_COMMIT = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_COMMIT = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CALVIN_TACTO_COMMIT = "dd53360d9a8c186f0d6439372ec0be0fa5e21731"
XVLA_SOURCE_COMMIT = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
OFFICIAL_SEQUENCE_COUNT = 1000
OFFICIAL_SEQUENCE_LENGTH = 5
UPSTREAM_OFFICIAL_SUBTASK_HORIZON = 360
RELEASED_XVLA_SUBTASK_HORIZON = 720
PROJECT_PROTOCOL = "calvin_official_sequence_prefix1_related_transfer_v2"
OFFICIAL_SEQUENCE_TABLE_SHA256 = "1e670a2dffe8d55d82a0c2c9a3300c677d459c0513085b2d28e745110c7965e7"
OFFICIAL_GENERATOR_SHA256 = "fbd08f8501b1c96ce564b085881e31a48396e9ab1aebdda1e75570e3387fad29"
OFFICIAL_RESET_SOURCE_SHA256 = "35e444a68a4a2cc9a8b3195a4f9ad957c2384d7096306e113557d3c0dd8788e8"

TASK_NAMES = (
    "rotate_red_block_right",
    "rotate_red_block_left",
    "rotate_blue_block_right",
    "rotate_blue_block_left",
    "rotate_pink_block_right",
    "rotate_pink_block_left",
    "push_red_block_right",
    "push_red_block_left",
    "push_blue_block_right",
    "push_blue_block_left",
    "push_pink_block_right",
    "push_pink_block_left",
    "move_slider_left",
    "move_slider_right",
    "open_drawer",
    "close_drawer",
    "lift_red_block_table",
    "lift_red_block_slider",
    "lift_red_block_drawer",
    "lift_blue_block_table",
    "lift_blue_block_slider",
    "lift_blue_block_drawer",
    "lift_pink_block_table",
    "lift_pink_block_slider",
    "lift_pink_block_drawer",
    "place_in_slider",
    "place_in_drawer",
    "stack_block",
    "unstack_block",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "push_into_drawer",
)

TASK_CATEGORIES = {
    **{name: 1 for name in TASK_NAMES[:12]},
    "move_slider_left": 2,
    "move_slider_right": 2,
    "open_drawer": 3,
    "close_drawer": 3,
    "lift_red_block_table": 4,
    "lift_blue_block_table": 4,
    "lift_pink_block_table": 4,
    "lift_red_block_slider": 5,
    "lift_blue_block_slider": 5,
    "lift_pink_block_slider": 5,
    "lift_red_block_drawer": 6,
    "lift_blue_block_drawer": 6,
    "lift_pink_block_drawer": 6,
    "place_in_slider": 7,
    "place_in_drawer": 7,
    "turn_on_lightbulb": 8,
    "turn_off_lightbulb": 8,
    "turn_on_led": 8,
    "turn_off_led": 8,
    "push_into_drawer": 9,
    "stack_block": 10,
    "unstack_block": 11,
}

TRANSFER_RELATIONS = (
    ("move_slider_left", "move_slider_right"),
    ("open_drawer", "close_drawer"),
    ("turn_on_led", "turn_off_led"),
)

INITIAL_KEYS = ("led", "lightbulb", "slider", "drawer", "red_block", "blue_block", "pink_block", "grasped")


@dataclass(frozen=True)
class CalvinSequence:
    sequence_id: str
    initial_condition: Mapping[str, int | str]
    tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"official_sequence_\d{4}", self.sequence_id) is None:
            raise StrictSchemaError("calvin sequence: invalid identity")
        condition = dict(self.initial_condition)
        if set(condition) != set(INITIAL_KEYS):
            raise StrictSchemaError("calvin sequence: invalid initial-condition keys")
        condition = {key: condition[key] for key in INITIAL_KEYS}
        tasks = tuple(self.tasks)
        if len(tasks) != OFFICIAL_SEQUENCE_LENGTH or any(task not in TASK_NAMES for task in tasks):
            raise StrictSchemaError("calvin sequence: invalid tasks")
        object.__setattr__(self, "initial_condition", condition)
        object.__setattr__(self, "tasks", tasks)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "initial_condition": dict(self.initial_condition),
            "tasks": list(self.tasks),
        }


def executed_reset_identity(
    task: str,
    sequence: CalvinSequence,
) -> tuple[str, tuple[tuple[str, int | str], ...]]:
    if task not in TASK_NAMES or not isinstance(sequence, CalvinSequence) or sequence.tasks[0] != task:
        raise StrictSchemaError("calvin executed reset: task and sequence prefix differ")
    return task, tuple((key, sequence.initial_condition[key]) for key in INITIAL_KEYS)


def select_unique_executed_resets(
    task: str,
    sequences: Sequence[CalvinSequence],
    count: int,
) -> tuple[CalvinSequence, ...]:
    if type(count) is not int or count < 1:
        raise StrictSchemaError("calvin executed reset: count must be a positive int")
    selected = []
    seen = set()
    for sequence in sequences:
        identity = executed_reset_identity(task, sequence)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(sequence)
        if len(selected) == count:
            return tuple(selected)
    raise RuntimeError(f"{task} requires {count} unique executed reset identities; only {len(seen)} are available")


def _next_states(state: Mapping[str, int | str], task: str) -> list[dict[str, int | str]]:
    current = dict(state)
    if task.startswith(("rotate_", "push_")) and task != "push_into_drawer":
        color = task.split("_")[1]
        return [current] if current[f"{color}_block"] == "table" and current["grasped"] == 0 else []
    if task == "move_slider_left":
        return [{**current, "slider": "left"}] if current["slider"] == "right" and current["grasped"] == 0 else []
    if task == "move_slider_right":
        return [{**current, "slider": "right"}] if current["slider"] == "left" and current["grasped"] == 0 else []
    if task == "open_drawer":
        return [{**current, "drawer": "open"}] if current["drawer"] == "closed" and current["grasped"] == 0 else []
    if task == "close_drawer":
        return [{**current, "drawer": "closed"}] if current["drawer"] == "open" and current["grasped"] == 0 else []
    if task.startswith("lift_"):
        _, color, _, location = task.split("_")
        block = f"{color}_block"
        valid = current["grasped"] == 0
        if location == "table":
            valid = valid and current[block] == "table"
        elif location == "slider":
            valid = valid and (
                (current[block] == "slider_left" and current["slider"] == "right")
                or (current[block] == "slider_right" and current["slider"] == "left")
            )
        else:
            valid = valid and current[block] == "drawer" and current["drawer"] == "open"
        return [{**current, block: "grasped", "grasped": 1}] if valid else []
    if task in {"place_in_slider", "place_in_drawer"}:
        if current["grasped"] != 1:
            return []
        results = []
        for color in ("red", "blue", "pink"):
            block = f"{color}_block"
            if current[block] != "grasped":
                continue
            if task == "place_in_drawer":
                if current["drawer"] == "open":
                    results.append({**current, block: "drawer", "grasped": 0})
            else:
                location = "slider_right" if current["slider"] == "right" else "slider_left"
                results.append({**current, block: location, "grasped": 0})
        return results
    if task.startswith("turn_"):
        _, target, device = task.split("_")
        start, end = (0, 1) if target == "on" else (1, 0)
        return [{**current, device: end}] if current[device] == start and current["grasped"] == 0 else []
    if task == "push_into_drawer":
        if current["drawer"] != "open" or current["grasped"] != 0:
            return []
        results = []
        for color in ("red", "blue", "pink"):
            block = f"{color}_block"
            others = [current[f"{other}_block"] for other in ("red", "blue", "pink") if other != color]
            if current[block] == "table" and all(value in {"slider_left", "slider_right"} for value in others):
                results.append({**current, block: "drawer"})
        return results
    if task == "stack_block":
        if current["grasped"] != 1:
            return []
        results = []
        for top in ("red", "blue", "pink"):
            if current[f"{top}_block"] != "grasped":
                continue
            for bottom in ("red", "blue", "pink"):
                if top != bottom and current[f"{bottom}_block"] == "table":
                    results.append({**current, f"{top}_block": "stacked_top", f"{bottom}_block": "stacked_bottom", "grasped": 0})
        return results
    if task == "unstack_block":
        if current["grasped"] != 0:
            return []
        results = []
        for top in ("red", "blue", "pink"):
            if current[f"{top}_block"] != "stacked_top":
                continue
            for bottom in ("red", "blue", "pink"):
                if top != bottom and current[f"{bottom}_block"] == "stacked_bottom":
                    results.append({**current, f"{top}_block": "table", f"{bottom}_block": "table"})
        return results
    raise StrictSchemaError(f"calvin sequence: unknown task {task!r}")


def _valid_sequence(initial: Mapping[str, int | str], tasks: Sequence[str]) -> bool:
    state = dict(initial)
    for task in tasks:
        next_states = _next_states(state, task)
        if len(next_states) != 1:
            return False
        state = next_states[0]
    categories = [TASK_CATEGORIES[task] for task in tasks]
    return len(categories) == len(set(categories))


def _initial_conditions() -> tuple[dict[str, int | str], ...]:
    choices = (
        (0, 1),
        (0, 1),
        ("right", "left"),
        ("closed", "open"),
        ("table", "slider_right", "slider_left"),
        ("table", "slider_right", "slider_left"),
        ("table", "slider_right", "slider_left"),
        (0,),
    )
    result = []
    for values in itertools.product(*choices):
        blocks = values[4:7]
        if blocks.count("table") not in {1, 2} or blocks.count("slider_right") >= 2 or blocks.count("slider_left") >= 2:
            continue
        result.append(dict(zip(INITIAL_KEYS, values)))
    return tuple(result)


def generate_official_sequences() -> tuple[CalvinSequence, ...]:
    initial = _initial_conditions()
    counts = [len(chunk) for chunk in np.array_split(range(OFFICIAL_SEQUENCE_COUNT), len(initial))]
    rows: list[tuple[dict[str, int | str], tuple[str, ...]]] = []
    for index, (condition, count) in enumerate(zip(initial, counts)):
        rng = np.random.RandomState(index)
        sequences = []
        while len(sequences) < count:
            tasks = tuple(str(task) for task in rng.choice(TASK_NAMES, size=OFFICIAL_SEQUENCE_LENGTH, replace=False))
            if _valid_sequence(condition, tasks):
                sequences.append(tasks)
        rows.extend((dict(condition), tasks) for tasks in sequences)
    np.random.RandomState(0).shuffle(rows)
    return tuple(
        CalvinSequence(f"official_sequence_{index:04d}", condition, tasks)
        for index, (condition, tasks) in enumerate(rows)
    )


def sequence_table_sha256(sequences: Sequence[CalvinSequence]) -> str:
    payload = [sequence.to_mapping() for sequence in sequences]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_sequence_manifest(path: str | Path) -> tuple[CalvinSequence, ...]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    required = {"schema_version", "benchmark", "source", "generator", "sequence_count", "sequence_length", "sequence_table_sha256", "sequences"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["benchmark"] != "CALVIN ABC to D official sequence identities"
    ):
        raise StrictSchemaError("calvin sequence manifest: invalid fields")
    if value["sequence_count"] != OFFICIAL_SEQUENCE_COUNT or value["sequence_length"] != OFFICIAL_SEQUENCE_LENGTH:
        raise StrictSchemaError("calvin sequence manifest: invalid dimensions")
    expected_source = {
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_COMMIT,
        "calvin_tacto_commit": CALVIN_TACTO_COMMIT,
        "xvla_commit": XVLA_SOURCE_COMMIT,
    }
    expected_generator = {
        "source_path": "calvin_models/calvin_agent/evaluation/multistep_sequences.py",
        "source_sha256": OFFICIAL_GENERATOR_SHA256,
        "reset_source_path": "calvin_models/calvin_agent/evaluation/utils.py",
        "reset_source_sha256": OFFICIAL_RESET_SOURCE_SHA256,
        "seed": 0,
    }
    if value["source"] != expected_source or value["generator"] != expected_generator:
        raise StrictSchemaError("calvin sequence manifest: source identity mismatch")
    if value["sequence_table_sha256"] != OFFICIAL_SEQUENCE_TABLE_SHA256:
        raise StrictSchemaError("calvin sequence manifest: unexpected table identity")
    rows = value["sequences"]
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != {"sequence_id", "initial_condition", "tasks"}
        for row in rows
    ):
        raise StrictSchemaError("calvin sequence manifest: invalid sequence row")
    sequences = tuple(
        CalvinSequence(row["sequence_id"], row["initial_condition"], tuple(row["tasks"]))
        for row in rows
    )
    expected_ids = tuple(f"official_sequence_{index:04d}" for index in range(OFFICIAL_SEQUENCE_COUNT))
    if tuple(row.sequence_id for row in sequences) != expected_ids:
        raise StrictSchemaError("calvin sequence manifest: noncanonical identities")
    if sequence_table_sha256(sequences) != value["sequence_table_sha256"]:
        raise StrictSchemaError("calvin sequence manifest: table hash mismatch")
    return sequences


def parse_project_scenario(value: str) -> int:
    match = re.fullmatch(r"official_sequence_(\d{4})_prefix_1", value)
    if match is None:
        raise StrictSchemaError("CALVIN scenario_id must be official_sequence_NNNN_prefix_1")
    index = int(match.group(1))
    if index >= OFFICIAL_SEQUENCE_COUNT:
        raise StrictSchemaError("CALVIN official sequence index is unavailable")
    return index


def _fnv1_32(value: str) -> int:
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = ((result * 16777619) & 0xFFFFFFFF) ^ byte
    return result


def official_reset_state(initial_condition: Mapping[str, int | str]) -> tuple[np.ndarray, np.ndarray]:
    condition = dict(initial_condition)
    if set(condition) != set(INITIAL_KEYS):
        raise StrictSchemaError("CALVIN initial condition has invalid keys")
    condition = {key: condition[key] for key in INITIAL_KEYS}
    robot = np.array(
        (
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ),
        dtype=np.float64,
    )
    rng = np.random.RandomState(_fnv1_32(str(condition.values())))
    block_table = [
        np.array((5.00000896e-02, -1.20000177e-01, 4.59990009e-01)),
        np.array((2.29995412e-01, -1.19995140e-01, 4.59990010e-01)),
    ]
    rng.shuffle(block_table)
    slider_left = np.array((-2.40851662e-01, 9.24044687e-02, 4.60990009e-01))
    slider_right = np.array((7.03416330e-02, 9.24044687e-02, 4.60990009e-01))
    scene = np.zeros(24, dtype=np.float64)
    if condition["slider"] == "left":
        scene[0] = 0.28
    if condition["drawer"] == "open":
        scene[1] = 0.22
    if condition["lightbulb"] == 1:
        scene[3] = 0.088
    scene[4] = condition["lightbulb"]
    scene[5] = condition["led"]
    locations = {"slider_right": slider_right, "slider_left": slider_left}
    scene[6:9] = locations.get(str(condition["red_block"]), block_table[0])
    scene[11] = rng.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    if condition["blue_block"] in locations:
        scene[12:15] = locations[str(condition["blue_block"])]
    elif condition["red_block"] == "table":
        scene[12:15] = block_table[1]
    else:
        scene[12:15] = block_table[0]
    scene[17] = rng.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    scene[18:21] = locations.get(str(condition["pink_block"]), block_table[1])
    scene[23] = rng.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    return robot, scene
