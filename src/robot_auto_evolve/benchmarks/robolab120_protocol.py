from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import fields, integer, sequence, string


ROBOLAB120_SOURCE_COMMIT = "92313e06dd90d2eeedf48da567755e0b1a8e32d1"
ROBOLAB120_METADATA_SHA256 = "ad7553707eed0d945e98ca4a5fdd8277eb31eecf5b2b64217a74debc47abb654"
ROBOLAB120_MODEL_ROUTE = "molmobot_robolab120"
ROBOLAB120_SUITE = "robolab120_droid_jointpos"
ROBOLAB120_BENCHMARK_ID = "molmobot_robolab120_vague_project_fixed_3_per_task_v1"
ROBOLAB120_BENCHMARK_PROTOCOL = "robolab120_vague_project_fixed_3_per_task_v1"
ROBOLAB120_INSTRUCTION_TYPE = "vague"
ROBOLAB120_TRIALS_PER_TASK = 3
ROBOLAB120_CONTROL_HZ = 15
ROBOLAB120_ENVIRONMENT_SEED = 1
ROBOLAB120_POLICY_SEED_BASE = 780_000


@dataclass(frozen=True)
class RoboLab120Task:
    task_id: str
    filename: str
    scene: str
    instruction_variants: dict[str, str]
    episode_seconds: int

    @property
    def horizon(self) -> int:
        return self.episode_seconds * ROBOLAB120_CONTROL_HZ


def _relative_task_path(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py" or path.parts[:1] != ("benchmark",):
        raise StrictSchemaError(f"{name}: expected benchmark-relative Python path")
    return path.as_posix()


def _instruction_variants(value: Any, name: str) -> dict[str, str]:
    obj = fields(value, {"default", "vague", "specific"}, path=name)
    return {key: string(obj[key], f"{name}.{key}") for key in ("default", "vague", "specific")}


def _source_class(path: Path, task_id: str) -> tuple[dict[str, str], int, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == task_id]
    if len(classes) != 1:
        raise StrictSchemaError(f"RoboLab task class differs: {task_id}")
    values: dict[str, Any] = {}
    scene = None
    for statement in classes[0].body:
        target = None
        value = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        if target.id in {"instruction", "episode_length_s"}:
            try:
                values[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise StrictSchemaError(f"RoboLab task literal differs: {task_id}.{target.id}") from exc
        if target.id == "scene" and isinstance(value, ast.Call) and value.args:
            argument = value.args[0]
            if isinstance(argument, ast.Constant) and type(argument.value) is str:
                scene = argument.value
    if set(values) != {"instruction", "episode_length_s"} or scene is None:
        raise StrictSchemaError(f"RoboLab task metadata is incomplete: {task_id}")
    instructions = _instruction_variants(values["instruction"], f"robolab_source.{task_id}.instruction")
    seconds = integer(values["episode_length_s"], f"robolab_source.{task_id}.episode_length_s", minimum=1)
    return instructions, seconds, scene


def load_robolab120_catalog(source_root: str | Path, manifest_path: str | Path) -> tuple[RoboLab120Task, ...]:
    source = Path(source_root).resolve()
    manifest_source = Path(manifest_path).resolve()
    metadata_path = source / "robolab" / "tasks" / "_metadata" / "task_metadata.json"
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != ROBOLAB120_METADATA_SHA256:
        raise StrictSchemaError("RoboLab task metadata hash differs")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StrictSchemaError(f"RoboLab catalog JSON is invalid: {exc}") from exc
    if type(metadata) is not list or len(metadata) != 120:
        raise StrictSchemaError("RoboLab metadata must contain 120 tasks")
    if manifest.get("source", {}).get("commit") != ROBOLAB120_SOURCE_COMMIT:
        raise StrictSchemaError("RoboLab benchmark manifest source differs")
    manifest_tasks = tuple(string(item, "robolab_manifest.tasks") for item in sequence(manifest.get("tasks"), "robolab_manifest.tasks"))
    if len(manifest_tasks) != 120 or len(set(manifest_tasks)) != 120:
        raise StrictSchemaError("RoboLab benchmark manifest task set differs")
    tasks = []
    for index, value in enumerate(metadata):
        if type(value) is not dict:
            raise StrictSchemaError(f"robolab_metadata[{index}]: expected object")
        required = {"task_name", "instruction_variants", "episode_s", "scene", "filename"}
        if not required <= set(value):
            raise StrictSchemaError(f"robolab_metadata[{index}]: required fields are absent")
        task_id = string(value["task_name"], f"robolab_metadata[{index}].task_name")
        filename = _relative_task_path(value["filename"], f"robolab_metadata[{index}].filename")
        instructions = _instruction_variants(
            value["instruction_variants"],
            f"robolab_metadata[{index}].instruction_variants",
        )
        try:
            seconds_value = int(string(value["episode_s"], f"robolab_metadata[{index}].episode_s"))
        except ValueError as exc:
            raise StrictSchemaError(f"robolab_metadata[{index}].episode_s: expected integer string") from exc
        seconds = integer(seconds_value, f"robolab_metadata[{index}].episode_s", minimum=1)
        scene = string(value["scene"], f"robolab_metadata[{index}].scene")
        source_instructions, source_seconds, source_scene = _source_class(
            source / "robolab" / "tasks" / filename,
            task_id,
        )
        if (source_instructions, source_seconds, source_scene) != (instructions, seconds, scene):
            raise StrictSchemaError(f"RoboLab task source differs from metadata: {task_id}")
        tasks.append(RoboLab120Task(task_id, filename, scene, instructions, seconds))
    if tuple(item.task_id for item in tasks) != manifest_tasks:
        raise StrictSchemaError("RoboLab metadata order differs from the benchmark manifest")
    if len({item.filename for item in tasks}) != 120:
        raise StrictSchemaError("RoboLab task source path is duplicated")
    return tuple(tasks)
