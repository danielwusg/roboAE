from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import (
    CameraObservation,
    CanonicalActionChunk,
    FairObservation,
    RobotProprioception,
    RobotStateVector,
    StrictSchemaError,
)
from robot_auto_evolve.provenance import EpisodeKey

from .robocerebra import (
    CONDITIONS,
    FULL_STACK_SMOKE_HORIZON,
    FULL_STACK_SMOKE_PROTOCOL,
    PUBLIC_PROTOCOL,
    SEGMENT_STEPS,
    SETTLE_STEPS,
    SMOLVLA_ROBOCEREBRA_ACTION_SPEC,
    RoboCerebraCase,
    load_case_catalog,
    parse_task_id,
)
from .depth3d import robosuite_camera_3d
from .smoke_horizon import smoke_horizon_override


SOURCE_COMMIT = "2573426c13dfcd5e7d7831c15587b058aaa1c0c0"
MOVABLE_OBJECTS = (
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cookies",
    "cream_cheese",
    "ketchup",
    "macaroni_and_cheese",
    "milk",
    "orange_juice",
    "popcorn",
    "salad_dressing",
    "new_salad_dressing",
    "tomato_sauce",
    "white_bowl",
    "akita_black_bowl",
    "plate",
    "glazed_rim_porcelain_ramekin",
    "red_coffee_mug",
    "porcelain_mug",
    "white_yellow_mug",
    "chefmate_8_frypan",
    "bowl_drainer",
    "moka_pot",
    "window",
    "faucet",
    "black_book",
    "yellow_book",
    "desk_caddy",
    "wine_bottle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source() -> tuple[Path, Path, Path, dict[str, Any]]:
    source_value = os.environ.get("ROBOT_AE_ROBOCEREBRA_SOURCE")
    asset_value = os.environ.get("ROBOT_AE_ROBOCEREBRA_ASSETS")
    manifest_value = os.environ.get("ROBOT_AE_ROBOCEREBRA_ASSET_MANIFEST")
    lock_value = os.environ.get("ROBOT_AE_ROBOCEREBRA_ASSET_LOCK")
    catalog_value = os.environ.get("ROBOT_AE_ROBOCEREBRA_CASE_CATALOG")
    if not all((source_value, asset_value, manifest_value, lock_value, catalog_value)):
        raise RuntimeError("RoboCerebra worker runtime paths are incomplete")
    source = Path(source_value).resolve()
    package = source / "LIBERO" / "libero" / "libero"
    if not (source / ".git").is_dir() or not (package / "envs" / "bddl_base_domain.py").is_file():
        raise RuntimeError("RoboCerebra source checkout is incomplete")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if head != SOURCE_COMMIT or dirty:
        raise RuntimeError("RoboCerebra source revision or cleanliness differs")
    assets = Path(asset_value).resolve()
    manifest_path = Path(manifest_value).resolve()
    lock_path = Path(lock_value).resolve()
    catalog_path = Path(catalog_value).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("kind") != "robocerebra_asset_verification"
        or Path(lock.get("asset_root", "")).resolve() != assets
        or lock.get("manifest", {}).get("sha256") != _sha256(manifest_path)
        or lock.get("file_count") != 538
        or lock.get("logical_size_bytes") != 355_198_111
    ):
        raise RuntimeError("RoboCerebra asset lock differs")
    return source, assets, catalog_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _asset(assets: Path, manifest: dict[str, Any], relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError("RoboCerebra asset path is unsafe")
    path = assets.joinpath(*value.parts)
    record = next((item for item in manifest["files"] if item["path"] == relative), None)
    if record is None or not path.is_file() or path.stat().st_size != record["size_bytes"]:
        raise RuntimeError(f"RoboCerebra runtime asset differs: {relative}")
    return path


def _parse_description(path: Path) -> tuple[list[str], list[int], str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    instruction = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Task:")), "")
    steps = []
    starts = []
    for index, line in enumerate(lines):
        if not line.startswith("Step:"):
            continue
        steps.append(line.split(":", 1)[1].strip())
        if index + 1 >= len(lines):
            raise RuntimeError("RoboCerebra task description lacks a state interval")
        match = re.fullmatch(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", lines[index + 1])
        if match is None:
            raise RuntimeError("RoboCerebra task description interval differs")
        starts.append(int(match.group(1)))
    if not instruction or not steps or len(steps) != len(starts):
        raise RuntimeError("RoboCerebra task description differs")
    return steps, starts, instruction


def _load_goal(path: Path) -> tuple[dict[str, list[list[str]]], dict[str, list[int]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    goals: dict[str, list[list[str]]] = {}
    goal_steps: dict[str, list[int]] = {}
    for object_id, relations in value.items():
        actions = []
        steps = []
        for item in relations:
            if isinstance(item, dict) and set(item) >= {"state_pair", "task_step"}:
                action = item["state_pair"]
                step = item["task_step"]
            elif isinstance(item, list):
                action = item
                step = len(actions)
            else:
                continue
            if len(action) in {2, 3}:
                actions.append([str(action[0]).lower(), *(str(value) for value in action[1:])])
                steps.append(int(step))
        goals[str(object_id)] = actions
        goal_steps[str(object_id)] = steps
    if not goals or not any(goals.values()):
        raise RuntimeError("RoboCerebra goal metadata differs")
    return goals, goal_steps


class RoboCerebraWorker:
    ACTION_SPEC = SMOLVLA_ROBOCEREBRA_ACTION_SPEC

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("RoboCerebra worker requires Profile and EpisodeKey")
        if profile.environment.suite != "robocerebra_public60" or profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("RoboCerebra profile differs")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("RoboCerebra worker requires one-action policy responses")
        condition, case_id = parse_task_id(episode.task_id)
        if episode.protocol not in {PUBLIC_PROTOCOL, FULL_STACK_SMOKE_PROTOCOL} or type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("RoboCerebra episode protocol or render GPU differs")
        self._profile = profile
        self._episode = episode
        self._condition = condition
        self._case_id = case_id
        self._render_gpu_id = render_gpu_id
        self._wants_depth = any(item.has_depth for item in profile.environment.cameras)
        self._case: RoboCerebraCase | None = None
        self._env: Any = None
        self._raw: dict[str, Any] | None = None
        self._goal: dict[str, list[list[str]]] = {}
        self._goal_steps: dict[str, list[int]] = {}
        self._step_states: list[np.ndarray] = []
        self._steps: list[str] = []
        self._segment = 0
        self._policy_step = 0
        self._sim_step = 0
        self._completed_previous = 0
        self._agent_subtasks = 0
        self._initial_excluded = 0
        self._resume_completed = 0
        self._skip_increment = False
        self._success = False
        self._dynamic_applied = False
        self._dynamic_due: int | None = None
        self._dynamic_info: dict[str, Any] | None = None
        self._toggle_direction = -1
        self._shifted_initial = condition in {"Mix", "Observation_Mismatching"}
        self._rng = np.random.default_rng(episode.environment_seed)
        self._closed = False

    def _set_anchor(self, index: int) -> None:
        self._env.sim.set_state_from_flattened(self._step_states[index])
        self._env.sim.forward()
        self._env._post_process()
        self._env._update_observables(force=True)
        self._raw = self._env._get_observations()

    def _check(self) -> tuple[int, bool]:
        _, completed, success = self._env._check_success(self._goal)
        return int(completed), bool(success)

    def _mark_prior(self, current_step: int) -> int:
        if not hasattr(self._env, "_state_progress"):
            return 0
        count = 0
        for object_id, actions in self._goal.items():
            for action_index, step in enumerate(self._goal_steps.get(object_id, ())):
                if step >= current_step:
                    continue
                if object_id in self._env._state_progress and self._env._state_progress[object_id] <= action_index:
                    self._env._state_progress[object_id] = action_index + 1
                    count += 1
        self._resume_completed += count
        return count

    def _find_y_address(self, object_name: str) -> int | None:
        for joint_name in (
            f"{object_name}_1_joint0",
            f"{object_name}_joint0",
            f"{object_name}_joint",
        ):
            if joint_name in self._env.sim.model.joint_names:
                return int(self._env.sim.model.get_joint_qpos_addr(joint_name)[0]) + 1
        return None

    def _prepare_dynamic(self, description_json: Path) -> None:
        if self._condition not in {"Mix", "Random_Disturbance"}:
            return
        value = json.loads(description_json.read_text(encoding="utf-8"))
        step_objects = {
            str(item.get("step", "")).removeprefix("Step: "): str(item.get("object", ""))
            for item in value
            if isinstance(item, dict)
        }
        related = []
        related_names = []
        for description in self._steps:
            object_name = step_objects.get(description, "")
            related_names.append(object_name)
            address = self._find_y_address(object_name) if object_name in MOVABLE_OBJECTS else None
            related.append(None if address is None else (address, float(self._env.sim.data.qpos[address])))
        unrelated = []
        for object_name in MOVABLE_OBJECTS:
            if object_name in related_names:
                continue
            address = self._find_y_address(object_name)
            if address is not None:
                unrelated.append((address, float(self._env.sim.data.qpos[address])))
        if any(item is not None for item in related) and unrelated:
            self._dynamic_info = {"related": related, "unrelated": unrelated}

    def _move_dynamic(self) -> None:
        if self._dynamic_info is None:
            self._dynamic_due = None
            return
        related = self._dynamic_info["related"][self._segment]
        if self._rng.random() < 0.5 and related is not None:
            address, base = related
        else:
            choices = self._dynamic_info["unrelated"]
            address, base = choices[int(self._rng.integers(0, len(choices)))]
        self._env.sim.data.qpos[address] = base + 0.15 * self._toggle_direction
        self._toggle_direction *= -1
        self._env.sim.forward()
        self._env._post_process()
        self._env._update_observables(force=True)
        self._raw = self._env._get_observations()
        self._dynamic_applied = True
        self._dynamic_due = None

    def _transition(self, new_segment: int) -> None:
        self._segment = new_segment
        self._set_anchor(new_segment)
        self._env.skip_pick_quat_once = True
        self._mark_prior(new_segment)
        self._skip_increment = True
        if self._condition in {"Mix", "Random_Disturbance"}:
            self._dynamic_due = self._sim_step + 10

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("RoboCerebra worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("RoboCerebra worker requires PYTHONNOUSERSITE=1")
        source, assets, catalog_path, manifest = _validate_source()
        catalog = load_case_catalog(catalog_path)
        self._case = next(item for item in catalog if item.task_id == self._episode.task_id)
        expected_horizon = self._case.horizon if self._episode.protocol == PUBLIC_PROTOCOL else FULL_STACK_SMOKE_HORIZON
        if (
            smoke_horizon_override() is None and self._episode.horizon != expected_horizon
        ) or self._condition not in CONDITIONS:
            raise StrictSchemaError("RoboCerebra episode horizon differs")
        for name, expected in {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(self._render_gpu_id)}.items():
            actual = os.environ.get(name)
            if actual is not None and actual != expected:
                raise RuntimeError(f"{name} differs from the worker render assignment")
            os.environ[name] = expected
        package = source / "LIBERO" / "libero" / "libero"
        import libero.libero as imported

        if Path(imported.__file__).resolve() != package / "__init__.py":
            raise RuntimeError("imported LIBERO package differs from RoboCerebra source")
        import h5py
        import libero.libero.envs.bddl_utils as bddl_utils
        from libero.libero.envs import TASK_MAPPING
        from robosuite import load_controller_config

        bddl = _asset(assets, manifest, self._case.bddl_path)
        demo = _asset(assets, manifest, self._case.demo_path)
        goal_path = _asset(assets, manifest, self._case.goal_path)
        description = _asset(assets, manifest, self._case.description_path)
        description_json = _asset(assets, manifest, self._case.description_json_path)
        info = bddl_utils.get_problem_info(str(bddl))
        controller = load_controller_config(default_controller="OSC_POSE")
        self._env = TASK_MAPPING[info["problem_name"]](
            bddl_file_name=str(bddl),
            robots=["Panda"],
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            camera_heights=256,
            camera_widths=256,
            camera_depths=self._wants_depth,
            control_freq=20,
        )
        with h5py.File(demo, "r") as handle:
            states = np.asarray(handle["data"]["demo_1"]["states"][()])
        self._steps, starts, _ = _parse_description(description)
        if tuple(self._steps) != self._case.steps or len(starts) != self._case.num_steps:
            raise RuntimeError("RoboCerebra catalog and downloaded description differ")
        if any(index >= len(states) for index in starts):
            raise RuntimeError("RoboCerebra demonstration anchor index differs")
        self._step_states = [np.asarray(states[index]) for index in starts]
        self._goal, self._goal_steps = _load_goal(goal_path)
        self._prepare_dynamic(description_json)
        self._env.seed(self._episode.environment_seed)
        self._raw = self._env.reset()
        for robot in self._env.robots:
            robot.controller.use_delta = True
        initial_anchor = 1 if self._shifted_initial else 0
        if initial_anchor >= len(self._step_states):
            raise RuntimeError("RoboCerebra shifted initial anchor is unavailable")
        self._set_anchor(initial_anchor)
        self._env.skip_pick_quat_once = True
        if initial_anchor:
            self._check()
            for object_id in getattr(self._env, "_state_progress", {}):
                self._env._state_progress[object_id] = 0
            self._initial_excluded = self._mark_prior(initial_anchor)
            self._skip_increment = True
        settle = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0), dtype=np.float32)
        for _ in range(SETTLE_STEPS):
            self._raw, _, _, _ = self._env.step(settle)
        self._sim_step = SETTLE_STEPS
        self._completed_previous, self._success = self._check()

    def observe(self) -> FairObservation:
        if self._env is None or self._raw is None or self._closed or self._case is None:
            raise RuntimeError("RoboCerebra worker is not active")
        specs = {item.name: item for item in self._profile.environment.cameras}
        if set(specs) != {"main", "wrist"}:
            raise StrictSchemaError("RoboCerebra camera profile differs")
        images = {
            "main": np.ascontiguousarray(self._raw["agentview_image"], dtype=np.uint8),
            "wrist": np.ascontiguousarray(self._raw["robot0_eye_in_hand_image"], dtype=np.uint8),
        }
        cameras = {}
        for name, image in images.items():
            spec = specs[name]
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                depth_m, depth_valid, intrinsics, camera_to_world = robosuite_camera_3d(
                    self._env.sim,
                    spec.frame_id,
                    self._raw[f"{spec.frame_id}_depth"],
                    height=spec.height,
                    width=spec.width,
                )
            cameras[name] = CameraObservation(
                frame_id=spec.frame_id,
                optical_convention=spec.optical_convention,
                rgb=image,
                depth_m=depth_m,
                depth_valid=depth_valid,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
        values = {
            "eef_pose": np.concatenate(
                (
                    np.asarray(self._raw["robot0_eef_pos"], dtype=np.float32),
                    np.asarray(self._raw["robot0_eef_quat"], dtype=np.float32),
                )
            ),
            "gripper_position": np.asarray(self._raw["robot0_gripper_qpos"], dtype=np.float32),
        }
        vectors = tuple(RobotStateVector(spec, np.ascontiguousarray(values[spec.name])) for spec in self._profile.environment.robot_state)
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._policy_step,
            timestamp_ns=self._policy_step * 50_000_000,
            instruction=self._steps[self._segment],
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCerebra worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("RoboCerebra action spec differs")
        if action.horizon != 1 or action.execution_count != 1 or action.start_step != self._policy_step:
            raise StrictSchemaError("RoboCerebra worker requires one current action")
        native = action.executable_values()[0]
        if native.shape != (7,):
            raise StrictSchemaError("RoboCerebra native action width differs")
        self._raw, _, _, _ = self._env.step(native)
        self._policy_step += 1
        self._sim_step += 1
        completed, self._success = self._check()
        difference = completed - self._completed_previous
        if difference > 0 and not self._skip_increment:
            self._agent_subtasks += difference
        self._completed_previous = completed
        self._skip_increment = False
        new_segment = self._sim_step // SEGMENT_STEPS
        if self._sim_step % SEGMENT_STEPS == 0 and new_segment < len(self._steps):
            if self._shifted_initial and new_segment == 1:
                self._segment = new_segment
            else:
                self._transition(new_segment)
        if self._dynamic_due is not None and self._sim_step == self._dynamic_due:
            self._move_dynamic()

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCerebra worker is not active")
        return self._success

    def private_metrics(self) -> dict[str, bool | float]:
        total = sum(len(actions) for actions in self._goal.values())
        possible = max(total - self._initial_excluded, 0)
        rate = self._agent_subtasks / possible if possible else 0.0
        return {
            "success": self._success,
            "subtask_completion": float(rate),
            "agent_subtasks": float(self._agent_subtasks),
            "possible_subtasks": float(possible),
            "resume_completed": float(self._resume_completed),
            "dynamic_applied": self._dynamic_applied,
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "protocol": PUBLIC_PROTOCOL,
            "condition": self._condition,
            "case_id": self._case_id,
            "settle_steps": SETTLE_STEPS,
            "segment_steps": SEGMENT_STEPS,
            "dynamic_seed": self._episode.environment_seed,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._raw = None
