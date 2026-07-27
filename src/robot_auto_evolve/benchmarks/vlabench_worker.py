from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
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
from robot_auto_evolve.runtime_paths import project_root_from_package

from .depth3d import dm_control_camera_3d
from .smoke_horizon import smoke_horizon_override
from .vlabench_assets import read_and_validate_vlabench_asset_record
from .xvla import VLABENCH_ACTION_SPEC, VLABENCH_BASE_ENV_STEP_S, VLABENCH_TASKS


VLABENCH_SOURCE_COMMIT = "cf588fe60c0c7282174fe979f5913170cfe69017"
VLABENCH_PROTOCOL = "xvla_vlabench_project_transfer_v1"
VLABENCH_SMOKE_PROTOCOL = "xvla_vlabench_project_transfer_smoke_v1"
VLABENCH_BENCHMARK_PROTOCOL = "xvla_vlabench_official_four_track_10ep_v1"
VLABENCH_MAX_SUBSTEPS = 10
VLABENCH_TOLERANCE = 1e-2
TRACK_FILES = {
    "track_1": ("track_1_in_distribution.json", "61b07440c6f4272705bea0c8d721bc196e744232f48408075613cc9b6f7c8c04"),
    "track_2": ("track_2_cross_category.json", "cf004383de22d994f83126ca30a315cbfc6a92fd0dbb8d25fa4c185dd75a21fb"),
    "track_3": ("track_3_common_sense.json", "2aaa298580e74bec13555a6950b95241edf6c5264b498eeaf0e919837952ec6d"),
    "track_4": ("track_4_semantic_instruction.json", "9aff9dfc778ae35f299dce4c3aa07afbdc5275847490abd8037f3f5bdd9de411"),
}
TASK_HORIZONS = {
    "select_poker": 100,
    "select_nth_largest_poker": 100,
    **{task: 200 for task in VLABENCH_TASKS - {"select_poker", "select_nth_largest_poker"}},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_vlabench_scenario(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"(track_[1-4])_config_(\d{3})", value)
    if match is None:
        raise StrictSchemaError("VLABench scenario_id must be track_N_config_NNN")
    index = int(match.group(2))
    if index >= 50:
        raise StrictSchemaError("VLABench scenario config index must be below 50")
    return match.group(1), index


def _validated_source() -> tuple[Path, Path]:
    source_value = os.environ.get("ROBOT_AE_VLABENCH_SOURCE")
    manifest_value = os.environ.get("ROBOT_AE_VLABENCH_ASSET_MANIFEST")
    if not source_value or not manifest_value:
        raise RuntimeError("VLABench worker requires source and asset manifest paths")
    source = Path(source_value).resolve()
    package = source / "VLABench"
    manifest = Path(manifest_value).resolve()
    expected_manifest = project_root_from_package() / "manifests" / "vlabench_assets.json"
    if manifest != expected_manifest:
        raise RuntimeError("VLABench asset manifest path differs from the project lock")
    if not (source / ".git").is_dir() or not (package / "envs" / "dm_env.py").is_file():
        raise RuntimeError("VLABench source checkout is incomplete")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != VLABENCH_SOURCE_COMMIT:
        raise RuntimeError(f"VLABench source revision mismatch: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"VLABench source working tree is dirty: {dirty.splitlines()[0]}")
    tracks = package / "configs" / "evaluation" / "tracks"
    for filename, digest in TRACK_FILES.values():
        path = tracks / filename
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"VLABench evaluation track differs: {filename}")
    if not manifest.is_file():
        raise RuntimeError("VLABench asset manifest is absent")
    read_and_validate_vlabench_asset_record(assets=package / "assets", manifest=manifest)
    for relative in ("assets/obj", "assets/scenes", "configs/task_config.json", "configs/robot_config.json"):
        if not (package / relative).exists():
            raise RuntimeError(f"VLABench required path is absent: {relative}")
    return source, package


def _episode_config(package: Path, task_id: str, track: str, index: int) -> dict[str, Any]:
    filename, _ = TRACK_FILES[track]
    table = json.loads((package / "configs" / "evaluation" / "tracks" / filename).read_text(encoding="utf-8"))
    if task_id not in table or index >= len(table[task_id]):
        raise StrictSchemaError("VLABench task is absent from its pinned track")
    return table[task_id][index]


class VLABenchWorker:
    ACTION_SPEC = VLABENCH_ACTION_SPEC

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("VLABench worker requires Profile and EpisodeKey")
        if profile.environment.suite != "vlabench_xvla_tracks_1_4":
            raise StrictSchemaError("VLABench worker requires the X-VLA four-track suite")
        if profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("VLABench worker action spec differs from profile")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("VLABench worker requires one target per policy response")
        if episode.task_id not in VLABENCH_TASKS:
            raise StrictSchemaError("VLABench worker received an unknown task")
        track, config_index = parse_vlabench_scenario(episode.scenario_id)
        expected_horizon = 1 if episode.protocol == VLABENCH_SMOKE_PROTOCOL else TASK_HORIZONS[episode.task_id]
        if episode.protocol not in {VLABENCH_PROTOCOL, VLABENCH_SMOKE_PROTOCOL, VLABENCH_BENCHMARK_PROTOCOL}:
            raise StrictSchemaError("VLABench episode protocol differs")
        if (
            smoke_horizon_override() is None and episode.horizon != expected_horizon
        ) or episode.environment_seed != config_index:
            raise StrictSchemaError("VLABench episode horizon or deterministic config seed differs")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._track = track
        self._config_index = config_index
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._instruction = ""
        self._step = 0
        self._base_steps = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("VLABench worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("VLABench worker requires PYTHONNOUSERSITE=1")
        for name, expected in {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(self._render_gpu_id)}.items():
            actual = os.environ.get(name)
            if actual not in {None, expected}:
                raise RuntimeError(f"{name} differs from the worker render assignment")
            os.environ[name] = expected
        source, package = _validated_source()
        os.environ["VLABENCH_ROOT"] = str(package)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        tasks = importlib.import_module("VLABench.tasks")
        robots = importlib.import_module("VLABench.robots")
        envs = importlib.import_module("VLABench.envs")
        imported = {
            Path(tasks.__file__).resolve(): package / "tasks" / "__init__.py",
            Path(robots.__file__).resolve(): package / "robots" / "__init__.py",
            Path(envs.__file__).resolve(): package / "envs" / "__init__.py",
        }
        if any(actual != expected for actual, expected in imported.items()):
            raise RuntimeError("imported VLABench package differs from the pinned source")
        np.random.seed(self._episode.environment_seed)
        random.seed(self._episode.environment_seed)
        config = _episode_config(package, self._episode.task_id, self._track, self._config_index)
        self._env = envs.load_env(
            self._episode.task_id,
            episode_config=config,
            random_init=False,
            eval=False,
            run_mode="eval",
        )
        self._env.reset()
        self._instruction = self._env.task.get_instruction()
        if not isinstance(self._instruction, str) or not self._instruction.strip():
            raise RuntimeError("VLABench episode instruction is absent")
        self._step = 0
        self._base_steps = 0
        self._success = False

    def _raw_observation(self) -> dict[str, Any]:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        return self._env.get_observation(require_pcd=False)

    def observe(self) -> FairObservation:
        raw = self._raw_observation()
        images = np.asarray(raw["rgb"], dtype=np.uint8)
        if images.ndim != 4 or images.shape[0] < 4 or images.shape[1:] != (480, 480, 3):
            raise RuntimeError(f"VLABench multiview RGB has invalid shape {images.shape}")
        # The same view indices are used for rgb, depth, intrinsic and extrinsic, because
        # VLABench builds all four arrays in one pass over the same camera list
        # (VLABench/envs/dm_env.py, get_observation).
        view_index = {"main": 0, "front": 2, "wrist": images.shape[0] - 1}
        selected = {name: images[index] for name, index in view_index.items()}
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        if set(camera_specs) != set(selected):
            raise StrictSchemaError("VLABench profile cameras differ from the released X-VLA client")
        cameras = {}
        for name, image in selected.items():
            spec = camera_specs[name]
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                # Revision 2. Nothing extra is rendered: VLABench's own get_observation already
                # renders depth and computes the two camera matrices for every camera on every
                # call, and they were simply being dropped here. The arm is reported in the world
                # frame on this route, so no re-expression is needed.
                depth_m, depth_valid, intrinsics, camera_to_world = dm_control_camera_3d(
                    np.asarray(raw["depth"])[view_index[name]],
                    np.asarray(raw["instrinsic"])[view_index[name]],
                    np.asarray(raw["extrinsic"])[view_index[name]],
                    height=spec.height,
                    width=spec.width,
                )
            cameras[name] = CameraObservation(
                frame_id=spec.frame_id,
                optical_convention=spec.optical_convention,
                rgb=np.ascontiguousarray(image),
                depth_m=depth_m,
                depth_valid=depth_valid,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
        ee_state = np.asarray(raw["ee_state"], dtype=np.float32)
        if ee_state.shape != (8,) or not np.isfinite(ee_state).all():
            raise RuntimeError(f"VLABench end-effector state has invalid shape {ee_state.shape}")
        state_values = {"eef_pose": ee_state[:7], "gripper_position": ee_state[7:]}
        vectors = tuple(
            RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name]))
            for spec in self._profile.environment.robot_state
        )
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=round(self._base_steps * VLABENCH_BASE_ENV_STEP_S * 1_000_000_000),
            instruction=self._instruction,
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("VLABench action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("VLABench worker requires one target at the current step")
        native = np.asarray(action.executable_values()[0], dtype=np.float64)
        if native.shape != (8,) or not np.isfinite(native).all():
            raise StrictSchemaError("VLABench native action must be finite width 8")
        if not np.isclose(native[6], native[7]) or not any(np.isclose(native[6], value) for value in (0.0, 0.04)):
            raise StrictSchemaError("VLABench finger targets must be equal and either 0 or 0.04 meter")
        utils = importlib.import_module("VLABench.utils.utils")
        quaternion = utils.euler_to_quaternion(*native[3:6])
        _, qpos = self._env.robot.get_qpos_from_ee_pos(
            physics=self._env.physics,
            pos=native[:3],
            quat=quaternion,
        )
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise RuntimeError("VLABench inverse kinematics returned an invalid target")
        command = np.concatenate((qpos, native[6:8]))
        for _ in range(VLABENCH_MAX_SUBSTEPS):
            timestep = self._env.step(command)
            self._base_steps += 1
            if timestep.last():
                self._success = True
                break
            current = np.asarray(self._env.task.robot.get_qpos(self._env.physics)).reshape(-1)
            difference = current - command[:7]
            if np.max(difference) < VLABENCH_TOLERANCE and np.min(difference) > -VLABENCH_TOLERANCE:
                break
        self._step += 1

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        return self._success

    def private_metrics(self) -> dict[str, float | bool]:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        return {
            "success": self._success,
            "intention_score": float(self._env.get_intention_score(threshold=0.1)),
            "progress_score": float(self._env.get_task_progress()),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        environment = self._env
        self._env = None
        if environment is not None:
            environment.close()
