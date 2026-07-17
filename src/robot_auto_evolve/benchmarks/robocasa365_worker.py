from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
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

from .robocasa365 import (
    RLDX_ROBOCASA365_ACTION_SPEC,
    TARGET_TASK_HORIZONS,
    native_action_dict,
    validate_robocasa365_rgb,
)


RLDX_SOURCE_COMMIT = "ebbfb4f6214bb38de07da1a70f597201feceb6da"
ROBOCASA365_SOURCE_COMMIT = "32b481a8986a397b4b68911195e14e379b7434a4"
ROBOSUITE_SOURCE_COMMIT = "783b350fd61a257058719007ad4f2d7e1135e13f"
PROTOCOL = "rldx_robocasa365_target_related_transfer_v1"
PUBLIC_BENCHMARK_PROTOCOL = "rldx_robocasa365_target50_public_50_per_task_v1"
PUBLIC_BASE_SEED = 42
PUBLIC_ENVIRONMENTS = 5
PUBLIC_EPISODES_PER_ENVIRONMENT = 10
PUBLIC_SEED_STRIDE = 100_000
SMOKE_PROFILE_ID = "rldx_robocasa365_end_to_end_smoke"
SMOKE_PROTOCOL = "rldx_robocasa365_target_smoke_v1"
SMOKE_HORIZON = 9
SMOKE_TASKS = {
    "evolve": frozenset({"OpenCabinet", "OpenDrawer", "PickPlaceCounterToCabinet", "TurnOnMicrowave"}),
    "selection": frozenset({"OpenCabinet", "OpenDrawer", "PickPlaceCounterToCabinet", "TurnOnMicrowave"}),
    "transfer": frozenset({"CloseFridge", "SlideDishwasherRack", "PickPlaceCounterToStove", "TurnOnElectricKettle"}),
}


def public_episode_coordinates(episode: EpisodeKey) -> tuple[int, int, int]:
    match = re.fullmatch(r"official_env_(\d)_episode_(\d{2})", episode.scenario_id)
    if match is None:
        raise StrictSchemaError("RoboCasa365 public benchmark scenario id differs")
    environment_index = int(match.group(1))
    episode_index = int(match.group(2))
    if environment_index >= PUBLIC_ENVIRONMENTS or episode_index >= PUBLIC_EPISODES_PER_ENVIRONMENT:
        raise StrictSchemaError("RoboCasa365 public benchmark scenario index differs")
    constructor_seed = PUBLIC_BASE_SEED + environment_index
    reset_seed = constructor_seed * PUBLIC_SEED_STRIDE + episode_index - 1
    if episode.environment_seed != reset_seed:
        raise StrictSchemaError("RoboCasa365 public benchmark environment seed differs")
    return environment_index, constructor_seed, reset_seed


def _git_source(variable: str, commit: str, marker: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"RoboCasa365 worker requires {variable}")
    source = Path(value).resolve()
    if not (source / marker).is_file() or not (source / ".git").exists():
        raise RuntimeError(f"{variable} source checkout is incomplete")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise RuntimeError(f"{variable} source revision mismatch: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"{variable} source working tree is dirty: {dirty.splitlines()[0]}")
    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_lock() -> tuple[Path, dict[str, Any]]:
    value = os.environ.get("ROBOT_AE_ROBOCASA_ASSET_LOCK")
    if not value:
        raise RuntimeError("RoboCasa365 worker requires ROBOT_AE_ROBOCASA_ASSET_LOCK")
    path = Path(value).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "source_commit", "archives", "files", "tree_sha256"}:
        raise RuntimeError("RoboCasa365 asset lock schema mismatch")
    if payload["schema_version"] != 1 or payload["source_commit"] != ROBOCASA365_SOURCE_COMMIT:
        raise RuntimeError("RoboCasa365 asset lock identity mismatch")
    if not isinstance(payload["archives"], list) or len(payload["archives"]) != 6:
        raise RuntimeError("RoboCasa365 asset archive lock is incomplete")
    if not isinstance(payload["files"], list) or not payload["files"]:
        raise RuntimeError("RoboCasa365 asset file lock is empty")
    return path, payload


def _validate_installed_sources(robocasa_source: Path, robosuite_source: Path) -> Path:
    import mujoco
    import robocasa
    import robosuite

    if mujoco.__version__ != "3.3.1" or robocasa.__version__ != "1.0.0" or robosuite.__version__ != "1.5.2":
        raise RuntimeError("RoboCasa365 installed package version mismatch")
    installed_robocasa = Path(robocasa.__file__).resolve().parent
    installed_robosuite = Path(robosuite.__file__).resolve().parent
    comparisons = (
        (installed_robocasa / "wrappers" / "gym_wrapper.py", robocasa_source / "robocasa" / "wrappers" / "gym_wrapper.py"),
        (installed_robocasa / "utils" / "env_utils.py", robocasa_source / "robocasa" / "utils" / "env_utils.py"),
        (
            installed_robosuite / "controllers" / "config" / "robots" / "default_pandaomron.json",
            robosuite_source / "robosuite" / "controllers" / "config" / "robots" / "default_pandaomron.json",
        ),
        (
            installed_robosuite / "controllers" / "composite" / "composite_controller.py",
            robosuite_source / "robosuite" / "controllers" / "composite" / "composite_controller.py",
        ),
    )
    if any(not installed.is_file() or _sha256(installed) != _sha256(source) for installed, source in comparisons):
        raise RuntimeError("RoboCasa365 installed source differs from pinned checkouts")
    return installed_robocasa


class RoboCasa365Worker:
    ACTION_SPEC = RLDX_ROBOCASA365_ACTION_SPEC

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("RoboCasa365 worker requires Profile and EpisodeKey")
        if profile.environment.suite != "robocasa365_target":
            raise StrictSchemaError("RoboCasa365 worker requires the target split")
        if profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("RoboCasa365 worker action spec differs from profile")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("RoboCasa365 worker requires one-action policy responses")
        if episode.task_id not in TARGET_TASK_HORIZONS:
            raise StrictSchemaError("RoboCasa365 worker received an unknown target task")
        if episode.split == "benchmark":
            public_episode_coordinates(episode)
            valid_episode = (
                episode.horizon == TARGET_TASK_HORIZONS[episode.task_id]
                and episode.protocol == PUBLIC_BENCHMARK_PROTOCOL
            )
        elif profile.profile_id == SMOKE_PROFILE_ID:
            valid_episode = (
                episode.task_id in SMOKE_TASKS[episode.split]
                and episode.horizon == SMOKE_HORIZON
                and episode.protocol == SMOKE_PROTOCOL
            )
        else:
            valid_episode = (
                episode.horizon == TARGET_TASK_HORIZONS[episode.task_id]
                and episode.protocol == PROTOCOL
            )
        if not valid_episode:
            raise StrictSchemaError("RoboCasa365 episode protocol or horizon mismatch")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._step = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("RoboCasa365 worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("RoboCasa365 worker requires PYTHONNOUSERSITE=1")
        robocasa_source = _git_source(
            "ROBOT_AE_ROBOCASA_SOURCE",
            ROBOCASA365_SOURCE_COMMIT,
            "robocasa/wrappers/gym_wrapper.py",
        )
        robosuite_source = _git_source(
            "ROBOT_AE_ROBOSUITE_SOURCE",
            ROBOSUITE_SOURCE_COMMIT,
            "robosuite/controllers/config/robots/default_pandaomron.json",
        )
        _, asset_lock = _asset_lock()
        for name, expected in {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(self._render_gpu_id),
        }.items():
            actual = os.environ.get(name)
            if actual is not None and actual != expected:
                raise RuntimeError(f"{name} differs from the worker render assignment")
            os.environ[name] = expected
        installed = _validate_installed_sources(robocasa_source, robosuite_source)
        marker = installed / "models" / "assets" / ".robot_auto_evolve_assets.json"
        if not marker.is_file():
            raise RuntimeError("RoboCasa365 installed asset seal is absent")
        seal = json.loads(marker.read_text(encoding="utf-8"))
        if seal != {"tree_sha256": asset_lock["tree_sha256"], "file_count": len(asset_lock["files"])}:
            raise RuntimeError("RoboCasa365 installed asset seal differs from the lock")
        import gymnasium as gym
        __import__("robocasa")

        constructor_seed = self._episode.environment_seed
        reset_seed = self._episode.environment_seed
        if self._episode.split == "benchmark":
            _, constructor_seed, reset_seed = public_episode_coordinates(self._episode)
        random.seed(constructor_seed)
        np.random.seed(constructor_seed)
        self._env = gym.make(
            f"robocasa/{self._episode.task_id}",
            split="target",
            seed=constructor_seed,
        )
        if self._episode.split == "benchmark":
            random.seed(reset_seed)
            np.random.seed(reset_seed)
            unwrapped = getattr(self._env, "unwrapped", None)
            robosuite_env = getattr(unwrapped, "env", unwrapped)
            if robosuite_env is not None:
                if hasattr(robosuite_env, "seed"):
                    robosuite_env.seed = reset_seed
                if hasattr(robosuite_env, "rng"):
                    robosuite_env.rng = np.random.default_rng(reset_seed)
        self._observation, _ = self._env.reset(seed=reset_seed)
        self._step = 0
        self._success = False

    def observe(self) -> FairObservation:
        if self._env is None or self._observation is None or self._closed:
            raise RuntimeError("RoboCasa365 worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        cameras = {}
        for name in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
            if name not in camera_specs:
                raise StrictSchemaError(f"RoboCasa365 profile camera {name!r} is absent")
            spec = camera_specs[name]
            image = validate_robocasa365_rgb(raw[f"video.{name}"], name)
            cameras[name] = CameraObservation(
                frame_id=spec.frame_id,
                optical_convention=spec.optical_convention,
                rgb=image,
                depth_m=None,
                depth_valid=None,
                intrinsics=None,
                camera_to_world=None,
            )
        state_values = {
            "base_position": np.asarray(raw["state.base_position"], dtype=np.float32),
            "base_rotation": np.asarray(raw["state.base_rotation"], dtype=np.float32),
            "end_effector_position_relative": np.asarray(
                raw["state.end_effector_position_relative"], dtype=np.float32
            ),
            "end_effector_rotation_relative": np.asarray(
                raw["state.end_effector_rotation_relative"], dtype=np.float32
            ),
            "gripper_qpos": np.asarray(raw["state.gripper_qpos"], dtype=np.float32),
        }
        vectors = tuple(
            RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name]))
            for spec in self._profile.environment.robot_state
        )
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=self._step * 50_000_000,
            instruction=str(raw["annotation.human.task_description"]),
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCasa365 worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("RoboCasa365 action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("RoboCasa365 worker requires one action at the current step")
        self._observation, _, _, _, info = self._env.step(native_action_dict(action.executable_values()[0]))
        self._success = self._success or bool(info["success"])
        self._step += 1

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCasa365 worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None
