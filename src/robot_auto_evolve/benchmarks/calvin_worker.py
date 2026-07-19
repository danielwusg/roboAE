from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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

from .calvin_protocol import (
    CALVIN_ENV_COMMIT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_TACTO_COMMIT,
    OFFICIAL_GENERATOR_SHA256,
    OFFICIAL_RESET_SOURCE_SHA256,
    PROJECT_PROTOCOL,
    RELEASED_XVLA_SUBTASK_HORIZON,
    XVLA_SOURCE_COMMIT,
    load_sequence_manifest,
    official_reset_state,
    parse_project_scenario,
)
from .smoke_horizon import smoke_horizon_override
from .xvla import CALVIN_ACTION_SPEC, CALVIN_TASKS


def calvin_egl_environment(render_gpu_id: int) -> dict[str, str]:
    if type(render_gpu_id) is not int or render_gpu_id < 0:
        raise StrictSchemaError("render_gpu_id must be a nonnegative int")
    return {"EGL_VISIBLE_DEVICES": str(render_gpu_id), "PYOPENGL_PLATFORM": "egl"}


def configure_calvin_egl(render_gpu_id: int) -> None:
    expected = calvin_egl_environment(render_gpu_id)
    if os.environ.get("EGL_VISIBLE_DEVICE") is not None:
        raise RuntimeError("CALVIN rejects unsupported singular EGL_VISIBLE_DEVICE")
    for name, value in expected.items():
        if os.environ.get(name) not in {None, value}:
            raise RuntimeError(f"{name} differs from the worker render assignment")
    os.environ.update(expected)


def _ignore_environment_close() -> None:
    return None


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _require_clean(path: Path, name: str) -> None:
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"{name} source working tree is dirty: {dirty.splitlines()[0]}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_sources() -> tuple[Path, Path, Path]:
    calvin_value = os.environ.get("ROBOT_AE_CALVIN_SOURCE")
    xvla_value = os.environ.get("ROBOT_AE_XVLA_SOURCE")
    manifest_value = os.environ.get("ROBOT_AE_CALVIN_SEQUENCE_MANIFEST")
    if not calvin_value or not xvla_value or not manifest_value:
        raise RuntimeError("CALVIN worker requires pinned source and sequence manifest paths")
    calvin = Path(calvin_value).resolve()
    calvin_env = calvin / "calvin_env"
    tacto = calvin_env / "tacto"
    xvla = Path(xvla_value).resolve()
    manifest = Path(manifest_value).resolve()
    expected = (CALVIN_SOURCE_COMMIT, CALVIN_ENV_COMMIT, CALVIN_TACTO_COMMIT, XVLA_SOURCE_COMMIT)
    actual = (_git_head(calvin), _git_head(calvin_env), _git_head(tacto), _git_head(xvla))
    if actual != expected:
        raise RuntimeError(f"CALVIN source revision mismatch: {actual}")
    _require_clean(calvin, "CALVIN")
    _require_clean(xvla, "X-VLA")
    if not (calvin_env / "calvin_env" / "envs" / "play_table_env.py").is_file():
        raise RuntimeError("CALVIN environment checkout is incomplete")
    validation = xvla / "evaluation" / "calvin" / "ABC_D" / "validation"
    required = (
        validation / ".hydra" / "merged_config.yaml",
        validation / "new_playtable_tasks.yaml",
        validation / "new_playtable_validation.yaml",
    )
    if any(not path.is_file() for path in required) or not manifest.is_file():
        raise RuntimeError("CALVIN evaluation files are incomplete")
    generator = calvin / "calvin_models" / "calvin_agent" / "evaluation" / "multistep_sequences.py"
    reset_source = calvin / "calvin_models" / "calvin_agent" / "evaluation" / "utils.py"
    if _sha256(generator) != OFFICIAL_GENERATOR_SHA256 or _sha256(reset_source) != OFFICIAL_RESET_SOURCE_SHA256:
        raise RuntimeError("CALVIN official sequence source hash mismatch")
    load_sequence_manifest(manifest)
    return calvin_env, validation, manifest


class CalvinWorker:
    ACTION_SPEC = CALVIN_ACTION_SPEC

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("CALVIN worker requires Profile and EpisodeKey")
        if profile.environment.suite != "calvin_abc_d_prefix1":
            raise StrictSchemaError("CALVIN worker requires the ABC-to-D prefix-one suite")
        if profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("CALVIN worker action spec differs from profile")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("CALVIN worker requires one-action policy responses")
        if episode.task_id not in CALVIN_TASKS:
            raise StrictSchemaError("CALVIN worker received an unknown task")
        allowed = {
            (RELEASED_XVLA_SUBTASK_HORIZON, PROJECT_PROTOCOL),
            (9, "xvla_calvin_prefix1_smoke_v1"),
        }
        allowed_protocols = {protocol for _, protocol in allowed}
        if smoke_horizon_override() is None:
            if (episode.horizon, episode.protocol) not in allowed:
                raise StrictSchemaError("CALVIN episode protocol or horizon differs")
        elif episode.protocol not in allowed_protocols:
            raise StrictSchemaError("CALVIN episode protocol or horizon differs")
        if episode.environment_seed != 0:
            raise StrictSchemaError("CALVIN official sequence environment seed must be zero")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._sequence_index = parse_project_scenario(episode.scenario_id)
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._oracle: Any = None
        self._observation: dict[str, Any] | None = None
        self._start_info: dict[str, Any] | None = None
        self._instruction = ""
        self._step = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("CALVIN worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("CALVIN worker requires PYTHONNOUSERSITE=1")
        configure_calvin_egl(self._render_gpu_id)
        calvin_env, validation, manifest = _validated_sources()
        import calvin_env as calvin_package
        import hydra
        from omegaconf import OmegaConf
        from calvin_env.envs.play_table_env import get_env

        if Path(calvin_package.__file__).resolve() != calvin_env / "calvin_env" / "__init__.py":
            raise RuntimeError("imported CALVIN package differs from the pinned source")
        sequences = load_sequence_manifest(manifest)
        sequence = sequences[self._sequence_index]
        if sequence.tasks[0] != self._episode.task_id:
            raise StrictSchemaError("CALVIN episode task differs from its official sequence")
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        if set(camera_specs) != {"static", "wrist"}:
            raise StrictSchemaError("CALVIN profile cameras must be static and wrist")
        if (camera_specs["static"].width, camera_specs["static"].height) != (200, 200):
            raise StrictSchemaError("CALVIN static camera must be 200x200")
        if (camera_specs["wrist"].width, camera_specs["wrist"].height) != (84, 84):
            raise StrictSchemaError("CALVIN wrist camera must be 84x84")
        task_cfg = OmegaConf.load(validation / "new_playtable_tasks.yaml")
        self._oracle = hydra.utils.instantiate(task_cfg)
        annotations = yaml.safe_load((validation / "new_playtable_validation.yaml").read_text(encoding="utf-8"))
        language = annotations[self._episode.task_id][0]
        self._instruction = language.split("\n", 1)[0].replace("\u2019", "'")
        robot_obs, scene_obs = official_reset_state(sequence.initial_condition)
        self._env = get_env(
            validation,
            obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
            show_gui=False,
        )
        self._env.seed(0)
        self._observation = self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self._start_info = self._env.get_info()
        self._step = 0
        self._success = False

    def observe(self) -> FairObservation:
        if self._env is None or self._observation is None or self._closed:
            raise RuntimeError("CALVIN worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        images = {
            "static": np.ascontiguousarray(raw["rgb_obs"]["rgb_static"], dtype=np.uint8),
            "wrist": np.ascontiguousarray(raw["rgb_obs"]["rgb_gripper"], dtype=np.uint8),
        }
        cameras = {
            name: CameraObservation(
                frame_id=camera_specs[name].frame_id,
                optical_convention=camera_specs[name].optical_convention,
                rgb=image,
                depth_m=None,
                depth_valid=None,
                intrinsics=None,
                camera_to_world=None,
            )
            for name, image in images.items()
        }
        robot = np.asarray(raw["robot_obs"], dtype=np.float32)
        if robot.shape != (15,):
            raise RuntimeError(f"CALVIN robot observation has invalid shape {robot.shape}")
        state_values = {
            "eef_euler_xyz": robot[3:6],
            "eef_position": robot[:3],
            "gripper_action": robot[-1:],
        }
        vectors = tuple(
            RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name], dtype=np.float32))
            for spec in self._profile.environment.robot_state
        )
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=round(self._step * 1_000_000_000 / 30),
            instruction=self._instruction,
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._start_info is None or self._closed:
            raise RuntimeError("CALVIN worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("CALVIN action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("CALVIN worker requires one action at the current step")
        native = action.executable_values()[0]
        if native.shape != (8,):
            raise StrictSchemaError("CALVIN native action must have width 8")
        if native[-1] not in {-1.0, 1.0}:
            raise StrictSchemaError("CALVIN gripper action must be -1 or 1")
        quaternion = np.asarray(native[3:7], dtype=np.float64)
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-4):
            raise StrictSchemaError("CALVIN action quaternion must be normalized")
        command = (
            np.asarray(native[:3], dtype=np.float64),
            quaternion,
            int(native[-1]),
        )
        self._observation, _, _, current_info = self._env.step(command)
        achieved = self._oracle.get_task_info_for_set(self._start_info, current_info, {self._episode.task_id})
        self._success = self._success or self._episode.task_id in achieved
        self._step += 1

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("CALVIN worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        environment = self._env
        self._env = None
        if environment is not None:
            environment.close()
            environment.close = _ignore_environment_close
        self._observation = None
        self._start_info = None
