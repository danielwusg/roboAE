from __future__ import annotations

import os
import re
import subprocess
import sysconfig
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
from robot_auto_evolve.runtime_paths import project_root_from_package

from .depth3d import robosuite_camera_3d
from .libero_paths import libero_config_paths
from .libero_suites import LIBERO_SUITE_TASKS, PI05_LIBERO_PROTOCOLS, RLINF_PI05_LIBERO_PROTOCOLS, XVLA_LIBERO_PROTOCOLS
from .pi05 import PI05_LIBERO_ACTION_SPEC
from .render_integrity import validate_mujoco_rgb
from .smoke_horizon import smoke_horizon_override
from .transforms import matrix_to_quaternion_xyzw
from .xvla import LIBERO_ACTION_SPEC


LIBERO_SOURCE_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def _validated_libero_source() -> Path:
    source_value = os.environ.get("ROBOT_AE_LIBERO_SOURCE")
    config_value = os.environ.get("LIBERO_CONFIG_PATH")
    if not source_value or not config_value:
        raise RuntimeError("LIBERO worker requires ROBOT_AE_LIBERO_SOURCE and LIBERO_CONFIG_PATH")
    source = Path(source_value).resolve()
    package = source / "libero" / "libero"
    if not (package / "__init__.py").is_file() or not (source / ".git").is_dir():
        raise RuntimeError("LIBERO source checkout is incomplete")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != LIBERO_SOURCE_COMMIT:
        raise RuntimeError(f"LIBERO source revision mismatch: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"LIBERO source working tree is dirty: {dirty.splitlines()[0]}")
    config_dir = Path(config_value).resolve()
    project_root = project_root_from_package()
    try:
        relative = config_dir.relative_to(project_root / "runs")
    except ValueError as exc:
        raise RuntimeError("LIBERO_CONFIG_PATH must be under the clean runs directory") from exc
    if len(relative.parts) < 2:
        raise RuntimeError("LIBERO_CONFIG_PATH must identify a run")
    config_file = config_dir / "config.yaml"
    if not config_file.is_file():
        raise RuntimeError("LIBERO config.yaml is absent")
    expected = libero_config_paths(source)
    actual = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if actual != expected or any(not Path(path).is_dir() for path in expected.values()):
        raise RuntimeError("LIBERO config paths differ from the pinned source and shared dataset directory")
    run_root = config_dir.parent
    for name in ("MPLCONFIGDIR", "ROBOSUITE_LOG_PATH", "TMPDIR"):
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"LIBERO worker requires {name}")
        try:
            Path(value).resolve().relative_to(run_root)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be under the LIBERO run directory") from exc
    log_utils = Path(sysconfig.get_paths()["purelib"]) / "robosuite" / "utils" / "log_utils.py"
    if 'logging.FileHandler(os.environ["ROBOSUITE_LOG_PATH"])' not in log_utils.read_text(encoding="utf-8"):
        raise RuntimeError("robosuite installation lacks the pinned output-local logging patch")
    return package


class LiberoWorker:
    ACTION_SPEC = LIBERO_ACTION_SPEC
    PROTOCOLS = XVLA_LIBERO_PROTOCOLS
    SETTLE_STEPS = 10
    USE_DELTA_CONTROL = False

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("libero worker requires Profile and EpisodeKey")
        suite = profile.environment.suite
        if suite not in LIBERO_SUITE_TASKS or suite not in self.PROTOCOLS:
            raise StrictSchemaError("libero worker received an unsupported suite")
        if profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("libero worker action spec differs from profile")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("libero worker requires one-action policy responses")
        if episode.task_id not in LIBERO_SUITE_TASKS[suite]:
            raise StrictSchemaError("episode task is absent from the selected LIBERO suite")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._render_gpu_id = render_gpu_id
        # Revision 2: per-camera `has_depth` in the route profile is the 3D switch (see
        # benchmarks/depth3d.py). All false => environment build and observation unchanged.
        self._wants_depth = any(item.has_depth for item in profile.environment.cameras)
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._task: Any = None
        self._step = 0
        self._success = False
        self._closed = False
        self._validate_episode()

    def _validate_episode(self) -> None:
        suite = self._profile.environment.suite
        expected = self.PROTOCOLS[suite].get(self._episode.protocol)
        if smoke_horizon_override() is None and expected != self._episode.horizon:
            raise StrictSchemaError("LIBERO episode protocol or horizon differs")

    def _configure_controllers(self) -> None:
        for robot in self._env.env.robots:
            robot.controller.use_delta = self.USE_DELTA_CONTROL

    def _eef_pose(self, raw: dict[str, Any]) -> np.ndarray:
        controller = self._env.env.robots[0].controller
        return np.concatenate(
            (
                np.asarray(controller.ee_pos, dtype=np.float32),
                np.asarray(matrix_to_quaternion_xyzw(controller.ee_ori_mat), dtype=np.float32),
            )
        )

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("libero worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("LIBERO worker requires PYTHONNOUSERSITE=1")
        package = _validated_libero_source()
        for name, expected in {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(self._render_gpu_id)}.items():
            actual = os.environ.get(name)
            if actual is not None and actual != expected:
                raise RuntimeError(f"{name} differs from the worker render assignment")
            os.environ[name] = expected
        import libero.libero as libero_package
        if Path(libero_package.__file__).resolve() != package / "__init__.py":
            raise RuntimeError("imported LIBERO package differs from ROBOT_AE_LIBERO_SOURCE")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[self._profile.environment.suite]()
        task_names = suite.get_task_names()
        if self._episode.task_id not in task_names:
            raise StrictSchemaError("episode task is absent from the pinned LIBERO suite")
        task_index = task_names.index(self._episode.task_id)
        self._task = suite.get_task(task_index)
        bddl = os.path.join(get_libero_path("bddl_files"), self._task.problem_folder, self._task.bddl_file)
        cameras = {item.name: item for item in self._profile.environment.cameras}
        if set(cameras) != {"main", "wrist"} or {(item.width, item.height) for item in cameras.values()} != {(256, 256)}:
            raise StrictSchemaError("LIBERO profile cameras must be main and wrist at 256x256")
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            camera_depths=self._wants_depth,
            render_gpu_device_id=self._render_gpu_id,
        )
        self._env.seed(self._episode.environment_seed)
        self._env.reset()
        match = re.fullmatch(r"init_state_(\d{2})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("LIBERO scenario_id must be init_state_NN")
        state_index = int(match.group(1))
        states = suite.get_task_init_states(task_index)
        if state_index >= len(states):
            raise StrictSchemaError("LIBERO initial-state index is unavailable")
        self._observation = self._env.set_init_state(states[state_index])
        settle = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0), dtype=np.float32)
        for _ in range(self.SETTLE_STEPS):
            self._observation, _, _, _ = self._env.step(settle)
        self._configure_controllers()
        self._step = 0
        self._success = False

    def observe(self) -> FairObservation:
        if self._env is None or self._observation is None or self._closed:
            raise RuntimeError("libero worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        images = {
            "main": validate_mujoco_rgb(
                np.ascontiguousarray(raw["agentview_image"], dtype=np.uint8), "LIBERO main"
            ),
            "wrist": validate_mujoco_rgb(
                np.ascontiguousarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8), "LIBERO wrist"
            ),
        }
        cameras = {}
        for name, image in images.items():
            spec = camera_specs[name]
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                depth_m, depth_valid, intrinsics, camera_to_world = robosuite_camera_3d(
                    self._env.sim,
                    spec.frame_id,
                    raw[f"{spec.frame_id}_depth"],
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
        state_values = {
            "eef_pose": self._eef_pose(raw),
            "gripper_position": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        }
        vectors = []
        for spec in self._profile.environment.robot_state:
            if spec.name not in state_values:
                raise StrictSchemaError(f"LIBERO worker cannot provide robot state {spec.name!r}")
            vectors.append(RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name])))
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=self._step * 50_000_000,
            instruction=self._task.language,
            cameras=cameras,
            proprioception=RobotProprioception(tuple(vectors)),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("libero worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("LIBERO action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("LIBERO worker requires one action at the current step")
        native = action.executable_values()[0]
        if native.shape != (7,):
            raise StrictSchemaError("LIBERO native action must have width 7")
        self._observation, _, _, _ = self._env.step(native)
        self._success = self._success or bool(self._env.check_success())
        self._step += 1

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("libero worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None


class Pi05LiberoWorker(LiberoWorker):
    ACTION_SPEC = PI05_LIBERO_ACTION_SPEC
    PROTOCOLS = PI05_LIBERO_PROTOCOLS
    USE_DELTA_CONTROL = True

    def _eef_pose(self, raw: dict[str, Any]) -> np.ndarray:
        position = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        quaternion = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise StrictSchemaError("pi0.5 LIBERO end-effector state shape mismatch")
        return np.ascontiguousarray(np.concatenate((position, quaternion)), dtype=np.float32)


class RLinfPi05LiberoWorker(Pi05LiberoWorker):
    PROTOCOLS = RLINF_PI05_LIBERO_PROTOCOLS
