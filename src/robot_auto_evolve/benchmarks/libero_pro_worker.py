from __future__ import annotations

import os
import re
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

from .libero_pro import (
    HARNESS_PROTOCOLS,
    HARNESS_SUITES,
    ONE_STEP_SMOKE_PROTOCOLS,
    parse_bddl_language,
    split_task_id,
    split_suite,
    upstream_suite,
)
from .libero_pro_paths import libero_pro_config_paths, validate_libero_pro_assets, validate_libero_pro_source
from .molmoact2 import MOLMOACT2_LIBERO_ACTION_SPEC
from .pi05 import PI05_LIBERO_ACTION_SPEC
from .render_integrity import validate_mujoco_rgb
from .rlinf_pi05 import RLINF_PI05_LIBERO_ACTION_SPEC
from .smoke_horizon import smoke_horizon_override
from .transforms import matrix_to_quaternion_xyzw
from .xvla import LIBERO_ACTION_SPEC as XVLA_LIBERO_ABSOLUTE_ACTION_SPEC


# W8: the LIBERO-Pro env is driven by ONE 7-D LIBERO action per step, in either DELTA control
# (use_delta=True) or ABSOLUTE control (use_delta=False), exactly like the standard-LIBERO worker.
# The 7-D-delta policies (RLinf pi0.5, LeRobot pi0.5, MolmoAct2) all use the IDENTICAL delta spec
# (verified equal); X-VLA uses the absolute spec. The worker derives the controller mode from the
# profile's action spec, so it is policy-agnostic across the whole LIBERO policy family.
LIBERO_PRO_DELTA_ACTION_SPECS = (
    RLINF_PI05_LIBERO_ACTION_SPEC,
    PI05_LIBERO_ACTION_SPEC,
    MOLMOACT2_LIBERO_ACTION_SPEC,
)


def _validated_runtime() -> tuple[Path, Path]:
    source_value = os.environ.get("ROBOT_AE_LIBERO_PRO_SOURCE")
    assets_value = os.environ.get("ROBOT_AE_LIBERO_PRO_ASSETS")
    config_value = os.environ.get("LIBERO_CONFIG_PATH")
    if not source_value or not assets_value or not config_value:
        raise RuntimeError("LIBERO-Pro worker requires source, asset, and config paths")
    source = validate_libero_pro_source(source_value)
    assets = Path(assets_value).resolve()
    validate_libero_pro_assets(assets)
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
        raise RuntimeError("LIBERO-Pro config.yaml is absent")
    actual = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    expected = libero_pro_config_paths(source, assets)
    if actual != expected:
        raise RuntimeError("LIBERO-Pro runtime config differs from pinned source and assets")
    run_root = config_dir.parent
    for name in ("MPLCONFIGDIR", "ROBOSUITE_LOG_PATH", "TMPDIR"):
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"LIBERO-Pro worker requires {name}")
        try:
            Path(value).resolve().relative_to(run_root)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be under the LIBERO-Pro run directory") from exc
    log_utils = Path(sysconfig.get_paths()["purelib"]) / "robosuite" / "utils" / "log_utils.py"
    if 'logging.FileHandler(os.environ["ROBOSUITE_LOG_PATH"])' not in log_utils.read_text(encoding="utf-8"):
        raise RuntimeError("robosuite installation lacks the output-local logging patch")
    return source / "libero" / "libero", assets


class RLinfPi05LiberoProWorker:
    SETTLE_STEPS = 10

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("LIBERO-Pro worker requires Profile and EpisodeKey")
        suite = profile.environment.suite
        task_suite, task_slug = split_task_id(episode.task_id)
        if suite not in HARNESS_SUITES or task_suite != suite:
            raise StrictSchemaError("LIBERO-Pro episode task and profile suite differ")
        expected_horizon = HARNESS_PROTOCOLS[suite].get(
            episode.protocol,
            ONE_STEP_SMOKE_PROTOCOLS[suite].get(episode.protocol),
        )
        if smoke_horizon_override() is None and expected_horizon != episode.horizon:
            raise StrictSchemaError("LIBERO-Pro episode protocol or horizon differs")
        if profile.policy.action_spec in LIBERO_PRO_DELTA_ACTION_SPECS:
            self._use_delta = True
        elif profile.policy.action_spec == XVLA_LIBERO_ABSOLUTE_ACTION_SPEC:
            self._use_delta = False
        else:
            raise StrictSchemaError("LIBERO-Pro worker action spec is not a supported LIBERO delta/absolute spec")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("LIBERO-Pro worker requires one-action policy responses")
        self._action_spec = profile.policy.action_spec
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._task_slug = task_slug
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._instruction: str | None = None
        self._step = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("LIBERO-Pro worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("LIBERO-Pro worker requires PYTHONNOUSERSITE=1")
        package, _ = _validated_runtime()
        for name, expected in {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(self._render_gpu_id)}.items():
            actual = os.environ.get(name)
            if actual is not None and actual != expected:
                raise RuntimeError(f"{name} differs from the worker render assignment")
            os.environ[name] = expected
        import libero.libero as libero_package
        if Path(libero_package.__file__).resolve() != package / "__init__.py":
            raise RuntimeError("imported LIBERO-Pro package differs from pinned source")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite_name = upstream_suite(self._profile.environment.suite)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task_names = suite.get_task_names()
        if self._task_slug not in task_names:
            raise StrictSchemaError("episode task is absent from the pinned LIBERO-Pro suite")
        task_index = task_names.index(self._task_slug)
        task = suite.get_task(task_index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self._instruction = parse_bddl_language(bddl)
        cameras = {item.name: item for item in self._profile.environment.cameras}
        if set(cameras) != {"main", "wrist"} or {(item.width, item.height) for item in cameras.values()} != {(256, 256)}:
            raise StrictSchemaError("LIBERO-Pro profile cameras must be main and wrist at 256x256")
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=self._render_gpu_id,
        )
        self._env.seed(self._episode.environment_seed)
        self._env.reset()
        match = re.fullmatch(r"init_state_(\d{2})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("LIBERO-Pro scenario_id must be init_state_NN")
        state_index = int(match.group(1))
        states = suite.get_task_init_states(task_index)
        if state_index >= len(states):
            raise StrictSchemaError("LIBERO-Pro initial-state index is unavailable")
        self._observation = self._env.set_init_state(states[state_index])
        settle = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0), dtype=np.float32)
        for _ in range(self.SETTLE_STEPS):
            self._observation, _, _, _ = self._env.step(settle)
        for robot in self._env.env.robots:
            robot.controller.use_delta = self._use_delta
        self._step = 0
        self._success = False

    def _eef_pose(self, raw: dict[str, Any]) -> np.ndarray:
        # Delta-family default (RLinf pi0.5, LeRobot pi0.5, MolmoAct2): the end-effector pose the
        # frozen policy sees comes straight from the raw simulator observation, EXACTLY as each
        # policy's standard-LIBERO worker does (workers.py Pi05LiberoWorker._eef_pose,
        # molmoact2_worker.py MolmoAct2LiberoWorker._eef_pose). X-VLA overrides this (controller
        # source) in XVLALiberoProWorker to match its absolute-control standard-LIBERO worker.
        return np.concatenate(
            (
                np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
                np.asarray(raw["robot0_eef_quat"], dtype=np.float32),
            )
        )

    def observe(self) -> FairObservation:
        if self._env is None or self._observation is None or self._instruction is None or self._closed:
            raise RuntimeError("LIBERO-Pro worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        images = {
            "main": validate_mujoco_rgb(
                np.ascontiguousarray(raw["agentview_image"], dtype=np.uint8), "LIBERO-Pro main"
            ),
            "wrist": validate_mujoco_rgb(
                np.ascontiguousarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8), "LIBERO-Pro wrist"
            ),
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
        state_values = {
            "eef_pose": np.ascontiguousarray(self._eef_pose(raw), dtype=np.float32),
            "gripper_position": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        }
        vectors = []
        for spec in self._profile.environment.robot_state:
            if spec.name not in state_values:
                raise StrictSchemaError(f"LIBERO-Pro worker cannot provide robot state {spec.name!r}")
            vectors.append(RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name])))
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=self._step * 50_000_000,
            instruction=self._instruction,
            cameras=cameras,
            proprioception=RobotProprioception(tuple(vectors)),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("LIBERO-Pro worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self._action_spec:
            raise StrictSchemaError("LIBERO-Pro action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("LIBERO-Pro worker requires one action at the current step")
        native = action.executable_values()[0]
        if native.shape != (7,):
            raise StrictSchemaError("LIBERO-Pro native action must have width 7")
        self._observation, _, _, _ = self._env.step(native)
        self._success = self._success or bool(self._env.check_success())
        self._step += 1

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("LIBERO-Pro worker is not active")
        return self._success

    def runtime_info(self) -> dict[str, object]:
        base_suite, perturbation = split_suite(self._profile.environment.suite)
        return {
            "suite": self._profile.environment.suite,
            "upstream_suite": upstream_suite(self._profile.environment.suite),
            "base_suite": base_suite,
            "perturbation": perturbation,
            "instruction_source": "bddl_language",
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None


# Per-policy LIBERO-Pro workers. The LIBERO-Pro env is identical across policies; what differs is
# the SAME per-policy fidelity the standard-LIBERO workers already encode, so each pro worker must
# match its own standard-LIBERO worker exactly (otherwise the pro eval starts each episode from a
# different proprioception/settle state than the policy's standard eval -> changed eval semantics):
#   - RLinf pi0.5 / LeRobot pi0.5  -> base RLinfPi05LiberoProWorker: raw eef, SETTLE 10, delta.
#     (Pi05LiberoWorker: raw eef, SETTLE 10.) The base is byte-unchanged from the committed,
#     GPU-validated rlinf_pi05_libero_pro route.
#   - X-VLA  -> XVLALiberoProWorker: CONTROLLER eef (ee_pos + quat(ee_ori_mat)), SETTLE 10,
#     absolute. Matches workers.py LiberoWorker (the standard X-VLA LIBERO worker).
#   - MolmoAct2 -> MolmoAct2LiberoProWorker: raw eef, SETTLE 50, delta. Matches
#     molmoact2_worker.py MolmoAct2LiberoWorker (which overrides SETTLE_STEPS=50).
# use_delta is still derived from the profile's action spec in __init__, so it stays correct here.
class XVLALiberoProWorker(RLinfPi05LiberoProWorker):
    def _eef_pose(self, raw: dict[str, Any]) -> np.ndarray:
        controller = self._env.env.robots[0].controller
        return np.concatenate(
            (
                np.asarray(controller.ee_pos, dtype=np.float32),
                np.asarray(matrix_to_quaternion_xyzw(controller.ee_ori_mat), dtype=np.float32),
            )
        )


class MolmoAct2LiberoProWorker(RLinfPi05LiberoProWorker):
    SETTLE_STEPS = 50
