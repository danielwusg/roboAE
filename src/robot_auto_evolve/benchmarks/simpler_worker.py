from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from itertools import product
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

from .xvla import SIMPLER_GOOGLE_ACTION_SPEC, SIMPLER_WIDOWX_ACTION_SPEC, WIDOWX_GRIPPER_THRESHOLDS


SIMPLER_COMMIT = "06accaca93535902d408da4855f21cece12bceb7"
MANISKILL_COMMIT = "ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682"
XVLA_COMMIT = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
HARNESS_COMMIT = "2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56"
SCENARIO_HASHES = {
    "google-VA/configs/coke_can.json": "37e5e69bb1c32a166828147ab8fd2aa2b962a92fc27ce831e0375e50ce6ea8e2",
    "google-VA/configs/move_near.json": "a04d88094d4b3a636ae5d37dec8ff2b49235d0c7abd52fffd66bdada1f6c8795",
    "google-VA/configs/open_close.json": "653f47263af0db4fccac4af986a37d4cc516b2939d25edcb1680bc14748cf6ae",
    "google-VA/configs/place_in.json": "77d9336b9388498208235994a87ec0da79d917dccddd5400cf1abeceb64e694d",
    "google-VM/configs/coke_can.json": "2e3a8578ca9f8f12d67effa46700c7c7be483c04bd9a7caa8457cce5ea8dbb90",
    "google-VM/configs/move_near.json": "868557ca1f9d15477bdd7fa7356772100d1bede8e5f2094f89a067d756a574b0",
    "google-VM/configs/open_close.json": "220835e6bad4dd0bfc30ad5c6e5bce2d4554526bd96a0d7e62a562acce20976c",
    "google-VM/configs/place_in.json": "67f68702c1085f2721eec24f1ba18b32e92ae0f786aceb612d3638cbe5d9429b",
}
TRANSFORM_HASHES = {
    "Dockerfile.simpler_xvla": "6288975f91a47a59445506c611bd3d24f99b89282c72a0861cc9a8f7f795f556",
    "xvla_absolute_ee.patch": "19803d0dbcb64602fb0dceb5286765bd8b1fd22b77529b1355de1f9443c89aad",
    "xvla_google_robot_patch.py": "7e1521515f7dc13405ef0547515cc0e12f409063d08e0847b061ee25b8a79260",
    "xvla_sink_camera.py": "f20f35eff9adcee17aa06024b5bbb86e3449e81ba2faa8387313f6839f5c48f9",
}
CRITICAL_FILES = frozenset(
    {
        *(f".robot_auto_evolve/xvla_simpler/{name}" for name in SCENARIO_HASHES),
        "simpler_env/__init__.py",
        "simpler_env/utils/env/observation_utils.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/agents/configs/google_robot/defaults.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/agents/configs/widowx/defaults.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/agents/controllers/pd_ee_pose.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/agents/robots/googlerobot.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/agents/robots/widowx.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/base_env.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/grasp_single_in_scene.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/move_near_in_scene.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/open_drawer_in_scene.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/put_on_in_scene.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/envs/sapien_env.py",
        "ManiSkill2_real2sim/mani_skill2_real2sim/utils/wrappers/observation.py",
    }
)
GOOGLE_CONTROL_MODE = "arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
WIDOWX_CONTROL_MODE = "arm_pd_ee_target_base_pose_gripper_pd_joint_pos"
RENDERER_DEVICE = "cuda:0"
GOOGLE_CONFIG_BY_TASK = {
    "google_robot_pick_coke_can": "coke_can.json",
    "google_robot_move_near": "move_near.json",
    "google_robot_open_drawer": "open_close.json",
    "google_robot_close_drawer": "open_close.json",
    "google_robot_place_apple_in_closed_top_drawer": "place_in.json",
}
GOOGLE_VA_PROTOCOLS = frozenset(
    {
        "xvla_google_va_drawer_transfer_v2",
        "xvla_google_va_full_stack_smoke_v1",
        "xvla_google_va_paper_headline_v1",
        "xvla_google_va_extended_v1",
    }
)
GOOGLE_VM_PROTOCOLS = frozenset(
    {
        "xvla_google_vm_drawer_transfer_v1",
        "xvla_google_vm_full_stack_smoke_v1",
        "xvla_google_vm_paper_headline_v1",
        "xvla_google_vm_extended_v1",
    }
)
WIDOWX_PROTOCOLS = frozenset(
    {
        "xvla_widowx_vm_related_transfer_v2",
        "xvla_widowx_vm_full_stack_smoke_v1",
        "xvla_widowx_vm_standard_v1",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.name not in {".robot_auto_evolve_xvla.json", "z"}):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            payload = b"L\0" + os.readlink(path).encode()
        elif path.is_file():
            payload = b"F\0" + bytes.fromhex(_sha256(path))
        elif path.is_dir():
            payload = b"D\0"
        else:
            raise RuntimeError(f"unsupported derived source entry: {path}")
        digest.update(relative + b"\0" + payload + b"\0")
    return digest.hexdigest()


def validate_simpler_source(source: str | Path, *, full_tree: bool = False) -> Path:
    source = Path(source).resolve()
    marker_path = source / ".robot_auto_evolve_xvla.json"
    if not marker_path.is_file():
        raise RuntimeError("derived X-VLA SimplerEnv marker is absent")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "simpler_commit",
        "maniskill_commit",
        "xvla_commit",
        "harness_commit",
        "transform_sha256",
        "scenario_sha256",
        "critical_file_sha256",
        "tree_sha256",
    }
    if set(marker) != required:
        raise RuntimeError("derived X-VLA SimplerEnv marker has invalid fields")
    expected = {
        "schema_version": 2,
        "simpler_commit": SIMPLER_COMMIT,
        "maniskill_commit": MANISKILL_COMMIT,
        "xvla_commit": XVLA_COMMIT,
        "harness_commit": HARNESS_COMMIT,
        "transform_sha256": TRANSFORM_HASHES,
        "scenario_sha256": SCENARIO_HASHES,
    }
    if any(marker[name] != value for name, value in expected.items()):
        raise RuntimeError("derived X-VLA SimplerEnv provenance differs")
    hashes = marker["critical_file_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != CRITICAL_FILES:
        raise RuntimeError("derived X-VLA SimplerEnv critical hashes are invalid")
    for relative, expected_hash in hashes.items():
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise RuntimeError("derived X-VLA SimplerEnv hash path escapes source") from exc
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"derived X-VLA SimplerEnv file differs: {relative}")
    if full_tree and marker["tree_sha256"] != _tree_sha256(source):
        raise RuntimeError("derived X-VLA SimplerEnv tree differs")
    return source


def validated_simpler_source() -> Path:
    value = os.environ.get("ROBOT_AE_SIMPLER_SOURCE")
    if not value:
        raise RuntimeError("SimplerEnv worker requires ROBOT_AE_SIMPLER_SOURCE")
    return validate_simpler_source(value)


def _range(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if not isinstance(value, list) or len(value) != 3:
        raise StrictSchemaError("Google VA range must be a number or [start,end,count]")
    count = int(value[2])
    if count < 1 or float(count) != float(value[2]):
        raise StrictSchemaError("Google VA range count must be a positive integer")
    return np.linspace(float(value[0]), float(value[1]), count).tolist()


def _euler_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy),
        dtype=np.float64,
    )


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def google_scenario_grid(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rpy = config["robot_init_rot_rpy_range"]
    if not isinstance(rpy, list) or len(rpy) != 9:
        raise StrictSchemaError("Google rotation range must have nine values")
    center = np.asarray(config["robot_init_rot_quat_center"], dtype=np.float64)
    if center.shape != (4,):
        raise StrictSchemaError("Google center quaternion must have four values")
    quaternions = tuple(
        _quaternion_multiply_wxyz(_euler_quaternion_wxyz(roll, pitch, yaw), center)
        for roll, pitch, yaw in product(_range(rpy[:3]), _range(rpy[3:6]), _range(rpy[6:9]))
    )
    if config["obj_variation_mode"] == "episode":
        objects = tuple({"episode_id": index} for index in range(int(config["episode_nums"])))
    elif config["obj_variation_mode"] == "xy":
        objects = tuple(
            {"init_xy": np.array((x, y), dtype=np.float64)}
            for x, y in product(_range(config["obj_init_x_range"]), _range(config["obj_init_y_range"]))
        )
    else:
        raise StrictSchemaError("Google object variation mode is unsupported")
    return tuple(
        {
            "robot_init_options": {
                "init_xy": np.array((x, y), dtype=np.float64),
                "init_rot_quat": quaternion.copy(),
            },
            "obj_init_options": {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obj.items()},
        }
        for x, y, quaternion, obj in product(
            _range(config["robot_init_x"]),
            _range(config["robot_init_y"]),
            quaternions,
            objects,
        )
    )


def _pose_vector(pose: Any) -> np.ndarray:
    value = np.concatenate((np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)))
    if value.shape != (7,) or not np.isfinite(value).all():
        raise StrictSchemaError("SimplerEnv robot-link pose must be finite with width 7")
    norm = np.linalg.norm(value[3:])
    if norm < 1e-8:
        raise StrictSchemaError("SimplerEnv robot-link quaternion is invalid")
    value[3:] /= norm
    return value


def _assert_pose_copy(name: str, value: Any, actual: np.ndarray) -> None:
    observed = np.asarray(value, dtype=np.float64).copy()
    if observed.shape != (7,) or not np.isfinite(observed).all():
        raise StrictSchemaError(f"SimplerEnv {name} pose copy must be finite with width 7")
    norm = np.linalg.norm(observed[3:])
    if norm < 1e-8:
        raise StrictSchemaError(f"SimplerEnv {name} pose copy has an invalid quaternion")
    observed[3:] /= norm
    position_matches = np.allclose(observed[:3], actual[:3], rtol=0.0, atol=1e-5)
    rotation_matches = abs(float(np.dot(observed[3:], actual[3:]))) >= 1.0 - 1e-5
    if not position_matches or not rotation_matches:
        raise StrictSchemaError(f"SimplerEnv {name} pose copy differs from the actual robot link")


def _base_relative_pose(raw: dict[str, Any], environment: Any) -> np.ndarray:
    if not isinstance(raw, dict) or not isinstance(raw.get("agent"), dict) or not isinstance(raw.get("extra"), dict):
        raise StrictSchemaError("SimplerEnv observation must contain agent and extra mappings")
    unwrapped = environment.unwrapped
    base_pose = unwrapped.agent.robot.pose
    tcp_pose = unwrapped.tcp.pose
    base = _pose_vector(base_pose)
    tcp = _pose_vector(tcp_pose)
    if "base_pose" not in raw["agent"]:
        raise StrictSchemaError("SimplerEnv observation is missing the robot base pose")
    _assert_pose_copy("agent.base_pose", raw["agent"]["base_pose"], base)
    if "tcp_pose" in raw["extra"]:
        _assert_pose_copy("extra.tcp_pose", raw["extra"]["tcp_pose"], tcp)
    pose = base_pose.inv() * tcp_pose
    relative = _pose_vector(pose)
    return np.ascontiguousarray(np.concatenate((relative[:3], relative[4:], relative[3:4])), dtype=np.float32)


def _value_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(name): _value_schema(item) for name, item in value.items()}
    array = np.asarray(value)
    return {"dtype": str(array.dtype), "shape": list(array.shape)}


def _observation_schema(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("agent"), dict) or not isinstance(raw.get("extra"), dict):
        raise StrictSchemaError("SimplerEnv observation schema is invalid")
    return {
        "top_level_keys": sorted(raw),
        "agent": _value_schema(raw["agent"]),
        "extra": _value_schema(raw["extra"]),
    }


def validate_simpler_rgb(value: Any, camera_name: str) -> np.ndarray:
    if type(camera_name) is not str or not camera_name:
        raise StrictSchemaError("SimplerEnv camera name must be a nonempty string")
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise StrictSchemaError(f"SimplerEnv camera {camera_name} RGB must be a uint8 array")
    if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] < 1 or value.shape[1] < 1:
        raise StrictSchemaError(f"SimplerEnv camera {camera_name} RGB must have nonempty HWC shape with three channels")
    minimum = int(value.min())
    maximum = int(value.max())
    if minimum == maximum:
        raise StrictSchemaError(f"SimplerEnv camera {camera_name} RGB is constant")
    return value


def simpler_rgb_evidence(value: Any, camera_name: str) -> dict[str, Any]:
    image = np.ascontiguousarray(validate_simpler_rgb(value, camera_name))
    return {
        "camera": camera_name,
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean(dtype=np.float64)),
        "std": float(image.std(dtype=np.float64)),
        "sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
    }


class _SimplerWorker:
    ACTION_SPEC = SIMPLER_WIDOWX_ACTION_SPEC
    SUITE = "simpler_widowx_vm"
    CONTROL_MODE = WIDOWX_CONTROL_MODE
    CONTROL_PERIOD_NS = 200_000_000

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("SimplerEnv worker requires Profile and EpisodeKey")
        if profile.environment.suite != self.SUITE or profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("SimplerEnv worker profile route differs")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("SimplerEnv worker requires one-action policy responses")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._render_gpu_id = render_gpu_id
        self._source: Path | None = None
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._step = 0
        self._success = False
        self._closed = False
        self._validate_episode()

    def _validate_episode(self) -> None:
        raise NotImplementedError

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _renderer_kwargs() -> dict[str, str]:
        return {"device": RENDERER_DEVICE}

    @staticmethod
    def _validated_source() -> Path:
        return validated_simpler_source()

    def _reset_environment(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._env.reset(seed=self._episode.environment_seed, options=options)

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("SimplerEnv worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("SimplerEnv worker requires PYTHONNOUSERSITE=1")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(self._render_gpu_id):
            raise RuntimeError("SimplerEnv render assignment differs from CUDA_VISIBLE_DEVICES")
        icd = os.environ.get("VK_ICD_FILENAMES")
        if not icd or not Path(icd).is_file():
            raise RuntimeError("SimplerEnv worker requires a valid VK_ICD_FILENAMES")
        self._source = self._validated_source()
        for path in (self._source, self._source / "ManiSkill2_real2sim"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import simpler_env

        if Path(simpler_env.__file__).resolve() != self._source / "simpler_env" / "__init__.py":
            raise RuntimeError("imported SimplerEnv package differs from derived source")
        self._env, reset_options = self._make_and_reset()
        self._observation, _ = self._reset_environment(reset_options)
        self._step = 0
        self._success = False

    def observe(self) -> FairObservation:
        if self._env is None or self._observation is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        image = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(self._env, self._observation))
        camera_spec = self._profile.environment.cameras[0]
        if image.shape != (camera_spec.height, camera_spec.width, 3):
            raise StrictSchemaError(f"SimplerEnv camera shape differs: {image.shape}")
        validate_simpler_rgb(image, camera_spec.name)
        camera = CameraObservation(
            frame_id=camera_spec.frame_id,
            optical_convention=camera_spec.optical_convention,
            rgb=image,
            depth_m=None,
            depth_valid=None,
            intrinsics=None,
            camera_to_world=None,
        )
        values = {"eef_pose": _base_relative_pose(self._observation, self._env)}
        vectors = tuple(
            RobotStateVector(spec, values[spec.name])
            for spec in self._profile.environment.robot_state
            if spec.name in values
        )
        if len(vectors) != len(self._profile.environment.robot_state):
            raise StrictSchemaError("SimplerEnv profile requests unavailable robot state")
        try:
            instruction = self._env.unwrapped.get_language_instruction()
        except AttributeError:
            instruction = self._env.get_wrapper_attr("get_language_instruction")()
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=self._step * self.CONTROL_PERIOD_NS,
            instruction=str(instruction),
            cameras={"main": camera},
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("SimplerEnv action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("SimplerEnv worker requires one action at the current step")
        native = action.executable_values()[0]
        if native.shape != (7,):
            raise StrictSchemaError("SimplerEnv native action must have width 7")
        self._observation, _, terminated, _, _ = self._env.step(native)
        self._update_success(bool(terminated))
        self._step += 1

    def _update_success(self, terminated: bool) -> None:
        self._success = self._success or terminated

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        return self._success

    def runtime_info(self) -> dict[str, Any]:
        if self._env is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        unwrapped = self._env.unwrapped
        control_mode = str(unwrapped.control_mode)
        if control_mode != self.CONTROL_MODE:
            raise RuntimeError("SimplerEnv selected controller differs")
        renderer = getattr(unwrapped, "_renderer", None)
        if renderer is None:
            raise RuntimeError("SimplerEnv renderer is unavailable")
        device_api = None
        observed_device = None
        for name in ("get_device", "device"):
            if hasattr(renderer, name):
                value = getattr(renderer, name)
                observed_device = str(value() if callable(value) else value)
                device_api = name
                break
        if observed_device is not None and RENDERER_DEVICE not in observed_device.lower():
            raise RuntimeError("SimplerEnv observed renderer device differs")
        return {
            "control_mode": control_mode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "eef_link_name": str(unwrapped.tcp.name),
            "eef_pose_source": "env.unwrapped.agent.robot.pose.inv()*env.unwrapped.tcp.pose",
            "observation_schema": _observation_schema(self._observation),
            "physical_gpu_id": self._render_gpu_id,
            "renderer_class": f"{type(renderer).__module__}.{type(renderer).__qualname__}",
            "renderer_device_api": device_api,
            "renderer_observed_device": observed_device,
            "renderer_requested_device": self._renderer_kwargs()["device"],
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None


class SimplerWidowXWorker(_SimplerWorker):
    def _validate_episode(self) -> None:
        if self._episode.task_id not in WIDOWX_GRIPPER_THRESHOLDS:
            raise StrictSchemaError("WidowX worker received an unknown task")
        if self._episode.protocol not in WIDOWX_PROTOCOLS:
            raise StrictSchemaError("WidowX episode protocol differs")
        expected_horizon = 31 if self._episode.protocol == "xvla_widowx_vm_full_stack_smoke_v1" else 1200
        if self._episode.horizon != expected_horizon:
            raise StrictSchemaError("WidowX episode horizon differs")
        match = re.fullmatch(r"episode_(\d{2})", self._episode.scenario_id)
        if match is None or int(match.group(1)) >= 24:
            raise StrictSchemaError("WidowX scenario must be episode_00 through episode_23")
        self._object_episode = int(match.group(1))

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        import simpler_env

        environment = simpler_env.make(
            self._episode.task_id,
            control_mode=WIDOWX_CONTROL_MODE,
            max_episode_steps=self._episode.horizon,
            renderer_kwargs=self._renderer_kwargs(),
        )
        return environment, {"obj_init_options": {"episode_id": self._object_episode}}


def _google_task(env_name: Any) -> str:
    value = str(env_name)
    prefixes = {
        "GraspSingle": "google_robot_pick_coke_can",
        "MoveNear": "google_robot_move_near",
        "Open": "google_robot_open_drawer",
        "Close": "google_robot_close_drawer",
        "Place": "google_robot_place_apple_in_closed_top_drawer",
    }
    matches = [task for prefix, task in prefixes.items() if value.startswith(prefix)]
    if len(matches) != 1:
        raise StrictSchemaError(f"unsupported Google environment name {value!r}")
    return matches[0]


class _SimplerGoogleWorker(_SimplerWorker):
    ACTION_SPEC = SIMPLER_GOOGLE_ACTION_SPEC
    SUITE = "simpler_google_va"
    CONTROL_MODE = GOOGLE_CONTROL_MODE
    CONTROL_PERIOD_NS = 333_333_333
    VARIANT = "google-VA"
    PROTOCOLS = GOOGLE_VA_PROTOCOLS

    def _validate_episode(self) -> None:
        if self._episode.task_id not in GOOGLE_CONFIG_BY_TASK:
            raise StrictSchemaError("Google worker received an unknown task")
        if self._episode.protocol not in self.PROTOCOLS:
            raise StrictSchemaError("Google episode protocol differs")
        match = re.fullmatch(r"([a-z0-9_]+)__grid_(\d{3})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("Google scenario identifier differs")
        self._scenario_name = match.group(1)
        self._grid_index = int(match.group(2))

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        if self._source is None:
            raise RuntimeError("Google source is not validated")
        config_path = (
            self._source
            / ".robot_auto_evolve"
            / "xvla_simpler"
            / self.VARIANT
            / "configs"
            / GOOGLE_CONFIG_BY_TASK[self._episode.task_id]
        )
        scenarios = json.loads(config_path.read_text(encoding="utf-8"))
        if self._scenario_name not in scenarios:
            raise StrictSchemaError("Google scenario is absent from pinned X-VLA config")
        config = scenarios[self._scenario_name]
        expected_horizon = 11 if self._episode.protocol.endswith("_full_stack_smoke_v1") else 2 * int(config["max_episode_steps"])
        if _google_task(config["env_name"]) != self._episode.task_id or self._episode.horizon != expected_horizon:
            raise StrictSchemaError("Google episode horizon or task differs from scenario")
        grid = google_scenario_grid(config)
        if self._grid_index >= len(grid):
            raise StrictSchemaError("Google grid member is unavailable")
        import gymnasium as gym
        import simpler_env  # noqa: F401

        kwargs = {
            "obs_mode": "rgbd",
            "max_episode_steps": self._episode.horizon,
            "robot": config["robot_name"],
            "sim_freq": 513,
            "control_freq": 3,
            "control_mode": GOOGLE_CONTROL_MODE,
            "scene_name": config["scene_name"],
            "camera_cfgs": {"add_segmentation": True},
            **config["additional_env_build_kwargs"],
            "renderer_kwargs": self._renderer_kwargs(),
        }
        for name in ("rgb_overlay_path", "rgb_overlay_cameras"):
            if name in config:
                value = config[name]
                if isinstance(value, str):
                    value = value.replace("{SIMPLER_DIR}", str(self._source))
                kwargs[name] = value
        if "rgb_overlay_path" in kwargs and "rgb_overlay_cameras" not in kwargs:
            kwargs["rgb_overlay_cameras"] = ["overhead_camera"]
        environment = gym.make(config["env_name"], **kwargs)
        return environment, grid[self._grid_index]


class SimplerGoogleVAWorker(_SimplerGoogleWorker):
    pass


class SimplerGoogleVMWorker(_SimplerGoogleWorker):
    SUITE = "simpler_google_vm"
    VARIANT = "google-VM"
    PROTOCOLS = GOOGLE_VM_PROTOCOLS
