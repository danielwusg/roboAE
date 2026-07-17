from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import (
    CameraObservation,
    FairObservation,
    RobotProprioception,
    RobotStateSpec,
    RobotStateVector,
    StrictSchemaError,
)

from .droid import DROID_ACTION_SPEC
from .robolab120_batching import RoboLabBatch
from .robolab120_rpc import (
    RoboLabActionBatch,
    RoboLabAppConfig,
    RoboLabObservationBatch,
    RoboLabPrivateStatus,
    RoboLabPrivateStatusBatch,
    RoboLabRpcRequest,
    RoboLabRpcResponse,
    read_robolab_request,
    write_robolab_response,
)


ROBOLAB_SOURCE_COMMIT = "92313e06dd90d2eeedf48da567755e0b1a8e32d1"
ROBOLAB_ASSET_TREE_SHA256 = "9249965eb7cab12970eda2a09d919bf9cd149451a0f3b22c12f7919558bffb89"
ROBOLAB_SUITE = "robolab120_droid_jointpos"
ROBOLAB_INSTRUCTION_TYPES = frozenset({"default", "specific", "vague"})
CONTROL_PERIOD_NS = round(1_000_000_000 / 15)

_ARM_SPEC = RobotStateSpec(
    name="arm_joint_position",
    quantity="joint_position",
    frame_id="franka_joint_space",
    reference_frame="robot_base",
    component_names=tuple(f"panda_joint{index}" for index in range(1, 8)),
    units=("radian",) * 7,
    representation="vector",
    quaternion_order="none",
)
_GRIPPER_SPEC = RobotStateSpec(
    name="gripper_position",
    quantity="gripper_position",
    frame_id="robotiq_2f_85",
    reference_frame="none",
    component_names=("closed_fraction",),
    units=("normalized",),
    representation="vector",
    quaternion_order="none",
)


def _array(
    value: Any,
    path: str,
    *,
    dtype: np.dtype[Any],
    ndim: int,
    tail: tuple[int, ...] = (),
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != ndim:
        raise StrictSchemaError(f"{path}: invalid array")
    if tail and value.shape[-len(tail) :] != tail:
        raise StrictSchemaError(f"{path}: invalid shape")
    if value.size == 0 or (np.issubdtype(dtype, np.floating) and not np.isfinite(value).all()):
        raise StrictSchemaError(f"{path}: invalid values")
    result = np.ascontiguousarray(value).copy()
    result.flags.writeable = False
    return result


def _bool_tuple(value: Any, path: str, size: int) -> tuple[bool, ...]:
    rows = tuple(value)
    if len(rows) != size or any(type(item) is not bool for item in rows):
        raise StrictSchemaError(f"{path}: expected {size} booleans")
    return rows


def _step_tuple(value: Any, size: int) -> tuple[int, ...]:
    rows = tuple(value)
    if len(rows) != size or any(type(item) is not int or item < 0 for item in rows):
        raise StrictSchemaError(f"robolab_snapshot.step_indices: expected {size} nonnegative integers")
    return rows


@dataclass(frozen=True)
class RoboLabRuntimeSnapshot:
    instruction: str
    external_rgb: np.ndarray
    wrist_rgb: np.ndarray
    arm_joint_position: np.ndarray
    gripper_position: np.ndarray
    step_indices: tuple[int, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    success: tuple[bool, ...]
    frozen: tuple[bool, ...]

    def __post_init__(self) -> None:
        if type(self.instruction) is not str or not self.instruction:
            raise StrictSchemaError("robolab_snapshot.instruction: expected nonempty string")
        external = _array(
            self.external_rgb,
            "robolab_snapshot.external_rgb",
            dtype=np.dtype("uint8"),
            ndim=4,
            tail=(720, 1280, 3),
        )
        size = external.shape[0]
        wrist = _array(
            self.wrist_rgb,
            "robolab_snapshot.wrist_rgb",
            dtype=np.dtype("uint8"),
            ndim=4,
            tail=(720, 1280, 3),
        )
        arm = _array(
            self.arm_joint_position,
            "robolab_snapshot.arm_joint_position",
            dtype=np.dtype("float32"),
            ndim=2,
            tail=(7,),
        )
        gripper = _array(
            self.gripper_position,
            "robolab_snapshot.gripper_position",
            dtype=np.dtype("float32"),
            ndim=2,
            tail=(1,),
        )
        if wrist.shape[0] != size or arm.shape[0] != size or gripper.shape[0] != size:
            raise StrictSchemaError("robolab_snapshot: batch dimensions differ")
        steps = _step_tuple(self.step_indices, size)
        terminated = _bool_tuple(self.terminated, "robolab_snapshot.terminated", size)
        truncated = _bool_tuple(self.truncated, "robolab_snapshot.truncated", size)
        success = _bool_tuple(self.success, "robolab_snapshot.success", size)
        frozen = _bool_tuple(self.frozen, "robolab_snapshot.frozen", size)
        for index in range(size):
            if terminated[index] and truncated[index]:
                raise StrictSchemaError("robolab_snapshot: slot is both terminated and truncated")
            if success[index] != terminated[index]:
                raise StrictSchemaError("robolab_snapshot: success and termination differ")
            if frozen[index] != (terminated[index] or truncated[index]):
                raise StrictSchemaError("robolab_snapshot: frozen state differs")
        object.__setattr__(self, "external_rgb", external)
        object.__setattr__(self, "wrist_rgb", wrist)
        object.__setattr__(self, "arm_joint_position", arm)
        object.__setattr__(self, "gripper_position", gripper)
        object.__setattr__(self, "step_indices", steps)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "frozen", frozen)

    @property
    def size(self) -> int:
        return int(self.external_rgb.shape[0])


class RoboLabRuntime(Protocol):
    def load(self, batch: RoboLabBatch) -> RoboLabRuntimeSnapshot: ...

    def observe(self) -> RoboLabRuntimeSnapshot: ...

    def step(self, actions: np.ndarray) -> RoboLabRuntimeSnapshot: ...

    def finish(self) -> None: ...

    def close(self) -> None: ...


def validate_robolab_profile(profile: Profile) -> None:
    if not isinstance(profile, Profile):
        raise StrictSchemaError("robolab profile: expected Profile")
    if profile.environment.suite != ROBOLAB_SUITE:
        raise StrictSchemaError("robolab profile: suite differs")
    if profile.policy.action_spec != DROID_ACTION_SPEC:
        raise StrictSchemaError("robolab profile: action specification differs")
    cameras = {item.name: item for item in profile.environment.cameras}
    if set(cameras) != {"external", "wrist"}:
        raise StrictSchemaError("robolab profile: camera set differs")
    expected_frames = {"external": "over_shoulder_left_camera", "wrist": "wrist_cam"}
    for name, frame in expected_frames.items():
        item = cameras[name]
        if (
            item.frame_id != frame
            or item.width != 1280
            or item.height != 720
            or item.has_depth
            or item.optical_convention != "opencv_rdf"
        ):
            raise StrictSchemaError(f"robolab profile: {name} camera differs")
    states = {item.name: item for item in profile.environment.robot_state}
    if states != {"arm_joint_position": _ARM_SPEC, "gripper_position": _GRIPPER_SPEC}:
        raise StrictSchemaError("robolab profile: robot state differs")


class RoboLab120Worker:
    def __init__(self, config: RoboLabAppConfig, runtime: RoboLabRuntime) -> None:
        if not isinstance(config, RoboLabAppConfig):
            raise StrictSchemaError("robolab worker: expected RoboLabAppConfig")
        if config.source_commit != ROBOLAB_SOURCE_COMMIT:
            raise StrictSchemaError("robolab worker: source commit differs")
        if config.asset_lock_sha256 != ROBOLAB_ASSET_TREE_SHA256:
            raise StrictSchemaError("robolab worker: asset tree differs")
        profile = Profile.from_mapping(config.static_profile)
        if profile.resolved_hash() != config.static_profile_sha256:
            raise StrictSchemaError("robolab worker: profile hash differs")
        validate_robolab_profile(profile)
        self.config = config
        self.profile = profile
        self.runtime = runtime
        self.batch: RoboLabBatch | None = None
        self.snapshot: RoboLabRuntimeSnapshot | None = None
        self.closed = False

    def _active(self) -> tuple[RoboLabBatch, RoboLabRuntimeSnapshot]:
        if self.closed:
            raise StrictSchemaError("robolab worker: closed")
        if self.batch is None or self.snapshot is None:
            raise StrictSchemaError("robolab worker: no batch is loaded")
        return self.batch, self.snapshot

    @staticmethod
    def _status(snapshot: RoboLabRuntimeSnapshot, batch: RoboLabBatch, index: int) -> RoboLabPrivateStatus:
        horizon = batch.episodes[index].horizon
        forced = snapshot.step_indices[index] >= horizon and not snapshot.frozen[index]
        return RoboLabPrivateStatus(
            episode_id=batch.episode_ids[index],
            step_index=min(snapshot.step_indices[index], horizon),
            terminated=snapshot.terminated[index],
            truncated=snapshot.truncated[index] or forced,
            success=snapshot.success[index],
            frozen=snapshot.frozen[index] or forced,
        )

    def load_batch(self, batch: RoboLabBatch) -> dict[str, Any]:
        if self.closed or self.batch is not None:
            raise StrictSchemaError("robolab worker: batch state differs")
        if not isinstance(batch, RoboLabBatch) or len(batch.episodes) > self.config.vector_batch_size:
            raise StrictSchemaError("robolab worker: batch exceeds application capacity")
        if batch.key.scenario_id not in ROBOLAB_INSTRUCTION_TYPES:
            raise StrictSchemaError("robolab worker: instruction type differs")
        snapshot = self.runtime.load(batch)
        if snapshot.size != len(batch.episodes) or any(snapshot.step_indices):
            raise StrictSchemaError("robolab worker: reset snapshot differs")
        self.batch = batch
        self.snapshot = snapshot
        observations = self.observe_batch()
        observations.validate_batch(batch)
        return {"batch_id": batch.batch_id, "episode_ids": list(batch.episode_ids), "loaded": True}

    def observe_batch(self) -> RoboLabObservationBatch:
        batch, _ = self._active()
        snapshot = self.runtime.observe()
        if snapshot.size != len(batch.episodes):
            raise StrictSchemaError("robolab worker: observation batch size differs")
        self.snapshot = snapshot
        observations = []
        for index, episode in enumerate(batch.episodes):
            status = self._status(snapshot, batch, index)
            observation = FairObservation(
                episode_id=episode.artifact_id(),
                step_index=status.step_index,
                timestamp_ns=status.step_index * CONTROL_PERIOD_NS,
                instruction=snapshot.instruction,
                cameras={
                    "external": CameraObservation(
                        frame_id="over_shoulder_left_camera",
                        optical_convention="opencv_rdf",
                        rgb=snapshot.external_rgb[index],
                        depth_m=None,
                        depth_valid=None,
                        intrinsics=None,
                        camera_to_world=None,
                    ),
                    "wrist": CameraObservation(
                        frame_id="wrist_cam",
                        optical_convention="opencv_rdf",
                        rgb=snapshot.wrist_rgb[index],
                        depth_m=None,
                        depth_valid=None,
                        intrinsics=None,
                        camera_to_world=None,
                    ),
                },
                proprioception=RobotProprioception(
                    (
                        RobotStateVector(_ARM_SPEC, snapshot.arm_joint_position[index]),
                        RobotStateVector(_GRIPPER_SPEC, snapshot.gripper_position[index]),
                    )
                ),
            )
            observations.append(self.profile.validate_observation(observation))
        result = RoboLabObservationBatch(batch.batch_id, tuple(observations))
        result.validate_batch(batch)
        return result

    def private_status_batch(self) -> RoboLabPrivateStatusBatch:
        batch, snapshot = self._active()
        result = RoboLabPrivateStatusBatch(
            batch.batch_id,
            tuple(self._status(snapshot, batch, index) for index in range(len(batch.episodes))),
        )
        result.validate_batch(batch)
        return result

    def apply_batch(self, action_batch: RoboLabActionBatch) -> dict[str, Any]:
        batch, snapshot = self._active()
        action_batch.validate_batch(batch)
        statuses = self.private_status_batch().statuses
        active_indices = [index for index, status in enumerate(statuses) if not status.frozen]
        if not active_indices:
            raise StrictSchemaError("robolab worker: batch is already complete")
        counts = {action_batch.actions[index].execution_count for index in active_indices}
        if len(counts) != 1:
            raise StrictSchemaError("robolab worker: active execution counts differ")
        count = counts.pop()
        for index, action in enumerate(action_batch.actions):
            if action.spec != DROID_ACTION_SPEC:
                raise StrictSchemaError("robolab worker: action specification differs")
            if not statuses[index].frozen and action.start_step != statuses[index].step_index:
                raise StrictSchemaError("robolab worker: action start step differs")
        for offset in range(count):
            statuses = self.private_status_batch().statuses
            native = np.zeros((len(batch.episodes), 8), dtype=np.float32)
            for index, status in enumerate(statuses):
                if status.frozen:
                    continue
                native[index] = action_batch.actions[index].values[offset]
            snapshot = self.runtime.step(native)
            if snapshot.size != len(batch.episodes):
                raise StrictSchemaError("robolab worker: stepped batch size differs")
            self.snapshot = snapshot
            if all(item.frozen for item in self.private_status_batch().statuses):
                break
        return {
            "batch_id": batch.batch_id,
            "episode_ids": list(batch.episode_ids),
            "applied": True,
        }

    def finish_batch(self) -> dict[str, Any]:
        batch, _ = self._active()
        if not all(item.frozen for item in self.private_status_batch().statuses):
            raise StrictSchemaError("robolab worker: cannot finish an active batch")
        self.runtime.finish()
        self.batch = None
        self.snapshot = None
        return {"batch_id": batch.batch_id, "episode_ids": list(batch.episode_ids), "finished": True}

    def candidate_barrier(self, barrier_id: str) -> dict[str, Any]:
        if type(barrier_id) is not str or not barrier_id or self.batch is not None or self.closed:
            raise StrictSchemaError("robolab worker: candidate barrier state differs")
        return {"barrier_id": barrier_id, "ready": True}

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self.runtime.close()
            self.batch = None
            self.snapshot = None
            self.closed = True
        return {"closed": True}


class IsaacRoboLabRuntime:
    def __init__(self, source_root: Path, runtime_root: Path) -> None:
        source = Path(source_root).resolve()
        if not (source / "robolab" / "registrations" / "droid" / "auto_env_registrations_jointpos.py").is_file():
            raise RuntimeError("RoboLab source checkout is incomplete")
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != ROBOLAB_SOURCE_COMMIT:
            raise RuntimeError("RoboLab source revision differs")
        root = Path(runtime_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(source))
        from isaaclab.app import AppLauncher

        self._launcher = AppLauncher(headless=True, enable_cameras=True, device="cuda:0")
        self._app = self._launcher.app
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT
        import torch

        self._set_output_dir = set_output_dir
        self._get_envs = get_envs
        self._create_env = create_env
        self._register = auto_register_droid_envs
        self._cameras = WRIST_LEFT
        self._torch = torch
        self._runtime_root = root
        self._counter = 0
        self._registered_envs: dict[str, str] = {}
        self._env: Any = None
        self._env_cfg: Any = None
        self._obs: Any = None

    def load(self, batch: RoboLabBatch) -> RoboLabRuntimeSnapshot:
        if self._env is not None:
            raise RuntimeError("RoboLab runtime already has a loaded batch")
        self._counter += 1
        output = self._runtime_root / f"batch-{self._counter:06d}-{batch.batch_id[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        self._set_output_dir(str(output))
        if batch.key.task_id not in self._registered_envs:
            self._register(task=[batch.key.task_id], cameras=self._cameras)
            env_names = self._get_envs(task=batch.key.task_id)
            if len(env_names) != 1:
                raise RuntimeError("RoboLab task registration is ambiguous")
            self._registered_envs[batch.key.task_id] = env_names[0]
        self._env, self._env_cfg = self._create_env(
            self._registered_envs[batch.key.task_id],
            device="cuda:0",
            seed=batch.key.environment_seed,
            num_envs=len(batch.episodes),
            instruction_type=batch.key.scenario_id,
            policy="robot_auto_evolve",
            renderer="realtime",
            rendering_mode="performance",
        )
        if int(self._env.max_episode_length) != batch.key.horizon:
            self.finish()
            raise RuntimeError("RoboLab task horizon differs from the episode plan")
        self._obs, _ = self._env.reset()
        self._obs, _ = self._env.reset()
        return self._snapshot()

    @staticmethod
    def _numpy(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
        return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=dtype)

    def _snapshot(self) -> RoboLabRuntimeSnapshot:
        if self._env is None or self._env_cfg is None or self._obs is None:
            raise RuntimeError("RoboLab runtime has no loaded batch")
        results = self._env.get_env_results()
        frozen = tuple(bool(item) for item in self._env._frozen_envs.detach().cpu().tolist())
        steps = []
        terminated = []
        truncated = []
        success = []
        buffers = self._env.episode_length_buf.detach().cpu().tolist()
        for index, item in enumerate(results):
            result = item["success"]
            steps.append(int(item["step"] if item["step"] is not None else buffers[index]))
            success.append(result is True)
            terminated.append(result is True)
            truncated.append(frozen[index] and result is False)
        return RoboLabRuntimeSnapshot(
            instruction=str(self._env_cfg.instruction),
            external_rgb=self._numpy(
                self._obs["image_obs"]["over_shoulder_left_camera"], np.dtype("uint8")
            ),
            wrist_rgb=self._numpy(self._obs["image_obs"]["wrist_cam"], np.dtype("uint8")),
            arm_joint_position=self._numpy(
                self._obs["proprio_obs"]["arm_joint_pos"], np.dtype("float32")
            ),
            gripper_position=self._numpy(
                self._obs["proprio_obs"]["gripper_pos"], np.dtype("float32")
            ),
            step_indices=tuple(steps),
            terminated=tuple(terminated),
            truncated=tuple(truncated),
            success=tuple(success),
            frozen=frozen,
        )

    def observe(self) -> RoboLabRuntimeSnapshot:
        return self._snapshot()

    def step(self, actions: np.ndarray) -> RoboLabRuntimeSnapshot:
        if self._env is None:
            raise RuntimeError("RoboLab runtime has no loaded batch")
        native = _array(actions, "robolab_runtime.actions", dtype=np.dtype("float32"), ndim=2, tail=(8,))
        if native.shape[0] != self._env.num_envs:
            raise RuntimeError("RoboLab action batch size differs")
        tensor = self._torch.as_tensor(native, device=self._env.device, dtype=self._torch.float32)
        self._obs, _, _, _, _ = self._env.step(tensor)
        return self._snapshot()

    def finish(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._env_cfg = None
        self._obs = None

    def close(self) -> None:
        self.finish()
        self._app.close()


class RoboLabWorkerService:
    def __init__(
        self,
        runtime_factory: Callable[[RoboLabAppConfig], RoboLabRuntime],
    ) -> None:
        self.runtime_factory = runtime_factory
        self.worker: RoboLab120Worker | None = None
        self.sequence = 0

    def dispatch(self, request: RoboLabRpcRequest) -> RoboLabRpcResponse:
        if request.sequence != self.sequence + 1:
            raise StrictSchemaError("robolab worker service: sequence differs")
        self.sequence = request.sequence
        operation = request.operation
        if operation == "initialize_app":
            if self.worker is not None:
                raise StrictSchemaError("robolab worker service: already initialized")
            config = RoboLabAppConfig.from_mapping(request.payload)
            self.worker = RoboLab120Worker(config, self.runtime_factory(config))
            result: Mapping[str, Any] = {"ready": True}
        else:
            worker = self.worker
            if worker is None:
                raise StrictSchemaError("robolab worker service: not initialized")
            if operation == "load_batch":
                result = worker.load_batch(RoboLabBatch.from_mapping(request.payload))
            elif operation == "observe_batch":
                result = worker.observe_batch().to_mapping()
            elif operation == "apply_batch":
                result = worker.apply_batch(RoboLabActionBatch.from_mapping(request.payload))
            elif operation == "private_status_batch":
                result = worker.private_status_batch().to_mapping()
            elif operation == "finish_batch":
                result = worker.finish_batch()
            elif operation == "candidate_barrier":
                result = worker.candidate_barrier(request.payload["barrier_id"])
            elif operation == "close":
                result = worker.close()
            else:
                raise StrictSchemaError("robolab worker service: unknown operation")
        return RoboLabRpcResponse.success(request, result)


def serve(source_root: Path, runtime_root: Path) -> int:
    protocol_out = os.dup(sys.stdout.fileno())
    sys.stdout = sys.stderr
    service = RoboLabWorkerService(lambda _: IsaacRoboLabRuntime(source_root, runtime_root))
    while True:
        try:
            request = read_robolab_request(sys.stdin.fileno())
        except EOFError:
            if service.worker is not None:
                service.worker.close()
            return 0
        try:
            response = service.dispatch(request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = RoboLabRpcResponse.failure(request, f"{type(exc).__name__}: {exc}")
        write_robolab_response(protocol_out, response)
        if request.operation == "close" and response.ok:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args(argv)
    return serve(args.source_root, args.runtime_root)


if __name__ == "__main__":
    raise SystemExit(main())
