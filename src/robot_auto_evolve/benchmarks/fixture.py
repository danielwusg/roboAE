from __future__ import annotations

import numpy as np

from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import (
    CameraObservation,
    CanonicalActionChunk,
    CanonicalActionSpec,
    FairObservation,
    RobotProprioception,
    RobotStateVector,
    StrictSchemaError,
)
from robot_auto_evolve.provenance import EpisodeKey


FIXTURE_ACTION_SPEC = CanonicalActionSpec(
    arm_names=("arm",),
    channel_names=("x",),
    channel_semantics=("delta",),
    coordinate_frame="robot_base",
    translation_unit="meter",
    rotation_representation="none",
    quaternion_order="none",
    gripper_convention="none",
    value_encoding="physical",
    controller_output_scale=(),
    control_period_s=0.05,
)


class FixtureWorker:
    ACTION_SPEC = FIXTURE_ACTION_SPEC

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("fixture worker requires Profile and EpisodeKey")
        if profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("fixture worker action spec differs from profile")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("fixture render_gpu_id must be nonnegative")
        self.profile = profile
        self.episode = episode
        self.step_index = 0
        self.success = False
        self.active = False

    def reset(self) -> None:
        if self.active:
            raise RuntimeError("fixture worker is single-use")
        self.active = True

    def observe(self) -> FairObservation:
        if not self.active:
            raise RuntimeError("fixture worker is not active")
        cameras = {}
        for spec in self.profile.environment.cameras:
            rgb = np.full((spec.height, spec.width, 3), self.step_index % 255, dtype=np.uint8)
            depth = np.ones((spec.height, spec.width), dtype=np.float32) if spec.has_depth else None
            cameras[spec.name] = CameraObservation(
                frame_id=spec.frame_id,
                optical_convention=spec.optical_convention,
                rgb=rgb,
                depth_m=depth,
                depth_valid=None if depth is None else np.ones(depth.shape, dtype=np.bool_),
                intrinsics=None if depth is None else np.eye(3, dtype=np.float32),
                camera_to_world=None if depth is None else np.eye(4, dtype=np.float32),
            )
        vectors = tuple(
            RobotStateVector(spec, np.zeros(len(spec.component_names), dtype=np.float32))
            for spec in self.profile.environment.robot_state
        )
        return FairObservation(
            episode_id=self.episode.artifact_id(),
            step_index=self.step_index,
            timestamp_ns=self.step_index * 50_000_000,
            instruction=self.episode.task_id,
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if not self.active or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("fixture action differs from contract")
        if action.horizon != 1 or action.execution_count != 1 or action.start_step != self.step_index:
            raise StrictSchemaError("fixture worker requires one current-step action")
        self.success = self.success or bool(action.values[0, 0] > 0)
        self.step_index += 1

    def private_success(self) -> bool:
        if not self.active:
            raise RuntimeError("fixture worker is not active")
        return self.success

    def close(self) -> None:
        self.active = False


class PrivateMetricsFixtureWorker(FixtureWorker):
    def private_metrics(self) -> dict[str, float | bool]:
        if not self.active:
            raise RuntimeError("fixture worker is not active")
        return {"evaluator_only_score": 0.25, "success": self.success}
