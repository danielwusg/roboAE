from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np

from robot_auto_evolve.protocol import FairObservation, StrictSchemaError

from .molmoact2 import MOLMOACT2_LIBERO_ACTION_SPEC
from .workers import LiberoWorker


MOLMOACT2_LIBERO_PROTOCOLS = MappingProxyType(
    {
        "libero_spatial": MappingProxyType(
            {
                "molmoact2_libero_spatial_transfer_v1": 280,
                "molmoact2_libero_spatial_smoke_v1": 11,
                "molmoact2_libero_spatial_canonical_50_per_task_v1": 280,
            }
        ),
        "libero_object": MappingProxyType(
            {
                "molmoact2_libero_object_transfer_v1": 280,
                "molmoact2_libero_object_smoke_v1": 11,
                "molmoact2_libero_object_canonical_50_per_task_v1": 280,
            }
        ),
        "libero_goal": MappingProxyType(
            {
                "molmoact2_libero_goal_transfer_v1": 300,
                "molmoact2_libero_goal_smoke_v1": 11,
                "molmoact2_libero_goal_canonical_50_per_task_v1": 300,
            }
        ),
        "libero_10": MappingProxyType(
            {
                "molmoact2_libero_10_transfer_v1": 520,
                "molmoact2_libero_10_smoke_v1": 11,
                "molmoact2_libero_10_canonical_50_per_task_v1": 520,
            }
        ),
    }
)


class MolmoAct2LiberoWorker(LiberoWorker):
    ACTION_SPEC = MOLMOACT2_LIBERO_ACTION_SPEC
    PROTOCOLS = MOLMOACT2_LIBERO_PROTOCOLS
    SETTLE_STEPS = 50
    USE_DELTA_CONTROL = True

    def _eef_pose(self, raw: dict[str, object]) -> np.ndarray:
        position = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        quaternion = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise StrictSchemaError("MolmoAct2 LIBERO end-effector state shape mismatch")
        return np.ascontiguousarray(np.concatenate((position, quaternion)), dtype=np.float32)

    def observe(self) -> FairObservation:
        observation = super().observe()
        return replace(observation, timestamp_ns=self._step * 50_000_000)
