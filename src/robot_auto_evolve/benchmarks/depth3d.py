
from __future__ import annotations

from typing import Any

import numpy as np

from robot_auto_evolve.protocol import StrictSchemaError


ROBOSUITE_FAR_LIMIT = 0.9999
MAX_USABLE_DEPTH_M = 20.0


def _pose_matrix(position: Any, rotation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


def _finish(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_m = np.ascontiguousarray(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
    if depth_m.shape != expected_shape:
        raise StrictSchemaError(f"depth shape {depth_m.shape} differs from rgb shape {expected_shape}")
    usable = np.ascontiguousarray(
        valid & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= MAX_USABLE_DEPTH_M)
    )
    if usable.shape != expected_shape:
        raise StrictSchemaError("depth validity shape differs from rgb shape")
    matrix = np.ascontiguousarray(intrinsics, dtype=np.float32)
    transform = np.ascontiguousarray(camera_to_world, dtype=np.float32)
    if matrix.shape != (3, 3) or transform.shape != (4, 4):
        raise StrictSchemaError("camera intrinsics or pose have the wrong shape")
    if not np.isfinite(matrix).all() or not np.isfinite(transform).all():
        raise StrictSchemaError("camera intrinsics or pose are not finite")
    return depth_m, usable, matrix, transform


def robosuite_camera_3d(
    sim: Any,
    camera_name: str,
    raw_depth: Any,
    *,
    height: int,
    width: int,
    world_to_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(raw_depth, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise StrictSchemaError("robosuite depth buffer must be [H, W] or [H, W, 1]")
    model = sim.model
    data = sim.data
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    clipped = np.clip(depth, 0.0, 1.0)
    metres = near / np.maximum(1.0 - clipped * (1.0 - near / far), 1e-12)
    valid = clipped < ROBOSUITE_FAR_LIMIT
    camera_id = model.camera_name2id(camera_name)
    fovy = float(model.cam_fovy[camera_id])
    focal = 0.5 * float(height) / np.tan(fovy * np.pi / 360.0)
    intrinsics = np.array(
        ((focal, 0.0, float(width) / 2.0), (0.0, focal, float(height) / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    camera_to_world = _pose_matrix(data.cam_xpos[camera_id], np.asarray(data.cam_xmat[camera_id]).reshape(3, 3))
    if world_to_reference is not None:
        camera_to_world = np.asarray(world_to_reference, dtype=np.float64) @ camera_to_world
    return _finish(metres, valid, intrinsics, camera_to_world, expected_shape=(int(height), int(width)))


def maniskill_camera_3d(
    image_entry: Any,
    camera_param: Any,
    *,
    height: int,
    width: int,
    world_to_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "depth" not in image_entry:
        raise StrictSchemaError("ManiSkill2 observation carries no depth (obs_mode must be rgbd)")
    depth = np.asarray(image_entry["depth"], dtype=np.float64)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise StrictSchemaError("ManiSkill2 depth must be [H, W] or [H, W, 1]")
    intrinsics = np.asarray(camera_param["intrinsic_cv"], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(camera_param["extrinsic_cv"], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3, :] = extrinsic
        extrinsic = homogeneous
    camera_to_world = np.linalg.inv(extrinsic)
    if world_to_reference is not None:
        camera_to_world = np.asarray(world_to_reference, dtype=np.float64) @ camera_to_world
    valid = np.isfinite(depth) & (depth > 0.0)
    return _finish(depth, valid, intrinsics, camera_to_world, expected_shape=(int(height), int(width)))


def dm_control_camera_3d(
    raw_depth: Any,
    intrinsic: Any,
    extrinsic: Any,
    *,
    height: int,
    width: int,
    world_to_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(raw_depth, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise StrictSchemaError("dm_control depth must be [H, W] or [H, W, 1]")
    matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    camera_to_world = np.asarray(extrinsic, dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise StrictSchemaError("dm_control extrinsic must be 4x4")
    if world_to_reference is not None:
        camera_to_world = np.asarray(world_to_reference, dtype=np.float64) @ camera_to_world
    valid = np.isfinite(depth) & (depth > 0.0)
    return _finish(depth, valid, matrix, camera_to_world, expected_shape=(int(height), int(width)))


def inverse_pose(position: Any, rotation_matrix: Any) -> np.ndarray:
    return np.linalg.inv(_pose_matrix(position, rotation_matrix))
