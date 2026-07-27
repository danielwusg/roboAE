"""Per-simulator 3D sensing: metric depth, lens parameters, and camera pose (Revision 2).

Each benchmark family renders depth differently and describes its camera differently. This
module turns each of them into the SAME four fields the observation schema declares, so a
scaffold sees one story regardless of route:

    depth_m         float32 [H, W]   metres, pixel-aligned with the camera's own .rgb array
    depth_valid     bool    [H, W]   False where the reading is the far plane or is not finite
    intrinsics      float32 [3, 3]   [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_to_world float32 [4, 4]   camera frame -> the frame the END EFFECTOR is reported in

The last line is the one that is easy to get wrong, so it is stated as a rule:

    `camera_to_world` maps into whatever frame this benchmark reports the arm in, NOT into some
    universal world frame.

LIBERO, LIBERO-Pro, RoboCerebra and VLABench report the end effector in the simulator's world
frame, so there `camera_to_world` really is camera-to-world. SimplerEnv reports the end
effector RELATIVE TO THE ROBOT BASE, so there this matrix is camera-to-robot-base -- built by
composing the camera's world pose with the inverse of the base's world pose. Getting this wrong
would put every computed target confidently in the wrong place while nothing crashed, which is
why it is centralised here instead of being repeated in six workers.

The paired `optical_convention` (declared per camera in the route profile and honoured by
`robot_auto_evolve.agent.geometry`) records how the stored rows relate to the camera axes:

    robosuite / MuJoCo (LIBERO family, RoboCasa365)  ->  "opengl_rub"
        robosuite is configured with IMAGE_CONVENTION="opengl" and does NOT flip the render
        buffer, and MuJoCo's mjr_readPixels fills it bottom-up. So row 0 of the stored array is
        the BOTTOM of the picture, and the description that matches it is the RAW MuJoCo camera
        frame: x right, y up, z backward. (robosuite's own get_camera_extrinsic_matrix applies a
        diag(1, -1, -1) correction to reach the OpenCV frame; that correction belongs with a
        TOP-DOWN image, which is not what is delivered here.) The rgb and depth arrays come out
        of one mjr_readPixels call and get the same (non-)flip, so depth[r, c] and rgb[r, c] are
        always the same pixel.
    SAPIEN / ManiSkill2 (SimplerEnv)                ->  "opencv_rdf"
    dm_control (VLABench)                           ->  "opencv_rdf"
        both hand back top-down images and describe their cameras in the OpenCV convention.

The robosuite line above was MEASURED, not assumed, because getting it wrong is silent. The
measurement is `rev/s22/diagnose_libero_frame.py`: it starts a real LIBERO-Pro episode, drives
the gripper 14 cm straight up with the harness's own movement commands, and looks at which
pixels of the picture actually changed. The arm's pixels changed in rows 191-255 of 256 -- the
bottom of the array. Bottom-up + the raw MuJoCo pose predicts the gripper at rows 195 -> 243,
inside that region; top-down + the corrected pose predicts rows 61 -> 13, in a part of the image
that did not change at all. `rev/s22/check_depth_gpu.py` re-confirms it with no ground truth by
reconstructing the same surface point from two different cameras and requiring the answers to
agree (they agree for 79-81% of mutually visible samples under this pairing, and for 2% under
the other one).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_auto_evolve.protocol import StrictSchemaError


# A robosuite depth buffer value of 1.0 is the far clipping plane -- nothing was hit. Values that
# close to 1 unproject to tens of metres and would poison any median, so they are marked invalid.
ROBOSUITE_FAR_LIMIT = 0.9999
# Any reading beyond this many metres is treated as "no surface here" on every family. Every
# scene in this project is a table top or a kitchen; 20 m is the sky.
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
    """MuJoCo/robosuite depth buffer -> metres + lens + camera pose, in "opencv_rdf".

    `raw_depth` is the `<camera>_depth` observable robosuite produces: the normalised OpenGL
    depth buffer in [0, 1], shaped [H, W, 1] or [H, W], and carrying exactly the same row order
    as the matching `<camera>_image`. `world_to_reference` optionally re-expresses the camera
    pose in another frame (unused by the LIBERO family, whose arm is reported in world).
    """
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
    # The standard OpenGL depth-buffer linearisation, identical to
    # robosuite.utils.camera_utils.get_real_depth_map.
    metres = near / np.maximum(1.0 - clipped * (1.0 - near / far), 1e-12)
    valid = clipped < ROBOSUITE_FAR_LIMIT
    camera_id = model.camera_name2id(camera_name)
    fovy = float(model.cam_fovy[camera_id])
    focal = 0.5 * float(height) / np.tan(fovy * np.pi / 360.0)
    intrinsics = np.array(
        ((focal, 0.0, float(width) / 2.0), (0.0, focal, float(height) / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    # The RAW MuJoCo camera frame (x right, y up, z backward), which is the one that matches the
    # bottom-up array robosuite delivers here. Deliberately NOT robosuite's
    # get_camera_extrinsic_matrix, which post-multiplies by diag(1, -1, -1) for a top-down image.
    # See the module docstring for the measurement that settles this.
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
    """ManiSkill2 (SAPIEN) rgbd observation -> metres + lens + camera pose, in "opencv_rdf".

    `image_entry` is `obs["image"][<camera>]` and `camera_param` is
    `obs["camera_param"][<camera>]`; both are already present because every SimplerEnv route in
    this project builds its environment with `obs_mode="rgbd"`. Nothing extra is rendered.
    `world_to_reference` is the matrix that takes a world point into the frame the arm is
    reported in -- on SimplerEnv that is the inverse of the robot base pose, and it is required.
    """
    if "depth" not in image_entry:
        raise StrictSchemaError("ManiSkill2 observation carries no depth (obs_mode must be rgbd)")
    depth = np.asarray(image_entry["depth"], dtype=np.float64)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise StrictSchemaError("ManiSkill2 depth must be [H, W] or [H, W, 1]")
    intrinsics = np.asarray(camera_param["intrinsic_cv"], dtype=np.float64).reshape(3, 3)
    # extrinsic_cv maps WORLD -> CAMERA in the OpenCV convention; we want the other direction.
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
    """dm_control (VLABench) depth render -> metres + lens + camera pose, in "opencv_rdf".

    VLABench's own `DMEnv.get_observation` already renders depth and computes the two matrices
    for every camera on every call, so this reads what is there rather than rendering anything
    new. Its `get_camera_matrix` post-multiplies the MuJoCo camera rotation by a 180-degree turn
    about x, which converts the raw RUB axes to OpenCV RDF -- matching dm_control's top-down
    image order.
    """
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
    """The 4x4 that takes a WORLD point into the frame at (position, rotation_matrix)."""
    return np.linalg.inv(_pose_matrix(position, rotation_matrix))
