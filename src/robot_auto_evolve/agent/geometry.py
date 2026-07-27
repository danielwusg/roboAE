"""Turn a pixel into a 3D point, and back.

Pure arithmetic on the `CameraObservation` the scaffold receives. No simulator access, no hidden
state, nothing privileged.

WHAT A CAMERA CARRIES WHEN THE ROUTE HAS 3D ON
    camera.rgb              uint8   [H, W, 3]
    camera.depth_m          float32 [H, W]     distance in METRES, pixel-aligned with .rgb
    camera.depth_valid      bool    [H, W]     False where the depth reading is unusable
    camera.intrinsics       float32 [3, 3]     [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera.camera_to_world  float32 [4, 4]     camera frame -> the robot's own reference frame

`has_3d(camera)` is True only when all of those are present. On a route without 3D they are all
None and every function here returns None rather than raising.

THE FRAME
    "camera_to_world" is named for the field in the observation schema. What it maps to is the
    SAME frame the benchmark reports the end effector in -- the `reference_frame` on the
    `eef_pose` entry of `observation.proprioception`. On some routes that frame is the simulator
    world, on others the robot base. Either way a point out of `pixel_to_world` can be compared
    directly with the end-effector position and handed straight to
    `robot_auto_evolve.agent.motion`.

PIXEL COORDINATES
    (u, v) are (column, row) into the arrays exactly as stored: u indexes the second axis of
    `rgb`, v indexes the first. The detector and the pointer return boxes and points in these
    same coordinates, so a detection can be fed straight in.

THE TWO CAMERA CONVENTIONS
    `camera.optical_convention` says how the camera axes and the stored rows relate:
      "opencv_rdf" -- x right, y down, z forward; array row 0 is the TOP of the picture.
      "opengl_rub" -- x right, y up,   z backward; array row 0 is the BOTTOM of the picture.
    MuJoCo hands back a bottom-up buffer; SAPIEN and dm_control hand back top-down buffers. Both
    are handled here, which is the point of this module.

EXAMPLE
    from robot_auto_evolve.agent.geometry import has_3d, pixel_to_world

    camera = request.observation.cameras["main"]
    if has_3d(camera):
        hit = pixel_to_world(camera, u, v)         # None if that pixel has no valid depth
        if hit is not None:
            ...                                     # hit is (x, y, z) in the robot's frame
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "depth_at",
    "has_3d",
    "pixel_to_camera",
    "pixel_to_world",
    "point_cloud",
    "world_to_camera",
    "world_to_pixel",
]


def has_3d(camera: Any) -> bool:
    """True when this camera carries everything needed to turn a pixel into a 3D point."""
    return (
        getattr(camera, "depth_m", None) is not None
        and getattr(camera, "depth_valid", None) is not None
        and getattr(camera, "intrinsics", None) is not None
        and getattr(camera, "camera_to_world", None) is not None
    )


def _require_3d(camera: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not has_3d(camera):
        raise ValueError("camera has no 3D sensing on this route (depth/intrinsics/pose are None)")
    return (
        np.asarray(camera.depth_m, dtype=np.float64),
        np.asarray(camera.depth_valid, dtype=bool),
        np.asarray(camera.intrinsics, dtype=np.float64),
        np.asarray(camera.camera_to_world, dtype=np.float64),
        str(camera.optical_convention),
    )


def depth_at(camera: Any, u: float, v: float, radius: int = 2) -> float | None:
    """Metres to the surface seen at pixel (u, v), or None if nothing valid is nearby.

    A single depth pixel can land on an object edge and read the background. `radius` widens
    the read to a square window and takes the MEDIAN of the valid readings inside it, which is
    what you almost always want. radius=0 reads the one pixel.
    """
    if not has_3d(camera):
        return None
    depth = np.asarray(camera.depth_m, dtype=np.float64)
    valid = np.asarray(camera.depth_valid, dtype=bool)
    height, width = depth.shape
    column = int(round(float(u)))
    row = int(round(float(v)))
    if not (0 <= column < width and 0 <= row < height):
        return None
    reach = max(0, int(radius))
    window = depth[
        max(0, row - reach) : min(height, row + reach + 1),
        max(0, column - reach) : min(width, column + reach + 1),
    ]
    window_valid = valid[
        max(0, row - reach) : min(height, row + reach + 1),
        max(0, column - reach) : min(width, column + reach + 1),
    ]
    usable = window[window_valid & np.isfinite(window) & (window > 0.0)]
    if usable.size == 0:
        return None
    return float(np.median(usable))


def pixel_to_camera(camera: Any, u: float, v: float, depth: float) -> np.ndarray:
    """(u, v) plus a distance in metres -> a point in the CAMERA's own frame."""
    _, _, intrinsics, _, convention = _require_3d(camera)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    distance = float(depth)
    x = (float(u) - cx) * distance / fx
    y = (float(v) - cy) * distance / fy
    if convention == "opencv_rdf":
        # x right, y down, z forward; v grows downward in the picture, which is the array order.
        return np.array((x, y, distance), dtype=np.float64)
    if convention == "opengl_rub":
        # x right, y up, z backward; array row 0 is the picture's bottom, so a growing v is +y,
        # and "in front of the camera" is negative z.
        return np.array((x, y, -distance), dtype=np.float64)
    raise ValueError(f"unsupported optical convention {convention!r}")


def pixel_to_world(camera: Any, u: float, v: float, depth: float | None = None, radius: int = 2) -> np.ndarray | None:
    """(u, v) -> a 3D point in the robot's own reference frame, or None if depth is unusable.

    Pass `depth` if you already have the distance; otherwise it is read from the camera with
    `depth_at(camera, u, v, radius)`.
    """
    if not has_3d(camera):
        return None
    distance = depth_at(camera, u, v, radius) if depth is None else float(depth)
    if distance is None or not np.isfinite(distance) or distance <= 0.0:
        return None
    point = pixel_to_camera(camera, u, v, distance)
    transform = np.asarray(camera.camera_to_world, dtype=np.float64)
    world = transform @ np.array((point[0], point[1], point[2], 1.0), dtype=np.float64)
    return np.asarray(world[:3], dtype=np.float64)


def world_to_camera(camera: Any, point: Any) -> np.ndarray:
    """A point in the robot's reference frame -> the same point in the camera's frame."""
    _, _, _, transform, _ = _require_3d(camera)
    value = np.asarray(point, dtype=np.float64).reshape(3)
    inverse = np.linalg.inv(transform)
    camera_point = inverse @ np.array((value[0], value[1], value[2], 1.0), dtype=np.float64)
    return np.asarray(camera_point[:3], dtype=np.float64)


def world_to_pixel(camera: Any, point: Any) -> tuple[float, float] | None:
    """A 3D point -> the (u, v) pixel it appears at, or None if it is behind the camera.

    The returned pixel may fall outside the image; check it against the image size yourself if
    that matters. This is the inverse of `pixel_to_world` and is the cheapest way to check that
    a 3D point you computed really is where you think it is.
    """
    if not has_3d(camera):
        return None
    _, _, intrinsics, _, convention = _require_3d(camera)
    local = world_to_camera(camera, point)
    if convention == "opencv_rdf":
        distance = float(local[2])
    elif convention == "opengl_rub":
        distance = -float(local[2])
    else:
        raise ValueError(f"unsupported optical convention {convention!r}")
    if not np.isfinite(distance) or distance <= 1e-9:
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return (float(local[0]) * fx / distance + cx, float(local[1]) * fy / distance + cy)


def point_cloud(camera: Any, stride: int = 4, mask: Any = None) -> np.ndarray | None:
    """Every valid pixel as a 3D point in the robot's frame, shape [N, 3].

    `stride` subsamples the image (stride=4 keeps one pixel in sixteen), which is usually
    plenty and much cheaper. `mask` is an optional bool [H, W] -- for example a segmentation
    mask -- restricting the cloud to one object.
    """
    if not has_3d(camera):
        return None
    depth, valid, intrinsics, transform, convention = _require_3d(camera)
    step = max(1, int(stride))
    height, width = depth.shape
    rows = np.arange(0, height, step)
    columns = np.arange(0, width, step)
    grid_v, grid_u = np.meshgrid(rows, columns, indexing="ij")
    sampled_depth = depth[::step, ::step]
    keep = valid[::step, ::step] & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if mask is not None:
        keep = keep & np.asarray(mask, dtype=bool)[::step, ::step]
    if not keep.any():
        return np.zeros((0, 3), dtype=np.float64)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    d = sampled_depth[keep]
    x = (grid_u[keep] - cx) * d / fx
    y = (grid_v[keep] - cy) * d / fy
    z = d if convention == "opencv_rdf" else -d
    local = np.stack((x, y, z, np.ones_like(d)), axis=1)
    world = local @ transform.T
    return np.ascontiguousarray(world[:, :3])
