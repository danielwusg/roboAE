from __future__ import annotations

from typing import Any

import numpy as np

from robot_auto_evolve.protocol import StrictSchemaError


MIN_VALUE_RANGE = 16
MIN_STANDARD_DEVIATION = 2.0
MIN_UNIQUE_VALUES = 16
DARK_VALUE = 8
MAX_DARK_FRACTION = 0.98


def rgb_integrity_evidence(value: Any, camera_name: str) -> dict[str, int | float | str | list[int]]:
    if type(camera_name) is not str or not camera_name:
        raise StrictSchemaError("camera name must be a nonempty string")
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise StrictSchemaError(f"camera {camera_name!r} RGB must be a uint8 array")
    if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] < 2 or value.shape[1] < 2:
        raise StrictSchemaError(f"camera {camera_name!r} RGB must be nonempty HWC with three channels")
    histogram = np.bincount(value.reshape(-1), minlength=256)
    minimum = int(np.flatnonzero(histogram)[0])
    maximum = int(np.flatnonzero(histogram)[-1])
    return {
        "camera": camera_name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "min": minimum,
        "max": maximum,
        "value_range": maximum - minimum,
        "mean": float(value.mean(dtype=np.float64)),
        "std": float(value.std(dtype=np.float64)),
        "dark_fraction": float(histogram[:DARK_VALUE].sum() / value.size),
        "unique_values": int(np.count_nonzero(histogram)),
    }


def validate_mujoco_rgb(value: Any, camera_name: str) -> np.ndarray:
    evidence = rgb_integrity_evidence(value, camera_name)
    if (
        evidence["value_range"] < MIN_VALUE_RANGE
        or evidence["std"] < MIN_STANDARD_DEVIATION
        or evidence["dark_fraction"] >= MAX_DARK_FRACTION
        or evidence["unique_values"] < MIN_UNIQUE_VALUES
    ):
        raise StrictSchemaError(
            f"camera {camera_name!r} failed render integrity: range={evidence['value_range']}, "
            f"mean={evidence['mean']:.6f}, std={evidence['std']:.6f}, "
            f"dark_fraction={evidence['dark_fraction']:.6f}, unique_values={evidence['unique_values']}"
        )
    return np.ascontiguousarray(value)
