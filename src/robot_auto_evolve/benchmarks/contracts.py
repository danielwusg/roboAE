from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robot_auto_evolve.protocol import (
    CameraObservation,
    CanonicalActionChunk,
    CanonicalActionSpec,
    FairObservation,
    RobotStateVector,
    StrictSchemaError,
)


class AdapterError(StrictSchemaError):
    pass


def observation(value: FairObservation | Mapping[str, Any]) -> FairObservation:
    if isinstance(value, FairObservation):
        return value
    try:
        return FairObservation.from_mapping(value)
    except (StrictSchemaError, TypeError) as exc:
        raise AdapterError(f"invalid fair observation: {exc}") from exc


def camera(value: FairObservation, name: str) -> CameraObservation:
    try:
        return value.cameras[name]
    except KeyError as exc:
        raise AdapterError(f"missing camera {name!r}") from exc


def state(
    value: FairObservation,
    name: str,
    width: int,
    *,
    quantity: str | None = None,
    representation: str | None = None,
    quaternion_order: str | None = None,
) -> np.ndarray:
    try:
        vector: RobotStateVector = value.proprioception.by_name(name)
    except KeyError as exc:
        raise AdapterError(f"missing robot state {name!r}") from exc
    spec = vector.spec
    if vector.values.shape != (width,):
        raise AdapterError(f"robot state {name!r} must have width {width}")
    checks = {
        "quantity": quantity,
        "representation": representation,
        "quaternion_order": quaternion_order,
    }
    for field, expected in checks.items():
        if expected is not None and getattr(spec, field) != expected:
            raise AdapterError(f"robot state {name!r} {field} must be {expected!r}")
    return vector.values


def policy_actions(value: Any, width: int) -> np.ndarray:
    if isinstance(value, Mapping):
        allowed = {"action", "actions"} & set(value)
        if len(allowed) != 1:
            raise AdapterError("policy response must contain exactly one action field")
        value = value[next(iter(allowed))]
    array = np.asarray(value)
    if array.dtype != np.dtype("float32"):
        array = array.astype(np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != width:
        raise AdapterError(f"policy actions must have shape [T,{width}]")
    if not np.isfinite(array).all():
        raise AdapterError("policy actions contain non-finite values")
    return np.ascontiguousarray(array)


def action_spec(
    *,
    arms: Sequence[str],
    channels: Sequence[str],
    semantics: Sequence[str],
    frame: str,
    rotation: str,
    gripper: str,
    period_s: float | None,
    quaternion_order: str = "none",
    value_encoding: str = "physical",
    controller_output_scale: Sequence[float] = (),
) -> CanonicalActionSpec:
    return CanonicalActionSpec(
        arm_names=tuple(arms),
        channel_names=tuple(channels),
        channel_semantics=tuple(semantics),
        coordinate_frame=frame,
        translation_unit="meter",
        rotation_representation=rotation,
        quaternion_order=quaternion_order,
        gripper_convention=gripper,
        value_encoding=value_encoding,
        controller_output_scale=tuple(controller_output_scale),
        control_period_s=period_s,
    )


def action_chunk(
    values: np.ndarray,
    *,
    spec: CanonicalActionSpec,
    execution_count: int,
    request_id: str,
    session_id: str,
    start_step: int,
) -> CanonicalActionChunk:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return CanonicalActionChunk(
        request_id=request_id,
        session_id=session_id,
        start_step=start_step,
        spec=spec,
        values=array,
        execution_count=execution_count,
    )
