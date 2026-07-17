from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np

from .schema import StrictSchemaError, fields, integer, sequence, string


_ARRAY_EXT = 17
_DTYPES = frozenset(
    {
        "|b1",
        "|u1",
        "<u2",
        "<u4",
        "<u8",
        "|i1",
        "<i2",
        "<i4",
        "<i8",
        "<f2",
        "<f4",
        "<f8",
    }
)


def _native_little_array(value: np.ndarray) -> np.ndarray:
    if value.dtype.kind not in "buif" or value.dtype.fields is not None or value.dtype.subdtype is not None:
        raise StrictSchemaError("message: unsupported numpy dtype")
    dtype = value.dtype
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    result = np.ascontiguousarray(value.astype(dtype, copy=False))
    if result.dtype.str not in _DTYPES:
        raise StrictSchemaError(f"message: unsupported numpy dtype {result.dtype.str}")
    return result


def _array_default(value: Any) -> msgpack.ExtType:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"unsupported type {type(value).__name__}")
    array = _native_little_array(value)
    payload = msgpack.packb(
        {"data": array.tobytes(order="C"), "dtype": array.dtype.str, "shape": list(array.shape)},
        use_bin_type=True,
        strict_types=True,
    )
    return msgpack.ExtType(_ARRAY_EXT, payload)


def _pairs(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise StrictSchemaError("message: expected string mapping key")
        if key in result:
            raise StrictSchemaError(f"message: duplicate key {key!r}")
        result[key] = value
    return result


def _array_hook(code: int, payload: bytes) -> Any:
    if code != _ARRAY_EXT:
        raise StrictSchemaError(f"message: unsupported extension {code}")
    try:
        obj = msgpack.unpackb(payload, raw=False, strict_map_key=True, object_pairs_hook=_pairs)
        obj = fields(obj, {"data", "dtype", "shape"}, path="array")
        dtype_name = string(obj["dtype"], "array.dtype")
        if dtype_name not in _DTYPES:
            raise StrictSchemaError(f"array.dtype: unsupported dtype {dtype_name}")
        shape = tuple(integer(x, f"array.shape[{i}]", minimum=0) for i, x in enumerate(sequence(obj["shape"], "array.shape")))
        if len(shape) > 8:
            raise StrictSchemaError("array.shape: too many dimensions")
        if type(obj["data"]) is not bytes:
            raise StrictSchemaError("array.data: expected bytes")
        dtype = np.dtype(dtype_name)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != len(obj["data"]):
            raise StrictSchemaError("array.data: size mismatch")
        result = np.frombuffer(obj["data"], dtype=dtype).reshape(shape).copy()
        result.flags.writeable = False
        return result
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"array: invalid extension: {exc}") from exc


def _prepare(value: Any) -> Any:
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _prepare(value.to_mapping())
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise StrictSchemaError("message: expected string mapping key")
            result[key] = _prepare(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_prepare(item) for item in value]
    if value is None or type(value) in (bool, int, float, str, bytes):
        if type(value) is float and not np.isfinite(value):
            raise StrictSchemaError("message: non-finite float")
        return value
    raise StrictSchemaError(f"message: unsupported type {type(value).__name__}")


def encode_message(value: Any) -> bytes:
    try:
        return msgpack.packb(_prepare(value), default=_array_default, use_bin_type=True, strict_types=True)
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"message: encoding failed: {exc}") from exc


def decode_message(payload: bytes) -> Any:
    if type(payload) is not bytes:
        raise StrictSchemaError("message: expected bytes")
    if len(payload) > 1 << 30:
        raise StrictSchemaError("message: payload too large")
    try:
        return msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            ext_hook=_array_hook,
            object_pairs_hook=_pairs,
        )
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"message: decoding failed: {exc}") from exc
