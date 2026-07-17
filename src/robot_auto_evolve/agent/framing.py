from __future__ import annotations

import os
import select
import struct
import time
from typing import Any

from robot_auto_evolve.protocol import StrictSchemaError, decode_message, encode_message


MAX_FRAME_BYTES = 1 << 30


def _read_exact(fd: int, count: int, timeout_s: float | None) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while remaining:
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            raise TimeoutError("agent frame read timed out")
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            raise EOFError("agent frame stream closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(fd: int, timeout_s: float | None = None) -> Any:
    header = _read_exact(fd, 8, timeout_s)
    size = struct.unpack(">Q", header)[0]
    if size > MAX_FRAME_BYTES:
        raise StrictSchemaError("agent frame exceeds size limit")
    return decode_message(_read_exact(fd, size, timeout_s))


def write_frame(fd: int, value: Any) -> None:
    payload = encode_message(value)
    if len(payload) > MAX_FRAME_BYTES:
        raise StrictSchemaError("agent frame exceeds size limit")
    data = memoryview(struct.pack(">Q", len(payload)) + payload)
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise EOFError("agent frame stream closed")
        data = data[written:]
