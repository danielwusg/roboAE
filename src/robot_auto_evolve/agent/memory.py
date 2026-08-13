from __future__ import annotations

import threading
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.protocol import StrictSchemaError


__all__ = ["ScaffoldMemory", "memory_key", "memory_value"]


MEMORY_MAX_DEPTH = 16
MEMORY_MAX_KEY = 512


def memory_key(value: Any) -> str:
    if type(value) is not str or not value or len(value) > MEMORY_MAX_KEY:
        raise StrictSchemaError(f"memory key: expected a nonempty string of at most {MEMORY_MAX_KEY} characters")
    return value


def memory_value(value: Any, path: str = "memory value", depth: int = 0) -> Any:
    if depth > MEMORY_MAX_DEPTH:
        raise StrictSchemaError(f"{path}: nested deeper than {MEMORY_MAX_DEPTH}")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise StrictSchemaError(f"{path}: expected a finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [memory_value(item, f"{path}[]", depth + 1) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            if type(name) is not str:
                raise StrictSchemaError(f"{path}: mapping keys must be strings")
            result[name] = memory_value(item, f"{path}.{name}", depth + 1)
        return result
    raise StrictSchemaError(f"{path}: {type(value).__name__} cannot be stored")


class ScaffoldMemory:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, Any] = {}
        self._writes = 0
        self._reads = 0
        self._hits = 0

    def remember(self, key: Any, value: Any) -> None:
        name = memory_key(key)
        stored = memory_value(value)
        with self._lock:
            self._entries[name] = stored
            self._writes += 1

    def recall(self, key: Any) -> Any:
        name = memory_key(key)
        with self._lock:
            self._reads += 1
            if name not in self._entries:
                return None
            self._hits += 1
            return memory_value(self._entries[name])

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._entries))

    def forget(self, key: Any) -> bool:
        name = memory_key(key)
        with self._lock:
            return self._entries.pop(name, _MISSING) is not _MISSING

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {name: memory_value(value) for name, value in sorted(self._entries.items())}

    def usage(self) -> dict[str, int]:
        with self._lock:
            return {
                "n_entries": len(self._entries),
                "n_writes": self._writes,
                "n_reads": self._reads,
                "n_read_hits": self._hits,
            }


_MISSING = object()
