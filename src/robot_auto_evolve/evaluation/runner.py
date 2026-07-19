from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from robot_auto_evolve.protocol.schema import StrictSchemaError, boolean, enum, integer, mapping, string


@dataclass(frozen=True)
class EpisodeExecution:
    state: str
    success: bool | None
    steps: int
    artifacts: Mapping[str, bytes]
    error: str | None = None

    def __post_init__(self) -> None:
        state = enum(self.state, {"complete", "partial", "error"}, "execution.state")
        success = None if self.success is None else boolean(self.success, "execution.success")
        steps = integer(self.steps, "execution.steps", minimum=0)
        artifacts = mapping(self.artifacts, "execution.artifacts")
        checked: dict[str, bytes] = {}
        for name, value in sorted(artifacts.items()):
            name = string(name, "execution.artifact name")
            if type(value) is not bytes:
                raise StrictSchemaError(f"execution.artifacts.{name}: expected bytes")
            checked[name] = value
        error = None if self.error is None else string(self.error, "execution.error")
        if state == "complete" and (success is None or error is not None or "trace.msgpack" not in checked):
            raise StrictSchemaError("execution: complete requires success, trace.msgpack, and null error")
        if state == "partial" and (success is not None or error is not None):
            raise StrictSchemaError("execution: partial requires null success and error")
        if state == "error" and (success is not None or error is None):
            raise StrictSchemaError("execution: error requires null success and non-null error")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "artifacts", checked)
        object.__setattr__(self, "error", error)
