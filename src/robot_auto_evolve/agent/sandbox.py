from __future__ import annotations

from dataclasses import dataclass


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxLimits:

    cpu_seconds: int
    address_space_bytes: int
    open_files: int
    file_size_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.cpu_seconds,
                self.address_space_bytes,
                self.open_files,
                self.file_size_bytes,
            )
        ):
            raise SandboxUnavailableError("agent worker resource limits must be positive integers")

    @classmethod
    def agent_default(cls) -> "SandboxLimits":
        return cls(3600, 8 * 1024**3, 128, 64 * 1024**2)
