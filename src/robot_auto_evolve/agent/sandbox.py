from __future__ import annotations

from dataclasses import dataclass


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxLimits:
    """Resource caps applied (via setrlimit) to the rollout agent-worker subprocess in
    ``agent/gateway.py`` -- a memory (RLIMIT_AS), CPU-time, open-files and file-size ceiling so a
    runaway scaffold cannot exhaust the node.

    There is no OS sandbox: the scaffold runs as a plain subprocess. Fairness is enforced by
    observation-stripping plus the tool relay through the trusted parent, by the agent conda
    environment's inability to import any simulator package, and by the committed-scaffold
    grep-guard (see ``agent/gateway.py`` and ``evolution/free_backend.py``). Only these resource
    caps remain.
    """

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
