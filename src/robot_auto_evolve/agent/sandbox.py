from __future__ import annotations

from dataclasses import dataclass


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxLimits:
    """Resource caps applied (via setrlimit) to the rollout agent-worker subprocess in
    ``agent/gateway.py`` -- a memory (RLIMIT_AS), CPU-time, open-files and file-size ceiling so a
    runaway scaffold cannot exhaust the node.

    History: through s16 the agent worker ran inside a full ``unshare`` user/mount/pid/net namespace
    + chroot sandbox, and this module held ~250 lines of mount/namespace machinery driven by an
    ``agent/_sandbox_entry.py`` helper. W3 (s17, operator-confirmed) dropped that OS sandbox: the
    scaffold now runs as a plain subprocess, and fairness is enforced instead by observation-stripping
    + the tool relay through the trusted parent, the agent conda env's inability to import ANY
    simulator package, and the committed-scaffold grep-guard (see ``agent/gateway.py`` +
    ``evolution/free_backend.py``). Only these resource caps remain. To restore the stronger OS
    isolation, recover ``agent/sandbox.py`` + ``agent/_sandbox_entry.py`` from git commit ``0458178``
    and re-wire ``gateway.start()`` to call the old ``sandbox_command``.
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
