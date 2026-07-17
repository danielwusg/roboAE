from __future__ import annotations

import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxMount:
    source: Path
    writable: bool = False
    device: bool = False
    target: Path | None = None

    def __post_init__(self) -> None:
        source = Path(self.source).absolute()
        target = source if self.target is None else Path(self.target)
        if not source.exists() or not target.is_absolute() or ".." in target.parts or target == Path("/"):
            raise SandboxUnavailableError("invalid sandbox mount")
        if type(self.writable) is not bool or type(self.device) is not bool:
            raise SandboxUnavailableError("invalid sandbox mount flags")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)

    def to_mapping(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "writable": self.writable,
            "device": self.device,
        }


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int
    address_space_bytes: int
    processes: int
    open_files: int
    file_size_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.cpu_seconds,
                self.address_space_bytes,
                self.processes,
                self.open_files,
                self.file_size_bytes,
            )
        ):
            raise SandboxUnavailableError("sandbox limits must be positive integers")

    @classmethod
    def agent_default(cls) -> "SandboxLimits":
        return cls(3600, 8 * 1024**3, 32, 128, 64 * 1024**2)

    @classmethod
    def revision_default(cls) -> "SandboxLimits":
        return cls(1800, 256 * 1024**3, 128, 256, 256 * 1024**2)

    @classmethod
    def relay_default(cls) -> "SandboxLimits":
        return cls(1800, 1024**3, 4, 64, 32 * 1024**2)

    def to_mapping(self) -> dict[str, int]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "address_space_bytes": self.address_space_bytes,
            "processes": self.processes,
            "open_files": self.open_files,
            "file_size_bytes": self.file_size_bytes,
        }


def _system_mounts() -> list[SandboxMount]:
    mounts = []
    device_paths = {"/dev/null", "/dev/random", "/dev/urandom", "/dev/zero"}
    for path in (
        "/lib",
        "/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/etc/ld.so.cache",
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/ssl/certs",
        "/dev/null",
        "/dev/random",
        "/dev/urandom",
        "/dev/zero",
    ):
        source = Path(path)
        if source.exists():
            mounts.append(SandboxMount(source, device=path in device_paths))
    return mounts


def _script_interpreter(executable: Path) -> Path | None:
    try:
        first_line = executable.open("rb").readline(4096)
    except OSError:
        return None
    if not first_line.startswith(b"#!"):
        return None
    command = first_line[2:].decode("utf-8", errors="strict").strip().split()
    if not command or not command[0].startswith("/"):
        raise SandboxUnavailableError("sandbox executable has unsupported shebang")
    interpreter = Path(command[0]).absolute()
    if not interpreter.is_file():
        raise SandboxUnavailableError("sandbox shebang interpreter does not exist")
    return interpreter


def executable_mounts(executable: Path, *, include_prefix: bool) -> list[SandboxMount]:
    executable = Path(executable).absolute()
    if not executable.is_file():
        raise SandboxUnavailableError("sandbox executable does not exist")
    mounts: list[SandboxMount] = []
    if include_prefix:
        prefix = executable.parent.parent
        mounts.append(SandboxMount(prefix))
    else:
        mounts.append(SandboxMount(executable))
    interpreter = _script_interpreter(executable)
    if interpreter is not None:
        mounts.append(SandboxMount(interpreter.parent.parent))
    return mounts


def _sandbox_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    isolation_dir: Path,
    mounts: Sequence[SandboxMount],
    limits: SandboxLimits,
    ready_fd: int | None = None,
    mount_proc: bool = True,
    isolate_network: bool,
) -> list[str]:
    unshare = shutil.which("unshare")
    mount = shutil.which("mount")
    if unshare is None or mount != "/usr/bin/mount":
        raise SandboxUnavailableError("required namespace tools are unavailable")
    if not command or not Path(command[0]).is_absolute():
        raise SandboxUnavailableError("sandbox command must use an absolute executable")
    if not isinstance(limits, SandboxLimits):
        raise SandboxUnavailableError("sandbox limits are required")
    if ready_fd is not None and (type(ready_fd) is not int or ready_fd < 3):
        raise SandboxUnavailableError("invalid sandbox readiness descriptor")
    if type(mount_proc) is not bool:
        raise SandboxUnavailableError("invalid sandbox proc setting")
    cwd = Path(cwd).absolute()
    isolation_dir = Path(isolation_dir).absolute()
    root = isolation_dir / "roots" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    merged: dict[Path, SandboxMount] = {}
    for item in [*_system_mounts(), *mounts]:
        target = Path(item.target)
        previous = merged.get(target)
        if previous is not None and previous != item:
            raise SandboxUnavailableError(f"conflicting sandbox mount: {target}")
        merged[target] = item
    if not any(
        Path(item.target) == cwd
        or Path(item.target) in cwd.parents
        or cwd == Path(item.target).parent
        for item in merged.values()
    ):
        raise SandboxUnavailableError("sandbox working directory is not mounted")
    specification = {
        "root": str(root),
        "mounts": [item.to_mapping() for item in merged.values()],
        "cwd": str(cwd),
        "command": list(command),
        "environment": dict(environment),
        "limits": limits.to_mapping(),
        "ready_fd": ready_fd,
        "mount_proc": mount_proc,
    }
    helper = Path(__file__).with_name("_sandbox_entry.py")
    namespace_arguments = [
        unshare,
        "--user",
        "--map-root-user",
        "--mount",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
    ]
    if isolate_network:
        namespace_arguments.insert(4, "--net")
    return [
        *namespace_arguments,
        sys.executable,
        str(helper),
        json.dumps(specification, sort_keys=True, separators=(",", ":")),
    ]


def sandbox_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    isolation_dir: Path,
    mounts: Sequence[SandboxMount],
    limits: SandboxLimits,
    ready_fd: int | None = None,
    mount_proc: bool = True,
) -> list[str]:
    return _sandbox_command(
        command,
        cwd=cwd,
        environment=environment,
        isolation_dir=isolation_dir,
        mounts=mounts,
        limits=limits,
        ready_fd=ready_fd,
        mount_proc=mount_proc,
        isolate_network=True,
    )


def trusted_relay_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    isolation_dir: Path,
    mounts: Sequence[SandboxMount],
    limits: SandboxLimits,
    ready_fd: int | None = None,
) -> list[str]:
    return _sandbox_command(
        command,
        cwd=cwd,
        environment=environment,
        isolation_dir=isolation_dir,
        mounts=mounts,
        limits=limits,
        ready_fd=ready_fd,
        mount_proc=True,
        isolate_network=False,
    )


def isolated_state_mounts(isolation_dir: Path) -> list[SandboxMount]:
    isolation_dir = Path(isolation_dir).absolute()
    state_root = isolation_dir / "processes" / uuid.uuid4().hex
    result = []
    for name in ("home", "cache", "tmp"):
        path = state_root / name
        path.mkdir(parents=True, exist_ok=True)
        result.append(SandboxMount(path, writable=True, target=Path("/sandbox-state") / name))
    return result
