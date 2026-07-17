from __future__ import annotations

import ctypes
import json
import os
import resource
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


_PR_SET_NO_NEW_PRIVS = 38
_LINUX_CAPABILITY_VERSION_3 = 0x20080522


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapabilityData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _run_mount(arguments: list[str]) -> None:
    result = subprocess.run(
        ["/usr/bin/mount", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "mount failed"
        raise RuntimeError(detail)


def _target(root: Path, value: str) -> Path:
    target = PurePosixPath(value)
    if not target.is_absolute() or ".." in target.parts or target == PurePosixPath("/"):
        raise RuntimeError("invalid sandbox mount target")
    return root.joinpath(*target.parts[1:])


def _bind(root: Path, mount: dict[str, Any]) -> None:
    if not isinstance(mount, dict) or set(mount) != {"source", "target", "writable", "device"}:
        raise RuntimeError("invalid sandbox mount")
    source = Path(mount["source"])
    destination = _target(root, mount["target"])
    writable = mount["writable"]
    device = mount["device"]
    if type(writable) is not bool or type(device) is not bool or not source.exists():
        raise RuntimeError("invalid sandbox mount source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        destination.mkdir(exist_ok=True)
    else:
        destination.touch(exist_ok=True)
    try:
        _run_mount(["--bind", str(source), str(destination)])
    except RuntimeError as exc:
        raise RuntimeError(f"bind {source} to {mount['target']}: {exc}") from exc
    options = ["remount", "bind", "rw" if writable else "ro", "nosuid"]
    if not device:
        options.append("nodev")
    _run_mount(["-o", ",".join(options), str(destination)])


def _drop_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    header = _CapabilityHeader(_LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapabilityData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("sandbox entry expects one specification")
    specification = json.loads(sys.argv[1])
    if not isinstance(specification, dict) or set(specification) != {
        "root",
        "mounts",
        "cwd",
        "command",
        "environment",
        "limits",
        "ready_fd",
        "mount_proc",
    }:
        raise RuntimeError("invalid sandbox specification")
    root = Path(specification["root"])
    root.mkdir(parents=True, exist_ok=True)
    _run_mount(["--make-rprivate", "/"])
    _run_mount(["-t", "tmpfs", "-o", "mode=0755,nosuid,nodev", "tmpfs", str(root)])
    try:
        for mount in specification["mounts"]:
            _bind(root, mount)
        cwd = _target(root, specification["cwd"])
        if not cwd.is_dir():
            raise RuntimeError("sandbox working directory is not mounted")
        command = specification["command"]
        environment = specification["environment"]
        limits = specification["limits"]
        ready_fd = specification["ready_fd"]
        mount_proc = specification["mount_proc"]
        if (
            not isinstance(command, list)
            or not command
            or type(command[0]) is not str
            or not command[0]
            or any(type(item) is not str for item in command[1:])
            or not isinstance(environment, dict)
            or any(type(key) is not str or type(value) is not str for key, value in environment.items())
        ):
            raise RuntimeError("invalid sandbox command")
        if not isinstance(limits, dict) or set(limits) != {
            "cpu_seconds",
            "address_space_bytes",
            "processes",
            "open_files",
            "file_size_bytes",
        } or any(type(value) is not int or value < 1 for value in limits.values()):
            raise RuntimeError("invalid sandbox limits")
        if ready_fd is not None and (type(ready_fd) is not int or ready_fd < 3):
            raise RuntimeError("invalid sandbox readiness descriptor")
        if type(mount_proc) is not bool:
            raise RuntimeError("invalid sandbox proc setting")
        if mount_proc:
            proc = root / "proc"
            proc.mkdir(exist_ok=True)
            _run_mount(["-t", "proc", "-o", "nosuid,nodev,noexec", "proc", str(proc)])
        os.chroot(root)
        os.chdir(specification["cwd"])
        os.umask(0o077)
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
        resource.setrlimit(
            resource.RLIMIT_AS,
            (limits["address_space_bytes"], limits["address_space_bytes"]),
        )
        resource.setrlimit(resource.RLIMIT_NPROC, (limits["processes"], limits["processes"]))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits["open_files"], limits["open_files"]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits["file_size_bytes"], limits["file_size_bytes"]))
        _drop_capabilities()
        if ready_fd is not None:
            os.write(ready_fd, b"R")
            os.close(ready_fd)
        os.execve(command[0], command, environment)
    except BaseException:
        raise


if __name__ == "__main__":
    raise SystemExit(main())
