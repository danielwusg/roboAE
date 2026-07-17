from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping

from robot_auto_evolve.protocol.schema import StrictSchemaError, mapping, string, string_tuple

from .http import MsgpackServiceClient, ServiceCallError
from .identity import ServiceIdentity


_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "PROXY",
    "SECRET",
    "TOKEN",
)

_BLOCKED_ENVIRONMENT_NAMES = {
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "TRANSFORMERS_CACHE",
    "VIRTUAL_ENV",
    "_CE_CONDA",
    "_CE_M",
}


def _register_process(process: subprocess.Popen[bytes], label: str) -> None:
    from robot_auto_evolve.process_lifecycle import register_owned_process

    register_owned_process(process, label)


def _unregister_process(process: subprocess.Popen[bytes]) -> None:
    from robot_auto_evolve.process_lifecycle import unregister_owned_process

    unregister_owned_process(process)


def scrubbed_service_environment(base: Mapping[str, str], overlay: Mapping[str, str]) -> dict[str, str]:
    environment = {
        key: value
        for key, value in base.items()
        if key not in _BLOCKED_ENVIRONMENT_NAMES
        and not any(fragment in key.upper() for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS)
    }
    environment.update(overlay)
    return environment


@dataclass(frozen=True)
class ServiceProcessSpec:
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    endpoint: str
    identity: ServiceIdentity
    startup_timeout_s: float = 900.0
    shutdown_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        command = string_tuple(self.command, "process.command")
        cwd = Path(self.cwd).resolve()
        if not cwd.is_dir():
            raise StrictSchemaError("process.cwd: expected directory")
        environment = mapping(self.environment, "process.environment")
        checked_environment = {
            string(key, "process.environment key"): string(value, f"process.environment.{key}", allow_empty=True)
            for key, value in environment.items()
        }
        endpoint = string(self.endpoint, "process.endpoint")
        if not isinstance(self.identity, ServiceIdentity):
            raise StrictSchemaError("process.identity: expected ServiceIdentity")
        if self.identity.gpu_ids:
            expected_visible = ",".join(str(item) for item in self.identity.gpu_ids)
            if checked_environment.get("CUDA_VISIBLE_DEVICES") != expected_visible:
                raise StrictSchemaError(
                    f"process.environment.CUDA_VISIBLE_DEVICES: expected {expected_visible}"
                )
        startup = float(self.startup_timeout_s)
        shutdown = float(self.shutdown_timeout_s)
        if startup <= 0 or shutdown <= 0:
            raise StrictSchemaError("process: expected positive timeouts")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "environment", checked_environment)
        object.__setattr__(self, "endpoint", endpoint.rstrip("/"))
        object.__setattr__(self, "startup_timeout_s", startup)
        object.__setattr__(self, "shutdown_timeout_s", shutdown)


class ServiceStartupError(RuntimeError):
    pass


class ServiceSupervisor:
    def __init__(self, spec: ServiceProcessSpec, log_dir: Path, *, reuse_exact: bool = True) -> None:
        if not isinstance(spec, ServiceProcessSpec):
            raise StrictSchemaError("supervisor.spec: expected ServiceProcessSpec")
        self.spec = spec
        self.log_dir = Path(log_dir)
        self.reuse_exact = bool(reuse_exact)
        self.client = MsgpackServiceClient(spec.endpoint, spec.identity, timeout=2.0)
        self.process: subprocess.Popen[bytes] | None = None
        self.reused = False
        self._stdout: IO[bytes] | None = None
        self._stderr: IO[bytes] | None = None
        self._startup_deadline: float | None = None

    def _existing_identity(self) -> ServiceIdentity | None:
        try:
            return self.client.identity()
        except ServiceCallError:
            return None

    def launch(self) -> None:
        if self.process is not None or self.reused:
            raise RuntimeError("supervisor already started")
        existing = self._existing_identity()
        if existing is not None:
            self.spec.identity.validate_exact(existing)
            if not self.reuse_exact:
                raise ServiceStartupError("exact service already occupies endpoint")
            self.reused = True
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout = open(self.log_dir / f"{self.spec.identity.replica_id}.stdout.log", "ab", buffering=0)
        self._stderr = open(self.log_dir / f"{self.spec.identity.replica_id}.stderr.log", "ab", buffering=0)
        environment = scrubbed_service_environment(os.environ, self.spec.environment)
        try:
            self.process = subprocess.Popen(
                self.spec.command,
                cwd=self.spec.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._stdout,
                stderr=self._stderr,
                start_new_session=True,
            )
            _register_process(
                self.process,
                f"service:{self.spec.identity.service_name}:{self.spec.identity.replica_id}",
            )
            self._startup_deadline = time.monotonic() + self.spec.startup_timeout_s
        except BaseException:
            self.stop()
            raise

    def wait_ready(self) -> MsgpackServiceClient:
        if self.reused:
            return self.client
        process = self.process
        deadline = self._startup_deadline
        if process is None or deadline is None:
            raise RuntimeError("supervisor has not launched")
        while True:
            if process.poll() is not None:
                raise ServiceStartupError(f"service exited with code {process.returncode}")
            try:
                actual = self.client.identity()
                self.spec.identity.validate_exact(actual)
                return self.client
            except ServiceCallError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ServiceStartupError("service startup timed out")
                time.sleep(min(0.25, remaining))

    def start(self) -> MsgpackServiceClient:
        try:
            self.launch()
            return self.wait_ready()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.reused:
            self.reused = False
            self._startup_deadline = None
            return
        process = self.process
        self.process = None
        self._startup_deadline = None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=self.spec.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=self.spec.shutdown_timeout_s)
            _unregister_process(process)
        for stream_name in ("_stdout", "_stderr"):
            stream = getattr(self, stream_name)
            if stream is not None:
                stream.close()
                setattr(self, stream_name, None)

    def __enter__(self) -> MsgpackServiceClient:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
