from __future__ import annotations

import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class RunInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        if not isinstance(signum, int) or signum not in {signal.SIGINT, signal.SIGTERM}:
            raise ValueError("unsupported interruption signal")
        self.signum = int(signum)
        self.signal_name = signal.Signals(self.signum).name
        self.exit_status = 128 + self.signum
        super().__init__(self.signal_name)


@dataclass(frozen=True)
class _OwnedProcess:
    process: Any
    label: str
    pid: int
    registered_ns: int


class OwnedProcessRegistry:
    def __init__(self, terminate_timeout_s: float = 5.0, kill_timeout_s: float = 3.0) -> None:
        if terminate_timeout_s <= 0 or kill_timeout_s <= 0:
            raise ValueError("process cleanup timeouts must be positive")
        self.terminate_timeout_s = float(terminate_timeout_s)
        self.kill_timeout_s = float(kill_timeout_s)
        self._lock = threading.RLock()
        self._active: dict[int, _OwnedProcess] = {}
        self._cleaned: dict[int, dict[str, Any]] = {}
        self._stopping = False
        self._signal_number: int | None = None
        self._started_ns: int | None = None
        self._finished_ns: int | None = None

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping

    @property
    def signal_number(self) -> int | None:
        with self._lock:
            return self._signal_number

    def register(self, process: Any, label: str) -> None:
        pid = getattr(process, "pid", None)
        if type(pid) is not int or pid < 1 or type(label) is not str or not label:
            raise ValueError("owned process requires a PID and label")
        record = _OwnedProcess(process, label, pid, time.time_ns())
        with self._lock:
            previous = self._active.get(pid)
            if previous is not None and previous.process is not process:
                raise RuntimeError(f"owned process PID collision: {pid}")
            self._active[pid] = record
            stopping = self._stopping
        if stopping:
            self._terminate((record,))

    def unregister(self, process: Any) -> None:
        pid = getattr(process, "pid", None)
        if type(pid) is not int:
            return
        with self._lock:
            record = self._active.get(pid)
            if record is not None and record.process is process:
                self._active.pop(pid, None)

    @staticmethod
    def _alive(records: tuple[_OwnedProcess, ...]) -> tuple[_OwnedProcess, ...]:
        return tuple(record for record in records if record.process.poll() is None)

    @staticmethod
    def _wait(records: tuple[_OwnedProcess, ...], timeout_s: float) -> tuple[_OwnedProcess, ...]:
        deadline = time.monotonic() + timeout_s
        remaining = OwnedProcessRegistry._alive(records)
        while remaining and time.monotonic() < deadline:
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
            remaining = OwnedProcessRegistry._alive(remaining)
        return remaining

    def _terminate(self, records: tuple[_OwnedProcess, ...]) -> None:
        with self._lock:
            records = tuple(
                record
                for record in records
                if id(record.process) not in self._cleaned
                or self._cleaned[id(record.process)]["alive_after_cleanup"] is True
            )
        if not records:
            return
        term_errors: dict[int, str] = {}
        kill_errors: dict[int, str] = {}
        term_sent: set[int] = set()
        kill_sent: set[int] = set()
        alive = self._alive(records)
        for record in alive:
            try:
                os.killpg(record.pid, signal.SIGTERM)
                term_sent.add(record.pid)
            except ProcessLookupError:
                pass
            except OSError as exc:
                term_errors[record.pid] = f"{type(exc).__name__}: {exc}"
        remaining = self._wait(alive, self.terminate_timeout_s)
        for record in remaining:
            try:
                os.killpg(record.pid, signal.SIGKILL)
                kill_sent.add(record.pid)
            except ProcessLookupError:
                pass
            except OSError as exc:
                kill_errors[record.pid] = f"{type(exc).__name__}: {exc}"
        remaining = self._wait(remaining, self.kill_timeout_s)
        remaining_pids = {record.pid for record in remaining}
        finished_ns = time.time_ns()
        with self._lock:
            for record in records:
                returncode = record.process.poll()
                self._cleaned[id(record.process)] = {
                    "label": record.label,
                    "pid": record.pid,
                    "process_group": record.pid,
                    "registered_ns": record.registered_ns,
                    "finished_ns": finished_ns,
                    "sigterm_sent": record.pid in term_sent,
                    "sigkill_sent": record.pid in kill_sent,
                    "returncode": returncode,
                    "alive_after_cleanup": record.pid in remaining_pids,
                    "sigterm_error": term_errors.get(record.pid),
                    "sigkill_error": kill_errors.get(record.pid),
                }
                current = self._active.get(record.pid)
                if current is not None and current.process is record.process and returncode is not None:
                    self._active.pop(record.pid, None)

    def terminate_all(self, signum: int | None = None) -> dict[str, Any]:
        if signum is not None and (not isinstance(signum, int) or signum not in {signal.SIGINT, signal.SIGTERM}):
            raise ValueError("unsupported cleanup signal")
        signum = None if signum is None else int(signum)
        with self._lock:
            if signum is not None and self._signal_number is None:
                self._signal_number = signum
            if self._started_ns is None:
                self._started_ns = time.time_ns()
            self._stopping = True
            records = tuple(self._active.values())
        self._terminate(records)
        with self._lock:
            self._finished_ns = time.time_ns()
        return self.to_mapping()

    def to_mapping(self) -> dict[str, Any]:
        with self._lock:
            active = [
                {
                    "label": record.label,
                    "pid": record.pid,
                    "process_group": record.pid,
                    "registered_ns": record.registered_ns,
                    "returncode": record.process.poll(),
                }
                for record in sorted(self._active.values(), key=lambda item: (item.label, item.pid))
            ]
            signal_number = self._signal_number
            return {
                "schema_version": 1,
                "signal_number": signal_number,
                "signal_name": None if signal_number is None else signal.Signals(signal_number).name,
                "started_ns": self._started_ns,
                "finished_ns": self._finished_ns,
                "groups": sorted(
                    self._cleaned.values(),
                    key=lambda item: (item["label"], item["registered_ns"], item["pid"]),
                ),
                "active_groups": active,
                "cleanup_complete": not active,
            }

    def write(self, path: Path) -> None:
        import json

        Path(path).write_text(json.dumps(self.to_mapping(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


_REGISTRY_LOCK = threading.RLock()
_ACTIVE_REGISTRY: OwnedProcessRegistry | None = None


def current_process_registry() -> OwnedProcessRegistry | None:
    with _REGISTRY_LOCK:
        return _ACTIVE_REGISTRY


@contextmanager
def process_registry(registry: OwnedProcessRegistry) -> Iterator[OwnedProcessRegistry]:
    if not isinstance(registry, OwnedProcessRegistry):
        raise TypeError("process registry is required")
    global _ACTIVE_REGISTRY
    with _REGISTRY_LOCK:
        if _ACTIVE_REGISTRY is not None:
            raise RuntimeError("a process registry is already active")
        _ACTIVE_REGISTRY = registry
    try:
        yield registry
    finally:
        with _REGISTRY_LOCK:
            if _ACTIVE_REGISTRY is registry:
                _ACTIVE_REGISTRY = None


def register_owned_process(process: Any, label: str) -> None:
    registry = current_process_registry()
    if registry is not None:
        registry.register(process, label)


def unregister_owned_process(process: Any) -> None:
    registry = current_process_registry()
    if registry is not None:
        registry.unregister(process)


@contextmanager
def interruption_handlers(registry: OwnedProcessRegistry) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("signal handlers require the main thread")
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(signum: int, frame: Any) -> None:
        del frame
        if registry.signal_number is not None:
            return
        try:
            registry.terminate_all(signum)
        finally:
            raise RunInterrupted(signum)

    for signum in previous:
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
