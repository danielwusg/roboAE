from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import select
import signal
import socketserver
import ssl
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Mapping, Sequence


RELAY_PROTOCOL_VERSION = 1
PROVIDER_HOST = "api.anthropic.com"
PROVIDER_PATH = "/v1/messages?beta=true"
CREDENTIAL_FILENAME = "oauth_token"
CLAUDE_KEY_HELPER = "/claude-key-helper"
CLAUDE_AUTH_TRACE = "/claude-auth-trace"
CLAUDE_RELAY_BASE_URL = f"http://{PROVIDER_HOST}"
OAUTH_BETA = "oauth-2025-04-20"
_RELAY_SOCKET = Path("/relay/api.sock")
_RELAY_STATE = Path("/relay/relay_state.json")
_MAX_HEADER_BYTES = 8192
_BETA_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_LINUX_MFD_CLOEXEC = 0x0001
_LINUX_MFD_ALLOW_SEALING = 0x0002
_LINUX_F_GETFD = 1
_LINUX_F_ADD_SEALS = 1033
_LINUX_F_GET_SEALS = 1034
_LINUX_FD_CLOEXEC = 0x0001
_LINUX_F_SEAL_SEAL = 0x0001
_LINUX_F_SEAL_SHRINK = 0x0002
_LINUX_F_SEAL_GROW = 0x0004
_LINUX_F_SEAL_WRITE = 0x0008
_LINUX_REQUIRED_SEALS = (
    _LINUX_F_SEAL_SEAL | _LINUX_F_SEAL_SHRINK | _LINUX_F_SEAL_GROW | _LINUX_F_SEAL_WRITE
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_secret(descriptor: int, maximum_bytes: int = 8192) -> bytes:
    result = bytearray()
    try:
        while True:
            block = os.read(descriptor, min(4096, maximum_bytes + 1 - len(result)))
            if not block:
                break
            result.extend(block)
            if len(result) > maximum_bytes:
                raise RuntimeError("relay secret exceeds size limit")
    finally:
        os.close(descriptor)
    value = bytes(result).strip()
    if not value or b"\x00" in value or b"\n" in value or b"\r" in value:
        raise RuntimeError("relay secret has invalid format")
    return value


def _header_value(name: str, value: str, *, allow_tab: bool = False) -> str:
    if type(value) is not str:
        raise RuntimeError(f"invalid {name} header")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"invalid {name} header") from exc
    if (
        not encoded
        or len(encoded) > _MAX_HEADER_BYTES
        or any(byte == 127 or byte < 32 and not (allow_tab and byte == 9) for byte in encoded)
    ):
        raise RuntimeError(f"invalid {name} header")
    return value


def _merged_anthropic_beta(values: Sequence[str]) -> str:
    if isinstance(values, (str, bytes)):
        raise RuntimeError("invalid anthropic-beta headers")
    tokens: list[str] = []
    seen: set[str] = set()
    inbound_bytes = 0
    for index, value in enumerate(values):
        checked = _header_value("anthropic-beta", value, allow_tab=True)
        inbound_bytes += len(checked.encode("ascii")) + (1 if index else 0)
        if inbound_bytes > _MAX_HEADER_BYTES:
            raise RuntimeError("anthropic-beta headers exceed size limit")
        for item in checked.split(","):
            token = item.strip(" \t")
            if _BETA_TOKEN.fullmatch(token) is None:
                raise RuntimeError("invalid anthropic-beta token")
            if token != OAUTH_BETA and token not in seen:
                tokens.append(token)
                seen.add(token)
    tokens.append(OAUTH_BETA)
    result = ",".join(tokens)
    if len(result.encode("ascii")) > _MAX_HEADER_BYTES:
        raise RuntimeError("anthropic-beta header exceeds size limit")
    return result


@dataclass(frozen=True)
class RelayLimits:
    max_requests: int = 64
    max_request_bytes: int = 64 * 1024**2
    max_response_bytes: int = 64 * 1024**2
    deadline_s: float = 1800.0
    provider_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        integer_values = (self.max_requests, self.max_request_bytes, self.max_response_bytes)
        if any(type(value) is not int or value < 1 for value in integer_values):
            raise ValueError("relay limits must be positive integers")
        if self.max_requests > 1024 or self.max_request_bytes > 256 * 1024**2:
            raise ValueError("relay request limits exceed hard bounds")
        if self.max_response_bytes > 256 * 1024**2:
            raise ValueError("relay response limit exceeds hard bound")
        if self.deadline_s <= 0 or self.provider_timeout_s <= 0:
            raise ValueError("relay time limits must be positive")
        if self.provider_timeout_s > self.deadline_s:
            raise ValueError("relay provider timeout exceeds deadline")

    @classmethod
    def from_mapping(cls, value: Any) -> "RelayLimits":
        if not isinstance(value, Mapping) or set(value) != {
            "max_requests",
            "max_request_bytes",
            "max_response_bytes",
            "deadline_s",
            "provider_timeout_s",
        }:
            raise ValueError("invalid relay limits")
        return cls(**value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "deadline_s": self.deadline_s,
            "provider_timeout_s": self.provider_timeout_s,
        }


def relay_provenance(coding_model: str, limits: RelayLimits) -> dict[str, Any]:
    source = Path(__file__)
    entry = source.with_name("_relay_entry.py")
    return {
        "protocol_version": RELAY_PROTOCOL_VERSION,
        "coding_model": coding_model,
        "provider": {"host": PROVIDER_HOST, "path": PROVIDER_PATH, "oauth_beta": OAUTH_BETA},
        "claude_auth": {
            "base_url": CLAUDE_RELAY_BASE_URL,
            "helper": CLAUDE_KEY_HELPER,
            "key_storage": "sealed_memfd",
            "read_mode": "pread_offset_zero",
            "status_artifact": "claude_auth/helper_status",
        },
        "limits": limits.to_mapping(),
        "source_sha256": _sha256(source),
        "entry_sha256": _sha256(entry),
    }


class _RelayRuntime:
    def __init__(
        self,
        *,
        coding_model: str,
        limits: RelayLimits,
        proxy_key: bytes,
        provider_token: bytes | None,
        offline: bool,
    ) -> None:
        self.coding_model = coding_model
        self.limits = limits
        self.proxy_key = proxy_key
        self.provider_token = provider_token
        self.offline = offline
        self.started_ns = time.time_ns()
        self.deadline = time.monotonic() + limits.deadline_s
        self.requests: list[dict[str, Any]] = []
        self.provider_contacted = False
        self.ready = False
        self.stop_requested = False
        self.status = "starting"
        self.error_type: str | None = None

    def persist(self) -> None:
        _write_json(
            _RELAY_STATE,
            {
                "schema_version": 1,
                "protocol_version": RELAY_PROTOCOL_VERSION,
                "offline": self.offline,
                "network_isolated": self.offline,
                "coding_model": self.coding_model,
                "started_ns": self.started_ns,
                "updated_ns": time.time_ns(),
                "ready": self.ready,
                "status": self.status,
                "provider_contacted": self.provider_contacted,
                "request_count": len(self.requests),
                "requests": list(self.requests),
                "error_type": self.error_type,
            },
        )

    def record(self, method: str, path: str, model: str | None, body_bytes: int, status: str) -> None:
        safe_path = path if len(path) <= 256 and all(32 <= ord(character) < 127 for character in path) else "<invalid>"
        safe_model = model if model is not None and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model) else None
        self.requests.append(
            {
                "index": len(self.requests) + 1,
                "method": method,
                "path": safe_path,
                "model": safe_model,
                "body_bytes": body_bytes,
                "status": status,
            }
        )
        self.persist()


class _RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def runtime(self) -> _RelayRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _respond(self, status: int, body: bytes, headers: Mapping[str, str] | None = None) -> None:
        content_type = next(
            (value for name, value in (headers or {}).items() if name.lower() == "content-type"),
            "application/json",
        )
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Content-Type", content_type)
        for name, value in (headers or {}).items():
            if name.lower() not in {"connection", "content-length", "content-type", "transfer-encoding"}:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _reject(self, status: int, error: str) -> None:
        self._respond(status, json.dumps({"type": "error", "error": error}).encode())

    def _body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding") is not None:
            self._reject(400, "unsupported transfer encoding")
            return None
        values = self.headers.get_all("Content-Length") or []
        if len(values) != 1 or not values[0].isdigit():
            self._reject(411, "content length required")
            return None
        length = int(values[0])
        if length < 1 or length > self.runtime.limits.max_request_bytes:
            self._reject(413, "request body exceeds limit")
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._reject(400, "incomplete request body")
            return None
        return body

    def _authorization_status(self) -> str:
        api_keys = self.headers.get_all("x-api-key") or []
        authorizations = self.headers.get_all("Authorization") or []
        if len(api_keys) > 1:
            return "duplicate_api_key"
        if len(authorizations) != 1:
            if len(authorizations) > 1:
                return "duplicate_authorization"
            if not api_keys:
                return "missing_authentication"
        if api_keys:
            try:
                api_key = api_keys[0].encode("ascii")
            except UnicodeEncodeError:
                return "invalid_api_key"
            if len(api_key) > _MAX_HEADER_BYTES or not hmac.compare_digest(api_key, self.runtime.proxy_key):
                return "api_key_mismatch"
        if authorizations:
            try:
                authorization = authorizations[0].encode("ascii")
            except UnicodeEncodeError:
                return "invalid_authorization"
            expected = b"Bearer " + self.runtime.proxy_key
            if len(authorization) > _MAX_HEADER_BYTES or not hmac.compare_digest(authorization, expected):
                return "authorization_mismatch"
        return "authorized"

    def _safe_headers(self) -> dict[str, str]:
        result = {"Content-Type": "application/json", "Authorization": f"Bearer {self.runtime.provider_token.decode('ascii')}"}
        for name in ("accept", "anthropic-version", "user-agent"):
            values = self.headers.get_all(name) or []
            if values:
                if len(values) != 1:
                    raise RuntimeError(f"duplicate {name} header")
                result[name] = _header_value(name, values[0])
        result["anthropic-beta"] = _merged_anthropic_beta(self.headers.get_all("anthropic-beta") or [])
        return result

    def do_POST(self) -> None:
        runtime = self.runtime
        if len(runtime.requests) >= runtime.limits.max_requests:
            self._reject(429, "relay request budget exhausted")
            runtime.stop_requested = True
            return
        authorization_status = self._authorization_status()
        if authorization_status != "authorized":
            runtime.record("POST", self.path, None, 0, authorization_status)
            self._reject(401, "relay authentication failed")
            runtime.stop_requested = True
            return
        body = self._body()
        if body is None:
            runtime.record("POST", self.path, None, 0, "invalid_body")
            runtime.stop_requested = True
            return
        try:
            payload = json.loads(body)
            model = payload.get("model") if isinstance(payload, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            model = None
        if type(model) is not str or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model) is None:
            runtime.record("POST", self.path, None, len(body), "invalid_json")
            self._reject(400, "request must contain a model")
            runtime.stop_requested = True
            return
        if self.path != PROVIDER_PATH or model != runtime.coding_model:
            runtime.record("POST", self.path, model, len(body), "route_or_model_rejected")
            self._reject(403, "relay route or model mismatch")
            runtime.stop_requested = True
            return
        if runtime.offline:
            runtime.record("POST", self.path, model, len(body), "offline_observed")
            self._reject(503, "offline relay probe")
            runtime.status = "offline_complete"
            runtime.stop_requested = True
            runtime.persist()
            return
        try:
            headers = self._safe_headers()
            runtime.provider_contacted = True
            runtime.record("POST", self.path, model, len(body), "provider_started")
            connection = http.client.HTTPSConnection(
                PROVIDER_HOST,
                443,
                timeout=max(0.1, min(runtime.limits.provider_timeout_s, runtime.deadline - time.monotonic())),
                context=ssl.create_default_context(),
            )
            try:
                connection.request("POST", PROVIDER_PATH, body=body, headers=headers)
                response = connection.getresponse()
                response_body = response.read(runtime.limits.max_response_bytes + 1)
                response_headers = {
                    name: value
                    for name, value in response.getheaders()
                    if name.lower() in {"content-encoding", "content-type", "request-id", "retry-after"}
                    or name.lower().startswith("anthropic-ratelimit-")
                }
                status = int(response.status)
            finally:
                connection.close()
            if len(response_body) > runtime.limits.max_response_bytes:
                runtime.requests[-1]["status"] = "provider_response_too_large"
                runtime.persist()
                self._reject(502, "provider response exceeds limit")
                runtime.stop_requested = True
                return
            runtime.requests[-1]["status"] = f"provider_{status}"
            runtime.persist()
            self._respond(status, response_body, response_headers)
            if status == 401:
                runtime.status = "provider_auth_failed"
                runtime.stop_requested = True
                runtime.persist()
        except BaseException as exc:
            runtime.error_type = type(exc).__name__
            if runtime.requests:
                runtime.requests[-1]["status"] = "provider_error"
            runtime.persist()
            self._reject(502, "provider request failed")

    def do_GET(self) -> None:
        self.runtime.record("GET", self.path, None, 0, "method_rejected")
        self._reject(405, "method not allowed")
        self.runtime.stop_requested = True


class _RelayServer(socketserver.UnixStreamServer):
    allow_reuse_address = False

    def __init__(self, path: str, runtime: _RelayRuntime) -> None:
        self.runtime = runtime
        super().__init__(path, _RelayHandler)


def relay_entry_main(specification: Mapping[str, Any]) -> int:
    if not isinstance(specification, Mapping) or set(specification) != {
        "protocol_version",
        "coding_model",
        "limits",
        "proxy_key_fd",
        "credential_fd",
        "offline",
    }:
        raise RuntimeError("invalid relay entry specification")
    if specification["protocol_version"] != RELAY_PROTOCOL_VERSION:
        raise RuntimeError("relay protocol version mismatch")
    coding_model = specification["coding_model"]
    proxy_key_fd = specification["proxy_key_fd"]
    credential_fd = specification["credential_fd"]
    offline = specification["offline"]
    if (
        type(coding_model) is not str
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", coding_model) is None
        or type(proxy_key_fd) is not int
        or proxy_key_fd < 3
    ):
        raise RuntimeError("invalid relay entry values")
    if type(offline) is not bool:
        raise RuntimeError("invalid relay mode")
    if offline != (credential_fd is None):
        raise RuntimeError("relay credential mode mismatch")
    if credential_fd is not None and (type(credential_fd) is not int or credential_fd < 3):
        raise RuntimeError("invalid relay credential descriptor")
    limits = RelayLimits.from_mapping(specification["limits"])
    proxy_key = _read_secret(proxy_key_fd)
    provider_token = None if credential_fd is None else _read_secret(credential_fd)
    runtime = _RelayRuntime(
        coding_model=coding_model,
        limits=limits,
        proxy_key=proxy_key,
        provider_token=provider_token,
        offline=offline,
    )

    def stop(signum: int, frame: Any) -> None:
        del signum, frame
        runtime.status = "terminated"
        runtime.stop_requested = True
        runtime.persist()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _RELAY_SOCKET.unlink(missing_ok=True)
    with _RelayServer(str(_RELAY_SOCKET), runtime) as server:
        os.chmod(_RELAY_SOCKET, 0o600)
        server.timeout = 0.2
        runtime.ready = True
        runtime.status = "ready"
        runtime.persist()
        while not runtime.stop_requested and time.monotonic() < runtime.deadline:
            server.handle_request()
        if time.monotonic() >= runtime.deadline and not runtime.stop_requested:
            runtime.status = "deadline"
        elif runtime.status == "ready":
            runtime.status = "stopped"
        runtime.persist()
    _RELAY_SOCKET.unlink(missing_ok=True)
    return 0


def _credential_descriptor(directory: Path) -> int:
    directory = Path(directory).absolute()
    directory_stat = directory.lstat()
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_ISLNK(directory_stat.st_mode)
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        raise PermissionError("Claude credential directory must be private and owned by the current user")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(CREDENTIAL_FILENAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_uid != os.getuid()
            or value.st_nlink != 1
            or stat.S_IMODE(value.st_mode) != 0o600
            or value.st_size < 1
            or value.st_size > 8192
        ):
            raise PermissionError("Claude OAuth token file must be private, regular, and singly linked")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _secret_pipe(value: bytes) -> int:
    read_fd, write_fd = os.pipe()
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(write_fd, value[offset:])
    finally:
        os.close(write_fd)
    return read_fd


def _linux_memfd_create() -> int:
    if sys.platform != "linux":
        raise RuntimeError("sealed memfd requires Linux")
    flags = _LINUX_MFD_CLOEXEC | _LINUX_MFD_ALLOW_SEALING
    native = getattr(os, "memfd_create", None)
    if callable(native):
        descriptor = native("claude-relay-key", flags)
    else:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            create = libc.memfd_create
        except (AttributeError, OSError) as exc:
            raise RuntimeError("libc memfd_create is unavailable") from exc
        create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        create.restype = ctypes.c_int
        ctypes.set_errno(0)
        descriptor = int(create(b"claude-relay-key", flags))
        if descriptor < 0:
            error = ctypes.get_errno() or errno.EIO
            raise OSError(error, os.strerror(error))
    if type(descriptor) is not int or descriptor < 3:
        if type(descriptor) is int and descriptor >= 0:
            os.close(descriptor)
        raise RuntimeError("memfd_create returned an invalid descriptor")
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("memfd_create returned a non-regular descriptor")
        if fcntl.fcntl(descriptor, _LINUX_F_GETFD) & _LINUX_FD_CLOEXEC != _LINUX_FD_CLOEXEC:
            raise RuntimeError("memfd_create omitted close-on-exec")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _sealed_memfd(value: bytes) -> int:
    if (
        type(value) is not bytes
        or not 32 <= len(value) <= 128
        or re.fullmatch(rb"[A-Za-z0-9_-]+", value) is None
    ):
        raise RuntimeError("invalid Claude relay key")
    descriptor = _linux_memfd_create()
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        fcntl.fcntl(descriptor, _LINUX_F_ADD_SEALS, _LINUX_REQUIRED_SEALS)
        if fcntl.fcntl(descriptor, _LINUX_F_GET_SEALS) != _LINUX_REQUIRED_SEALS:
            raise RuntimeError("Claude key memfd seal verification failed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class ClaudeKeyMaterial:
    descriptor: int
    helper: Path
    interpreter: Path
    shell: Path
    trace: Path


class RelaySession:
    def __init__(
        self,
        *,
        isolation_dir: Path,
        artifact_dir: Path,
        coding_model: str,
        limits: RelayLimits,
        credential_dir: Path | None,
        offline: bool,
    ) -> None:
        self.isolation_dir = Path(isolation_dir).resolve()
        self.artifact_dir = Path(artifact_dir).resolve()
        self.coding_model = coding_model
        self.limits = limits
        self.credential_dir = None if credential_dir is None else Path(credential_dir).absolute()
        self.offline = offline
        if type(coding_model) is not str or not coding_model or type(offline) is not bool:
            raise ValueError("invalid relay session configuration")
        if offline != (credential_dir is None):
            raise ValueError("offline relay must not receive credentials")
        self.process: subprocess.Popen[bytes] | None = None
        self.started_ns: int | None = None
        self.finished_ns: int | None = None
        self.termination_reason: str | None = None
        self.session_root: Path | None = None
        self.exchange_dir: Path | None = None
        self._proxy_key: bytes | None = None

    @property
    def socket_mount(self) -> Path:
        if self.exchange_dir is None:
            raise RuntimeError("relay session has not started")
        return self.exchange_dir

    def _state(self) -> dict[str, Any] | None:
        if self.exchange_dir is None:
            return None
        path = self.exchange_dir / _RELAY_STATE.name
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise RuntimeError("relay state is invalid")
        return value

    def _process_record(self) -> dict[str, Any]:
        state = self._state()
        process = self.process
        return {
            "schema_version": 1,
            "pid": None if process is None else process.pid,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "returncode": None if process is None else process.poll(),
            "termination_reason": self.termination_reason,
            "offline": self.offline,
            "network_isolated": self.offline,
            "provenance": relay_provenance(self.coding_model, self.limits),
            "state": state,
        }

    def _write_process_record(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.artifact_dir / "relay_process.json", self._process_record())

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3.0)

    def start(self) -> "RelaySession":
        if self.process is not None:
            raise RuntimeError("relay session already started")
        from robot_auto_evolve.agent.sandbox import (
            SandboxLimits,
            SandboxMount,
            executable_mounts,
            sandbox_command,
            trusted_relay_command,
        )

        self.isolation_dir.mkdir(parents=True, exist_ok=True)
        self.isolation_dir.chmod(0o700)
        session_root = self.isolation_dir / "relay_sessions" / uuid.uuid4().hex
        code_dir = session_root / "code"
        exchange_dir = session_root / "exchange"
        code_dir.mkdir(parents=True)
        exchange_dir.mkdir()
        session_root.chmod(0o700)
        code_dir.chmod(0o700)
        exchange_dir.chmod(0o700)
        source = Path(__file__)
        entry = source.with_name("_relay_entry.py")
        for item in (source, entry):
            target = code_dir / item.name
            target.write_bytes(item.read_bytes())
            target.chmod(0o444)
        self.session_root = session_root
        self.exchange_dir = exchange_dir
        proxy_key = secrets.token_urlsafe(32).encode("ascii")
        self._proxy_key = proxy_key
        relay_key_fd = _secret_pipe(proxy_key)
        credential_fd = None
        ready_read = None
        ready_write = None
        process = None
        stderr = None
        try:
            if not self.offline:
                if self.credential_dir is None:
                    raise ValueError("online relay requires a credential directory")
                credential_fd = _credential_descriptor(self.credential_dir)
            ready_read, ready_write = os.pipe()
            specification = {
                "protocol_version": RELAY_PROTOCOL_VERSION,
                "coding_model": self.coding_model,
                "limits": self.limits.to_mapping(),
                "proxy_key_fd": relay_key_fd,
                "credential_fd": credential_fd,
                "offline": self.offline,
            }
            python = Path("/usr/bin/python3").resolve()
            command = [str(python), "/relay-code/_relay_entry.py", json.dumps(specification, sort_keys=True)]
            mounts = [
                *executable_mounts(python, include_prefix=False),
                SandboxMount(code_dir, target=Path("/relay-code")),
                SandboxMount(exchange_dir, writable=True, target=Path("/relay")),
            ]
            if not self.offline:
                mounts.extend(
                    SandboxMount(path)
                    for path in (Path("/etc/resolv.conf"), Path("/etc/hosts"))
                    if path.exists()
                )
            builder = sandbox_command if self.offline else trusted_relay_command
            sandboxed = builder(
                command,
                cwd=Path("/relay"),
                environment={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
                isolation_dir=self.isolation_dir,
                mounts=mounts,
                limits=SandboxLimits.relay_default(),
                ready_fd=ready_write,
            )
            inherited = tuple(value for value in (ready_write, relay_key_fd, credential_fd) if value is not None)
            self.started_ns = time.time_ns()
            stderr = (exchange_dir / "relay.stderr.log").open("wb")
            process = subprocess.Popen(
                sandboxed,
                cwd=self.isolation_dir,
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                start_new_session=True,
                pass_fds=inherited,
            )
            self.process = process
            stderr.close()
            stderr = None
            os.close(ready_write)
            ready_write = None
            os.close(relay_key_fd)
            relay_key_fd = -1
            if credential_fd is not None:
                os.close(credential_fd)
                credential_fd = None
            ready, _, _ = select.select([ready_read], [], [], min(15.0, self.limits.deadline_s))
            marker = os.read(ready_read, 1) if ready else b""
            os.close(ready_read)
            ready_read = None
            if marker != b"R":
                raise RuntimeError("relay sandbox failed before executable launch")
            deadline = time.monotonic() + min(15.0, self.limits.deadline_s)
            while time.monotonic() < deadline:
                state = self._state()
                if state is not None and state.get("ready") is True and (exchange_dir / "api.sock").exists():
                    self._write_process_record()
                    return self
                if process.poll() is not None:
                    raise RuntimeError("relay exited before readiness")
                time.sleep(0.02)
            raise RuntimeError("relay readiness timed out")
        except BaseException:
            if process is not None:
                self._terminate(process)
            self.finished_ns = time.time_ns()
            self.termination_reason = "startup_failure"
            self._write_process_record()
            raise
        finally:
            for descriptor in (ready_read, ready_write, None if relay_key_fd == -1 else relay_key_fd, credential_fd):
                if descriptor is not None:
                    os.close(descriptor)
            if stderr is not None:
                stderr.close()

    def claude_key_material(self) -> ClaudeKeyMaterial:
        if self._proxy_key is None:
            raise RuntimeError("relay key is unavailable")
        if self.session_root is None:
            raise RuntimeError("relay session has no private state root")
        descriptor = _sealed_memfd(self._proxy_key)
        interpreter = Path("/usr/bin/python3").resolve()
        shell = Path("/bin/sh").resolve()
        helper = self.session_root / "claude-key-helper"
        trace = self.artifact_dir / "claude_auth"
        trace.mkdir(parents=True, exist_ok=True)
        trace.chmod(0o700)
        source = (
            f"#!{interpreter}\n"
            "import os\n"
            f"status = '{CLAUDE_AUTH_TRACE}/helper_status'\n"
            "def report(value):\n"
            "    target = os.open(status, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)\n"
            "    try:\n"
            "        offset = 0\n"
            "        while offset < len(value):\n"
            "            offset += os.write(target, value[offset:])\n"
            "    finally:\n"
            "        os.close(target)\n"
            "report(b'started\\n')\n"
            "try:\n"
            f"    source = os.open('/proc/1/fd/{descriptor}', os.O_RDONLY)\n"
            "    try:\n"
            "        value = os.pread(source, 129, 0)\n"
            "    finally:\n"
            "        os.close(source)\n"
            "except OSError as error:\n"
            "    report(f'open_error:{error.errno}\\n'.encode('ascii'))\n"
            "    raise SystemExit(1)\n"
            "if not 32 <= len(value) <= 128 or any(character not in b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for character in value):\n"
            "    report(b'invalid_key\\n')\n"
            "    raise SystemExit(1)\n"
            "report(b'success\\n')\n"
            "offset = 0\n"
            "while offset < len(value):\n"
            "    offset += os.write(1, value[offset:])\n"
        )
        try:
            helper.write_text(source, encoding="utf-8")
            helper.chmod(0o500)
        except BaseException:
            os.close(descriptor)
            raise
        self._proxy_key = None
        return ClaudeKeyMaterial(descriptor, helper, interpreter, shell, trace)

    def wait_for_request(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._state()
            if state is not None and state.get("request_count", 0) >= 1:
                return state
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(0.02)
        raise RuntimeError("offline relay probe observed no API request")

    def stop(self, reason: str) -> None:
        if self.finished_ns is not None:
            return
        self.termination_reason = reason
        if self.process is not None:
            self._terminate(self.process)
        self.finished_ns = time.time_ns()
        self._proxy_key = None
        self._write_process_record()
