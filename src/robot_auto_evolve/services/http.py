from __future__ import annotations

import http.client
import logging
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from robot_auto_evolve.protocol import StrictSchemaError, decode_message, encode_message
from robot_auto_evolve.protocol.schema import fields, string

from .identity import ServiceIdentity


class ServiceCallError(RuntimeError):
    pass


class ServiceProtocolError(ServiceCallError):
    pass


def serialize_methods(
    methods: Mapping[str, Callable[[Any, str, str], Any]],
) -> dict[str, Callable[[Any, str, str], Any]]:
    lock = threading.Lock()

    def wrap(method: Callable[[Any, str, str], Any]) -> Callable[[Any, str, str], Any]:
        def invoke(payload: Any, session_id: str, request_id: str) -> Any:
            with lock:
                return method(payload, session_id, request_id)

        return invoke

    return {name: wrap(method) for name, method in methods.items()}


def _call_request(value: Any) -> tuple[str, str, str, Any]:
    obj = fields(value, {"method", "request_id", "session_id", "payload"}, path="call")
    return (
        string(obj["method"], "call.method"),
        string(obj["request_id"], "call.request_id"),
        string(obj["session_id"], "call.session_id"),
        obj["payload"],
    )


def _call_response(value: Any, request_id: str) -> Any:
    obj = fields(value, {"ok", "request_id", "payload", "error"}, path="response")
    if type(obj["ok"]) is not bool:
        raise ServiceProtocolError("response.ok: expected bool")
    if string(obj["request_id"], "response.request_id") != request_id:
        raise ServiceProtocolError("response.request_id: mismatch")
    if obj["ok"]:
        if obj["error"] is not None:
            raise ServiceProtocolError("response.error: expected null")
        return obj["payload"]
    error = string(obj["error"], "response.error")
    if obj["payload"] is not None:
        raise ServiceProtocolError("response.payload: expected null")
    raise ServiceCallError(error)


class MsgpackServiceServer:
    def __init__(
        self,
        identity: ServiceIdentity,
        methods: Mapping[str, Callable[[Any, str, str], Any]],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not isinstance(identity, ServiceIdentity):
            raise StrictSchemaError("server.identity: expected ServiceIdentity")
        if not methods:
            raise StrictSchemaError("server.methods: empty mapping")
        checked: dict[str, Callable[[Any, str, str], Any]] = {}
        for name, method in methods.items():
            checked[string(name, "server.method name")] = method
            if not callable(method):
                raise StrictSchemaError(f"server.methods.{name}: expected callable")
        self.identity = identity
        self.methods = checked
        self.max_request_bytes = int(max_request_bytes)
        if self.max_request_bytes < 1:
            raise StrictSchemaError("server.max_request_bytes: expected positive int")
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, status: int, value: Any) -> None:
                body = encode_message(value)
                self.send_response(status)
                self.send_header("Content-Type", "application/msgpack")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/identity":
                    self._send(200, outer.identity.to_mapping())
                elif self.path == "/health":
                    self._send(200, {"identity": outer.identity.to_mapping(), "status": "ok"})
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self) -> None:
                if self.path != "/call":
                    self._send(404, {"error": "not_found"})
                    return
                try:
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        raise StrictSchemaError("request: missing Content-Length")
                    length = int(raw_length)
                    if length < 0 or length > outer.max_request_bytes:
                        raise StrictSchemaError("request: invalid Content-Length")
                    content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
                    if content_type != "application/msgpack":
                        raise StrictSchemaError("request: expected application/msgpack")
                    method, request_id, session_id, payload = _call_request(decode_message(self.rfile.read(length)))
                    if method not in outer.methods:
                        self._send(
                            404,
                            {"ok": False, "request_id": request_id, "payload": None, "error": "unknown_method"},
                        )
                        return
                    result = outer.methods[method](payload, session_id, request_id)
                    self._send(200, {"ok": True, "request_id": request_id, "payload": result, "error": None})
                except (StrictSchemaError, ValueError) as exc:
                    request_id = locals().get("request_id", "invalid")
                    self._send(
                        400,
                        {"ok": False, "request_id": request_id, "payload": None, "error": str(exc)},
                    )
                except Exception as exc:
                    request_id = locals().get("request_id", "invalid")
                    logging.exception("service call failed")
                    self._send(
                        500,
                        {
                            "ok": False,
                            "request_id": request_id,
                            "payload": None,
                            "error": f"service_error:{type(exc).__name__}",
                        },
                    )

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("server already started")
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "MsgpackServiceServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class MsgpackServiceClient:
    def __init__(
        self,
        base_url: str,
        expected_identity: ServiceIdentity,
        *,
        timeout: float = 30.0,
        max_response_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.base_url = string(base_url, "client.base_url").rstrip("/")
        if not isinstance(expected_identity, ServiceIdentity):
            raise StrictSchemaError("client.expected_identity: expected ServiceIdentity")
        self.expected_identity = expected_identity
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self._validated_identity: ServiceIdentity | None = None
        if self.timeout <= 0 or self.max_response_bytes < 1:
            raise StrictSchemaError("client: invalid limits")

    def _request(self, path: str, body: bytes | None = None) -> Any:
        headers = {"Accept": "application/msgpack"}
        method = "GET"
        if body is not None:
            method = "POST"
            headers["Content-Type"] = "application/msgpack"
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/msgpack":
                    raise ServiceProtocolError(f"response: unexpected content type {content_type!r}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > self.max_response_bytes:
                    raise ServiceProtocolError("response: too large")
                payload = response.read(self.max_response_bytes + 1)
                if len(payload) > self.max_response_bytes:
                    raise ServiceProtocolError("response: too large")
                return decode_message(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read(self.max_response_bytes + 1)
            try:
                value = decode_message(payload)
            except Exception:
                raise ServiceCallError(f"http {exc.code}") from exc
            return value
        except urllib.error.URLError as exc:
            raise ServiceCallError(f"service unavailable: {exc.reason}") from exc
        except http.client.HTTPException as exc:
            raise ServiceCallError(f"service unavailable: HTTP protocol error: {exc}") from exc

    def identity(self) -> ServiceIdentity:
        return ServiceIdentity.from_mapping(self._request("/identity"))

    def validate_identity(self) -> ServiceIdentity:
        actual = self.identity()
        self.expected_identity.validate_exact(actual)
        self._validated_identity = actual
        return actual

    def health(self) -> bool:
        try:
            value = fields(self._request("/health"), {"identity", "status"}, path="health")
            if string(value["status"], "health.status") != "ok":
                return False
            self.expected_identity.validate_exact(ServiceIdentity.from_mapping(value["identity"]))
            return True
        except Exception:
            return False

    def call(
        self,
        method: str,
        payload: Any,
        *,
        session_id: str,
        request_id: str | None = None,
    ) -> Any:
        method = string(method, "call.method")
        session_id = string(session_id, "call.session_id")
        request_id = string(request_id or uuid.uuid4().hex, "call.request_id")
        if self._validated_identity is None:
            self.validate_identity()
        request = {
            "method": method,
            "request_id": request_id,
            "session_id": session_id,
            "payload": payload,
        }
        return _call_response(self._request("/call", encode_message(request)), request_id)
