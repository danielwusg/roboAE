from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from typing import Any, Mapping

from robot_auto_evolve.agent.framing import read_frame, write_frame
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.private_metrics import validate_private_metrics
from robot_auto_evolve.protocol import CanonicalActionChunk, StrictSchemaError
from robot_auto_evolve.provenance import EpisodeKey
from robot_auto_evolve.runtime_paths import assert_clean_import_origin, project_root_from_package


def _worker_class(entrypoint: str) -> type[Any]:
    if type(entrypoint) is not str or ":" not in entrypoint:
        raise StrictSchemaError("environment.adapter: expected module:class")
    module_name, class_name = entrypoint.split(":", 1)
    if not module_name.startswith("robot_auto_evolve.benchmarks.") or not class_name.isidentifier():
        raise StrictSchemaError("environment.adapter: untrusted entrypoint")
    worker_class = getattr(importlib.import_module(module_name), class_name, None)
    if not isinstance(worker_class, type):
        raise StrictSchemaError("environment.adapter: class not found")
    return worker_class


def _envelope(sequence: Any, ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"sequence": sequence, "ok": ok, "result": result, "error": error}


def serve() -> int:
    assert_clean_import_origin(project_root_from_package())
    protocol_out = os.dup(sys.stdout.fileno())
    sys.stdout = sys.stderr
    worker: Any = None
    while True:
        try:
            request = read_frame(sys.stdin.fileno())
        except EOFError:
            return 0
        sequence = request.get("sequence") if isinstance(request, Mapping) else None
        try:
            if not isinstance(request, Mapping) or set(request) != {"sequence", "operation", "payload"}:
                raise StrictSchemaError("simulator request: invalid envelope")
            operation = request["operation"]
            payload = request["payload"]
            if operation == "initialize":
                if worker is not None or not isinstance(payload, Mapping) or set(payload) != {
                    "profile",
                    "episode",
                    "render_gpu_id",
                }:
                    raise StrictSchemaError("simulator initialize: invalid payload")
                profile = Profile.from_mapping(payload["profile"])
                episode = EpisodeKey.from_mapping(payload["episode"])
                worker_class = _worker_class(profile.environment.adapter)
                worker = worker_class(profile, episode, render_gpu_id=payload["render_gpu_id"])
                result: Any = {"ready": True}
            elif operation == "reinitialize":
                if worker is None or not isinstance(payload, Mapping) or set(payload) != {"profile", "episode", "render_gpu_id"}:
                    raise StrictSchemaError("simulator reinitialize: invalid payload")
                profile = Profile.from_mapping(payload["profile"])
                episode = EpisodeKey.from_mapping(payload["episode"])
                worker.close()
                worker_class = _worker_class(profile.environment.adapter)
                worker = worker_class(profile, episode, render_gpu_id=payload["render_gpu_id"])
                result = {"ready": True}
            elif operation == "reset":
                if worker is None or payload != {}:
                    raise StrictSchemaError("simulator reset: invalid request")
                worker.reset()
                result = {"reset": True}
            elif operation == "observe":
                if worker is None or payload != {}:
                    raise StrictSchemaError("simulator observe: invalid request")
                result = worker.observe().to_mapping()
            elif operation == "apply":
                if worker is None or not isinstance(payload, Mapping) or set(payload) != {"action"}:
                    raise StrictSchemaError("simulator apply: invalid request")
                worker.apply(CanonicalActionChunk.from_mapping(payload["action"]))
                result = {"applied": True}
            elif operation == "private_success":
                if worker is None or payload != {}:
                    raise StrictSchemaError("simulator private_success: invalid request")
                result = {"success": bool(worker.private_success())}
            elif operation == "private_metrics":
                if worker is None or payload != {}:
                    raise StrictSchemaError("simulator private_metrics: invalid request")
                method = getattr(worker, "private_metrics", None)
                result = (
                    {"available": False, "metrics": {}}
                    if not callable(method)
                    else {"available": True, "metrics": validate_private_metrics(method())}
                )
            elif operation == "runtime_info":
                if worker is None or payload != {} or not callable(getattr(worker, "runtime_info", None)):
                    raise StrictSchemaError("simulator runtime_info: invalid request")
                result = worker.runtime_info()
            elif operation == "close":
                if payload != {}:
                    raise StrictSchemaError("simulator close: invalid request")
                if worker is not None:
                    worker.close()
                write_frame(protocol_out, _envelope(sequence, True, {"closed": True}))
                return 0
            else:
                raise StrictSchemaError("simulator request: unknown operation")
            write_frame(protocol_out, _envelope(sequence, True, result))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            write_frame(protocol_out, _envelope(sequence, False, error=f"{type(exc).__name__}: {exc}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
