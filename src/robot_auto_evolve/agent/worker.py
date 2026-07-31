from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

from robot_auto_evolve.protocol import CanonicalActionChunk, StrictSchemaError

from .api import AgentRequest, AgentStepResult
from .framing import read_frame, write_frame
from .tools import RelayedToolbox


def _load_scaffold(path: Path) -> tuple[Any, frozenset[str], frozenset[str]]:
    spec = importlib.util.spec_from_file_location("editable_scaffold", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scaffold module")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    factory = getattr(module, "create_scaffold", None)
    if not callable(factory):
        raise RuntimeError("scaffold must define create_scaffold()")
    scaffold = factory()
    if not callable(getattr(scaffold, "act", None)):
        raise RuntimeError("scaffold must define act(request, tools)")
    config = getattr(module, "SCAFFOLD_CONFIG", None)
    if not isinstance(config, Mapping) or set(config) != {"required_capabilities", "optional_capabilities"}:
        raise RuntimeError("scaffold must define exact SCAFFOLD_CONFIG")
    required = frozenset(config["required_capabilities"])
    optional = frozenset(config["optional_capabilities"])
    if not required or required & optional or "vla" not in required:
        raise RuntimeError("scaffold capability configuration is invalid")
    return scaffold, required, optional


def _envelope(sequence: Any, ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"sequence": sequence, "ok": ok, "result": result, "error": error}


def serve(scaffold_path: Path) -> int:
    protocol_out = os.dup(sys.stdout.fileno())
    sys.stdout = sys.stderr
    scaffold, required_capabilities, optional_capabilities = _load_scaffold(scaffold_path)
    toolbox: RelayedToolbox | None = None
    sessions: set[str] = set()
    while True:
        step_events = None
        try:
            request = read_frame(sys.stdin.fileno())
        except EOFError:
            return 0
        sequence = request.get("sequence") if isinstance(request, Mapping) else None
        try:
            if not isinstance(request, Mapping) or set(request) != {"sequence", "operation", "payload"}:
                raise StrictSchemaError("worker request: invalid envelope")
            operation = request["operation"]
            payload = request["payload"]
            if operation == "initialize":
                if toolbox is not None or not isinstance(payload, Mapping) or set(payload) != {"capabilities"}:
                    raise StrictSchemaError("worker initialize: invalid payload")
                capabilities = payload["capabilities"]
                if not isinstance(capabilities, Mapping):
                    raise StrictSchemaError("worker initialize: capabilities must be a mapping")
                declared = required_capabilities | optional_capabilities
                if set(capabilities) - declared or not required_capabilities <= set(capabilities):
                    raise StrictSchemaError("worker initialize: relayed capabilities differ from scaffold config")
                for capability, declaration in capabilities.items():
                    if not isinstance(declaration, Mapping) or declaration.get("required") != (
                        capability in required_capabilities
                    ):
                        raise StrictSchemaError("worker initialize: required capability mismatch")
                toolbox = RelayedToolbox(capabilities, sys.stdin.fileno(), protocol_out)
                result: Any = {"ready": True}
            elif operation == "act":
                if toolbox is None:
                    raise RuntimeError("worker is not initialized")
                agent_request = AgentRequest.from_mapping(payload)
                if agent_request.session_id not in sessions:
                    raise RuntimeError("worker session must be reset before act")
                toolbox.begin_step(
                    agent_request.observation.step_index,
                    agent_request.request_id,
                    agent_request.session_id,
                )
                action = scaffold.act(agent_request, toolbox)
                if not isinstance(action, CanonicalActionChunk):
                    raise StrictSchemaError("scaffold act: expected CanonicalActionChunk")
                step_events = toolbox.finish_step()
                result = AgentStepResult(action, step_events).to_mapping()
            elif operation == "reset":
                if toolbox is None:
                    raise RuntimeError("worker is not initialized")
                if not isinstance(payload, Mapping) or set(payload) != {"session_id", "policy_seed", "task_id"}:
                    raise StrictSchemaError("worker reset: invalid payload")
                reset = getattr(scaffold, "reset", None)
                if callable(reset):
                    reset(payload["session_id"])
                sessions.add(payload["session_id"])
                result = {"reset": True}
            elif operation == "end_session":
                if toolbox is None:
                    raise RuntimeError("worker is not initialized")
                if not isinstance(payload, Mapping) or set(payload) != {"session_id"}:
                    raise StrictSchemaError("worker end_session: invalid payload")
                if payload["session_id"] not in sessions:
                    raise RuntimeError("worker end_session: unknown session")
                reset = getattr(scaffold, "reset", None)
                if callable(reset):
                    reset(payload["session_id"])
                sessions.discard(payload["session_id"])
                result = {"ended": True}
            elif operation == "close":
                write_frame(protocol_out, _envelope(sequence, True, {"closed": True}))
                return 0
            else:
                raise StrictSchemaError("worker request: unknown operation")
            write_frame(protocol_out, _envelope(sequence, True, result))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if toolbox is not None and isinstance(request, Mapping) and request.get("operation") == "act" and step_events is None:
                step_events = toolbox.finish_step()
            error = {
                "message": f"{type(exc).__name__}: {exc}",
                "events": [] if step_events is None else [event.to_mapping() for event in step_events],
            }
            write_frame(protocol_out, _envelope(sequence, False, error=error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.scaffold.resolve()
    if not path.is_file() or path.is_symlink():
        parser.error("--scaffold must be a regular file")
    return serve(path)


if __name__ == "__main__":
    raise SystemExit(main())
