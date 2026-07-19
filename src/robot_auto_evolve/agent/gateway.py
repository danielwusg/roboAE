from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation, StrictSchemaError
from robot_auto_evolve.runtime_paths import clean_python_path, project_root_from_package, verify_python_import_origin

from .api import AgentEvent, AgentRequest, AgentStepResult
from .framing import read_frame, write_frame
from .sandbox import SandboxLimits, SandboxMount, executable_mounts, isolated_state_mounts, sandbox_command
from .tools import ToolEndpoint, Toolbox


class AgentProcessError(RuntimeError):
    def __init__(self, message: str, events: tuple[AgentEvent, ...] = ()) -> None:
        super().__init__(message)
        self.events = events


_FORBIDDEN_AGENT_MODULES = (
    "calvin_env",
    "libero",
    "mujoco",
    "robocasa",
    "robosuite",
    "sapien",
    "simpler_env",
)


def _register_process(process: subprocess.Popen[bytes], label: str) -> None:
    from robot_auto_evolve.process_lifecycle import register_owned_process

    register_owned_process(process, label)


def _unregister_process(process: subprocess.Popen[bytes]) -> None:
    from robot_auto_evolve.process_lifecycle import unregister_owned_process

    unregister_owned_process(process)


def scrubbed_environment(agent_python: Path, isolation_dir: Path, scaffold_dir: Path) -> dict[str, str]:
    del isolation_dir
    home = Path("/sandbox-state/home")
    cache = Path("/sandbox-state/cache")
    temporary = Path("/sandbox-state/tmp")
    project_root = project_root_from_package()
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": str(cache / "huggingface"),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_PROXY": "127.0.0.1,localhost",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PATH": os.pathsep.join((str(agent_python.parent), "/usr/bin", "/bin")),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": clean_python_path(project_root, scaffold_dir),
        "TMPDIR": str(temporary),
        "TRANSFORMERS_CACHE": str(cache / "transformers"),
        "XDG_CACHE_HOME": str(cache),
    }


def validate_agent_python(agent_python: Path, isolation_dir: Path, scaffold_dir: Path) -> None:
    verify_python_import_origin(agent_python, project_root_from_package())
    modules = repr(_FORBIDDEN_AGENT_MODULES)
    probe = (
        "import importlib.util,sys;"
        f"found=[name for name in {modules} if importlib.util.find_spec(name) is not None];"
        "print(','.join(found));sys.exit(3 if found else 0)"
    )
    result = subprocess.run(
        [str(agent_python), "-I", "-c", probe],
        env=scrubbed_environment(agent_python, isolation_dir, scaffold_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        found = result.stdout.strip() or result.stderr.strip() or "probe failed"
        raise AgentProcessError(f"agent Python contains forbidden simulator packages: {found}")


@dataclass(frozen=True)
class GatewayConfig:
    scaffold_path: Path
    endpoints: Mapping[str, ToolEndpoint]
    agent_python: Path
    isolation_dir: Path
    expected_action_spec: CanonicalActionSpec
    max_horizon: int
    max_execution_count: int
    start_timeout_s: float = 15.0
    call_timeout_s: float = 120.0
    stderr_path: Path | None = None
    sandbox_limits: SandboxLimits = SandboxLimits.agent_default()

    def __post_init__(self) -> None:
        path = Path(self.scaffold_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise StrictSchemaError("gateway.scaffold_path: expected regular file")
        agent_python = Path(self.agent_python).resolve()
        if not agent_python.is_file():
            raise StrictSchemaError("gateway.agent_python: expected dedicated conda Python")
        isolation_dir = Path(self.isolation_dir).resolve()
        if isolation_dir == path.parent or isolation_dir in path.parents:
            raise StrictSchemaError("gateway.isolation_dir: must not contain editable scaffold")
        if not isinstance(self.expected_action_spec, CanonicalActionSpec):
            raise StrictSchemaError("gateway.expected_action_spec: expected CanonicalActionSpec")
        if type(self.max_horizon) is not int or self.max_horizon < 1:
            raise StrictSchemaError("gateway.max_horizon: expected positive int")
        if type(self.max_execution_count) is not int or not 1 <= self.max_execution_count <= self.max_horizon:
            raise StrictSchemaError("gateway.max_execution_count: expected 1..max_horizon")
        if type(self.start_timeout_s) not in (int, float) or float(self.start_timeout_s) <= 0:
            raise StrictSchemaError("gateway.start_timeout_s: expected positive number")
        if type(self.call_timeout_s) not in (int, float) or float(self.call_timeout_s) <= 0:
            raise StrictSchemaError("gateway.call_timeout_s: expected positive number")
        if not isinstance(self.sandbox_limits, SandboxLimits):
            raise StrictSchemaError("gateway.sandbox_limits: expected SandboxLimits")
        object.__setattr__(self, "scaffold_path", path)
        object.__setattr__(self, "endpoints", dict(self.endpoints))
        object.__setattr__(self, "agent_python", agent_python)
        object.__setattr__(self, "isolation_dir", isolation_dir)
        object.__setattr__(self, "start_timeout_s", float(self.start_timeout_s))
        object.__setattr__(self, "call_timeout_s", float(self.call_timeout_s))
        if self.stderr_path is not None:
            object.__setattr__(self, "stderr_path", Path(self.stderr_path).resolve())


class AgentProcessGateway:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: Any = None
        self._sequence = 0
        self.last_events: tuple[AgentEvent, ...] = ()
        self._sessions: set[str] = set()
        self._toolbox: Toolbox | None = None

    def start(self) -> None:
        if self._process is not None:
            raise AgentProcessError("agent process already started")
        validate_agent_python(
            self.config.agent_python,
            self.config.isolation_dir,
            self.config.scaffold_path.parent,
        )
        try:
            self._toolbox = Toolbox(self.config.endpoints)
        except Exception as exc:
            raise AgentProcessError(f"trusted tool identity preflight failed: {exc}") from exc
        if self.config.stderr_path is None:
            stderr: Any = subprocess.DEVNULL
        else:
            self.config.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = self.config.stderr_path.open("ab", buffering=0)
            stderr = self._stderr_file
        command = [
            str(self.config.agent_python),
            "-m",
            "robot_auto_evolve.agent.worker",
            "--scaffold",
            str(self.config.scaffold_path),
        ]
        package_root = Path(__file__).resolve().parents[2]
        sandboxed_command = sandbox_command(
            command,
            cwd=self.config.scaffold_path.parent,
            environment=scrubbed_environment(
                self.config.agent_python,
                self.config.isolation_dir,
                self.config.scaffold_path.parent,
            ),
            isolation_dir=self.config.isolation_dir,
            mounts=[
                *executable_mounts(self.config.agent_python, include_prefix=True),
                SandboxMount(package_root / "robot_auto_evolve" / "__init__.py"),
                SandboxMount(package_root / "robot_auto_evolve" / "runtime_paths.py"),
                SandboxMount(package_root / "robot_auto_evolve" / "agent"),
                SandboxMount(package_root / "robot_auto_evolve" / "protocol"),
                SandboxMount(package_root / "robot_auto_evolve" / "services"),
                SandboxMount(self.config.scaffold_path),
                *isolated_state_mounts(self.config.isolation_dir),
            ],
            limits=self.config.sandbox_limits,
        )
        self._process = subprocess.Popen(
            sandboxed_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            cwd=self.config.isolation_dir,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
            start_new_session=True,
        )
        _register_process(self._process, f"agent:{self.config.isolation_dir.name}")
        try:
            response = self._rpc(
                "initialize",
                {"capabilities": self._toolbox.relay_declaration()},
                timeout_s=self.config.start_timeout_s,
            )
            if response != {"ready": True}:
                raise AgentProcessError("agent process returned invalid handshake")
        except Exception:
            self.close(force=True)
            raise

    def _rpc(
        self,
        operation: str,
        payload: Mapping[str, Any],
        timeout_s: float | None = None,
        relay_context: tuple[str, str] | None = None,
    ) -> Any:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise AgentProcessError("agent process is not started")
        if process.poll() is not None:
            raise AgentProcessError(f"agent process exited with code {process.returncode}")
        self._sequence += 1
        sequence = self._sequence
        duration = self.config.call_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + duration
        try:
            write_frame(process.stdin.fileno(), {"sequence": sequence, "operation": operation, "payload": dict(payload)})
            while True:
                response = read_frame(process.stdout.fileno(), max(0.0, deadline - time.monotonic()))
                if isinstance(response, Mapping) and set(response) == {
                    "type",
                    "relay_id",
                    "capability",
                    "operation",
                    "payload",
                    "session_id",
                    "request_id",
                }:
                    if response["type"] != "tool_request" or relay_context is None:
                        raise AgentProcessError("agent process sent an unauthorized tool relay")
                    session_id, request_id = relay_context
                    if response["session_id"] != session_id or response["request_id"] != request_id:
                        raise AgentProcessError("agent process tool relay identity mismatch")
                    toolbox = self._toolbox
                    if toolbox is None:
                        raise AgentProcessError("trusted tool broker is unavailable")
                    try:
                        relay_result = toolbox.dispatch_relay(
                            response["capability"],
                            response["operation"],
                            response["payload"],
                            session_id,
                            request_id,
                        )
                        relay_response = {
                            "type": "tool_response",
                            "relay_id": response["relay_id"],
                            "ok": True,
                            "result": relay_result,
                            "error": None,
                        }
                    except Exception as exc:
                        relay_response = {
                            "type": "tool_response",
                            "relay_id": response["relay_id"],
                            "ok": False,
                            "result": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    write_frame(process.stdin.fileno(), relay_response)
                    continue
                break
        except Exception as exc:
            raise AgentProcessError(f"agent process communication failed: {exc}") from exc
        if not isinstance(response, Mapping) or set(response) != {"sequence", "ok", "result", "error"}:
            raise AgentProcessError("agent process returned invalid envelope")
        if response["sequence"] != sequence:
            raise AgentProcessError("agent process sequence mismatch")
        if response["ok"] is not True:
            error = response["error"]
            if isinstance(error, Mapping) and set(error) == {"message", "events"}:
                events = tuple(AgentEvent.from_mapping(event) for event in error["events"])
                self.last_events = events
                raise AgentProcessError(str(error["message"]), events)
            raise AgentProcessError(str(error))
        return response["result"]

    def act(
        self,
        observation: FairObservation,
        session_id: str,
        request_id: str | None = None,
    ) -> CanonicalActionChunk:
        return self.act_with_events(observation, session_id, request_id).action

    def act_with_events(
        self,
        observation: FairObservation,
        session_id: str,
        request_id: str | None = None,
    ) -> AgentStepResult:
        if not isinstance(observation, FairObservation):
            raise StrictSchemaError("gateway.observation: expected FairObservation")
        if session_id not in self._sessions:
            raise AgentProcessError("agent session must be reset with a policy seed before act")
        request = AgentRequest(
            request_id=request_id or uuid.uuid4().hex,
            session_id=session_id,
            observation=observation,
        )
        toolbox = self._toolbox
        if toolbox is None:
            raise AgentProcessError("trusted tool broker is unavailable")
        toolbox.begin_step(observation.step_index, request.request_id, session_id)
        try:
            result = AgentStepResult.from_mapping(
                self._rpc("act", request.to_mapping(), relay_context=(session_id, request.request_id))
            )
        finally:
            toolbox.finish_step()
        self.last_events = result.events
        if result.action.request_id != request.request_id or result.action.session_id != request.session_id:
            raise AgentProcessError("agent response identity mismatch")
        if result.action.start_step != observation.step_index:
            raise AgentProcessError("agent response start step mismatch")
        if result.action.spec != self.config.expected_action_spec:
            raise AgentProcessError("agent response action specification differs from frozen profile")
        if result.action.horizon > self.config.max_horizon:
            raise AgentProcessError("agent response horizon exceeds frozen profile limit")
        if result.action.execution_count > self.config.max_execution_count:
            raise AgentProcessError("agent response execution count exceeds frozen profile limit")
        return result

    def reset(self, session_id: str, policy_seed: int, task_id: str) -> None:
        if type(session_id) is not str or not session_id:
            raise StrictSchemaError("gateway.session_id: expected nonempty string")
        if type(policy_seed) is not int or policy_seed < 0:
            raise StrictSchemaError("gateway.policy_seed: expected nonnegative int")
        if type(task_id) is not str or not task_id:
            raise StrictSchemaError("gateway.task_id: expected nonempty string")
        toolbox = self._toolbox
        if toolbox is None:
            raise AgentProcessError("trusted tool broker is unavailable")
        toolbox.reset_policy(session_id, policy_seed, task_id)
        try:
            result = self._rpc("reset", {"session_id": session_id, "policy_seed": policy_seed, "task_id": task_id})
        except BaseException:
            try:
                toolbox.close_session(session_id)
            finally:
                raise
        if result != {"reset": True}:
            raise AgentProcessError("agent process returned invalid reset response")
        self._sessions.add(session_id)

    def close(self, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if not force and process.poll() is None:
            try:
                self._rpc("close", {}, timeout_s=2.0)
            except AgentProcessError:
                force = True
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3.0)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        _unregister_process(process)
        self._process = None
        toolbox = self._toolbox
        self._toolbox = None
        if toolbox is not None:
            for session_id in sorted(self._sessions):
                try:
                    toolbox.close_session(session_id)
                except Exception:
                    if not force:
                        raise
        self._sessions.clear()

    def __enter__(self) -> "AgentProcessGateway":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(force=exc is not None)
