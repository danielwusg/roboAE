from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_auto_evolve.agent import AgentEvent
from robot_auto_evolve.agent.sandbox import (
    SandboxLimits,
    SandboxMount,
    executable_mounts,
    isolated_state_mounts,
    sandbox_command,
)
from robot_auto_evolve.process_lifecycle import register_owned_process, unregister_owned_process
from robot_auto_evolve.provenance import EpisodeKey
from robot_auto_evolve.protocol import (
    CameraObservation,
    CanonicalActionChunk,
    CanonicalActionSpec,
    FairObservation,
    RobotProprioception,
    RobotStateSpec,
    RobotStateVector,
)

from .benchmark_evidence import BenchmarkPublicEvidence
from .evidence import PublicEpisodeEvidence, PublicEvolutionEvidence, PublicStepEvidence
from .hashing import file_sha256
from .models import EvaluationResult
from .relay import (
    CLAUDE_AUTH_TRACE,
    CLAUDE_KEY_HELPER,
    CLAUDE_RELAY_BASE_URL,
    PROVIDER_PATH,
    RelayLimits,
    RelaySession,
    relay_provenance,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class CommandEvaluator:
    def __init__(
        self,
        command: Sequence[str],
        timeout_s: float,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command or timeout_s <= 0:
            raise ValueError("evaluator command is empty")
        self.command = tuple(command)
        self.timeout_s = float(timeout_s)
        self.environment = None if environment is None else dict(environment)

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)

    def evaluate(self, scaffold_dir: Path, split: str, output_dir: Path) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=False)
        command = [
            *self.command,
            "--scaffold",
            str(scaffold_dir.resolve()),
            "--split",
            split,
            "--output",
            str(output_dir.resolve()),
        ]
        started_ns = time.time_ns()
        failure: BaseException | None = None
        failure_traceback = None
        environment = dict(os.environ if self.environment is None else self.environment)
        environment.pop("PYTHONPATH", None)
        environment.pop("VIRTUAL_ENV", None)
        process: subprocess.Popen[bytes] | None = None
        timed_out = False
        with (output_dir / "stdout.log").open("wb") as stdout, (output_dir / "stderr.log").open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    start_new_session=True,
                )
                register_owned_process(process, f"command-evaluator:{split}")
                returncode = process.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                if process is None:
                    raise
                self._terminate(process)
                returncode = int(process.returncode)
            except BaseException:
                if process is not None:
                    self._terminate(process)
                    returncode = int(process.returncode)
                else:
                    returncode = -1
                failure = sys.exc_info()[1]
                failure_traceback = sys.exc_info()[2]
        if process is not None:
            unregister_owned_process(process)
        else:
            raise RuntimeError("evaluator process did not start")
        _write_json(
            output_dir / "process.json",
            {
                "command": command,
                "started_ns": started_ns,
                "finished_ns": time.time_ns(),
                "returncode": returncode,
                "timed_out": timed_out,
                "process_group": process.pid,
                "interrupted": None if failure is None else type(failure).__name__,
            },
        )
        if failure is not None:
            raise failure.with_traceback(failure_traceback)
        if timed_out:
            raise RuntimeError(f"evaluator timed out after {self.timeout_s} seconds")
        if returncode != 0:
            raise RuntimeError(f"evaluator exited with code {returncode}")
        report = EvaluationResult.load(output_dir / "report.json")
        if report.split != split:
            raise RuntimeError("evaluator report split mismatch")
        return report


class FixtureEvaluator:
    _VARIANT = re.compile(r'^FIXTURE_VARIANT\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE)

    def __init__(self, variants: Mapping[str, Mapping[str, Sequence[bool]]]) -> None:
        self.variants = {
            str(variant): {str(split): tuple(values) for split, values in splits.items()}
            for variant, splits in variants.items()
        }

    @classmethod
    def load(cls, path: Path) -> "FixtureEvaluator":
        value = json.loads(path.read_text())
        if not isinstance(value, Mapping) or set(value) != {"variants"}:
            raise ValueError("fixture evaluations must contain only variants")
        return cls(value["variants"])

    def evaluate(self, scaffold_dir: Path, split: str, output_dir: Path) -> EvaluationResult:
        source = (scaffold_dir / "scaffold.py").read_text()
        match = self._VARIANT.search(source)
        if match is None:
            raise RuntimeError("fixture scaffold has no FIXTURE_VARIANT")
        variant = match.group(1)
        values = self.variants[variant][split]
        output_dir.mkdir(parents=True, exist_ok=False)
        outcomes = []
        for index, success in enumerate(values):
            task_index = index % 2
            key = EpisodeKey(
                split=split,
                task_id=f"fixture_task_{task_index}",
                scenario_id=f"scenario_{index // 2}",
                environment_seed=index,
                policy_seed=1000 + index,
                replicate_id="fixture",
                horizon=10,
                protocol="fixture-v1",
            )
            from robot_auto_evolve.evaluation import EpisodeOutcome

            outcomes.append(EpisodeOutcome(key, bool(success)))
        evidence_hash = None
        evidence_count = 0
        if split == "evolve":
            joint_spec = RobotStateSpec(
                name="joint_position",
                quantity="joint_position",
                frame_id="robot",
                reference_frame="robot_base",
                component_names=("joint_0",),
                units=("radian",),
                representation="vector",
                quaternion_order="none",
            )
            action_spec = CanonicalActionSpec(
                arm_names=("arm",),
                channel_names=("x",),
                channel_semantics=("delta",),
                coordinate_frame="robot_base",
                translation_unit="meter",
                rotation_representation="none",
                quaternion_order="none",
                gripper_convention="none",
                value_encoding="physical",
                controller_output_scale=(),
                control_period_s=0.05,
            )
            episodes = []
            for index, outcome in enumerate(outcomes):
                observation = FairObservation(
                    episode_id=outcome.key.artifact_id(),
                    step_index=0,
                    timestamp_ns=index,
                    instruction=f"fixture instruction for {outcome.key.task_id}",
                    cameras={
                        "agent": CameraObservation(
                            frame_id="agent_camera",
                            optical_convention="opencv_rdf",
                            rgb=np.full((8, 8, 3), index % 255, dtype=np.uint8),
                            depth_m=None,
                            depth_valid=None,
                            intrinsics=None,
                            camera_to_world=None,
                        )
                    },
                    proprioception=RobotProprioception(
                        (RobotStateVector(joint_spec, np.asarray([index / 100.0], dtype=np.float32)),)
                    ),
                )
                action = CanonicalActionChunk(
                    request_id=f"fixture-{index}",
                    session_id=outcome.key.artifact_id(),
                    start_step=0,
                    spec=action_spec,
                    values=np.asarray([[1.0 if outcome.success else 0.0]], dtype=np.float32),
                    execution_count=1,
                )
                event = AgentEvent(0, "decision", "ok", f"fixture variant {variant}")
                episodes.append(
                    PublicEpisodeEvidence(
                        outcome,
                        "success" if outcome.success else "horizon",
                        (PublicStepEvidence(observation, action, (event,)),),
                    )
                )
            by_outcome = {episode.outcome: episode for episode in episodes}
            evidence = PublicEvolutionEvidence.create(
                output_dir / "public_evidence",
                tuple(outcomes),
                by_outcome.__getitem__,
            )
            evidence_hash = evidence.bundle_sha256
            evidence_count = len(evidence.episodes)
        report = EvaluationResult(
            split,
            tuple(outcomes),
            {"backend": "fixture", "variant": variant},
            evidence_hash,
            evidence_count,
        )
        _write_json(output_dir / "report.json", report.to_mapping())
        return report


class FixtureRevisionBackend:
    def __init__(self, revisions: Sequence[str]) -> None:
        self.revisions = tuple(revisions)

    @classmethod
    def load(cls, path: Path) -> "FixtureRevisionBackend":
        value = json.loads(path.read_text())
        if not isinstance(value, Mapping) or set(value) != {"revisions"}:
            raise ValueError("fixture revisions must contain only revisions")
        return cls(tuple(str(item) for item in value["revisions"]))

    def revise(self, prompt: str, candidate_dir: Path, log_dir: Path, attempt_index: int) -> None:
        del prompt
        try:
            source = self.revisions[attempt_index - 1]
        except IndexError as exc:
            raise RuntimeError(f"fixture has no revision for attempt {attempt_index}") from exc
        (candidate_dir / "scaffold.py").write_text(source)
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_json(log_dir / "fixture_revision.json", {"attempt": attempt_index, "bytes": len(source.encode())})


def claude_environment(
    executable: Path,
    isolation_dir: Path,
    *,
    relay_socket: str | None = None,
) -> dict[str, str]:
    del isolation_dir
    home = Path("/sandbox-state/home")
    cache = Path("/sandbox-state/cache")
    temporary = Path("/sandbox-state/tmp")
    environment = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.pathsep.join((str(executable.parent), "/usr/bin", "/bin")),
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(cache),
    }
    if relay_socket is not None:
        if not relay_socket.startswith("/"):
            raise ValueError("invalid Claude relay environment")
        environment["ANTHROPIC_UNIX_SOCKET"] = relay_socket
        environment["ANTHROPIC_BASE_URL"] = CLAUDE_RELAY_BASE_URL
    return environment


@dataclass(frozen=True)
class LaunchCheck:
    pid: int
    terminated_process_group: bool
    returncode: int


@dataclass(frozen=True)
class OfflineRelayProbe:
    pid: int
    terminated_process_group: bool
    returncode: int
    method: str
    path: str
    model: str
    body_bytes: int
    provider_contacted: bool
    network_isolated: bool


@dataclass
class _RunningClaude:
    process: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any
    started_ns: int
    relay_mode: str


class ClaudeRevisionBackend:
    def __init__(
        self,
        executable: Path,
        isolation_dir: Path,
        coding_model: str,
        timeout_s: float = 900.0,
        max_turns: int = 30,
        credential_dir: Path | None = None,
        relay_limits: RelayLimits | None = None,
        sandbox_limits: SandboxLimits = SandboxLimits.revision_default(),
        public_evidence_format: str = "episode_samples",
    ) -> None:
        self.executable = Path(executable).resolve()
        self.executable_sha256 = file_sha256(self.executable) if self.executable.is_file() else None
        self.isolation_dir = Path(isolation_dir).resolve()
        self.timeout_s = float(timeout_s)
        self.max_turns = max_turns
        self.coding_model = str(coding_model)
        self.credential_dir = None if credential_dir is None else Path(credential_dir).absolute()
        self.relay_limits = relay_limits or RelayLimits(
            deadline_s=self.timeout_s,
            provider_timeout_s=min(300.0, self.timeout_s),
        )
        self.sandbox_limits = sandbox_limits
        self.public_evidence_format = public_evidence_format
        if (
            not self.executable.is_file()
            or self.timeout_s <= 0
            or type(self.max_turns) is not int
            or self.max_turns < 1
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.coding_model) is None
            or not isinstance(self.relay_limits, RelayLimits)
            or not isinstance(sandbox_limits, SandboxLimits)
            or self.public_evidence_format not in {"episode_samples", "full_benchmark"}
        ):
            raise ValueError("invalid Claude backend configuration")

    def _command(self, prompt: str, relay_enabled: bool = False) -> list[str]:
        command = [
            str(self.executable),
            "-p",
            prompt,
            "--output-format",
            "text",
            "--max-turns",
            str(self.max_turns),
            "--model",
            self.coding_model,
            "--safe-mode",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--no-session-persistence",
            "--tools",
            "Read,Edit",
            "--allowedTools",
            "Read,Edit",
            "--permission-mode",
            "acceptEdits",
            "--no-chrome",
        ]
        if relay_enabled:
            command.extend(
                [
                    "--bare",
                    "--settings",
                    json.dumps({"apiKeyHelper": CLAUDE_KEY_HELPER}, sort_keys=True, separators=(",", ":")),
                ]
            )
        return command

    def _start(
        self,
        prompt: str,
        candidate_dir: Path,
        log_dir: Path,
        require_public_input: bool = False,
        relay: RelaySession | None = None,
    ) -> _RunningClaude:
        candidate_dir = Path(candidate_dir).resolve()
        if not candidate_dir.is_dir() or candidate_dir.is_symlink():
            raise ValueError("Claude candidate must be a regular directory")
        if (
            self.isolation_dir == candidate_dir
            or self.isolation_dir in candidate_dir.parents
            or candidate_dir in self.isolation_dir.parents
        ):
            raise ValueError("Claude isolation and candidate directories must be disjoint")
        log_dir.mkdir(parents=True, exist_ok=True)
        if self.public_evidence_format == "full_benchmark":
            public_input = candidate_dir.parent / "public_input.json"
            public_evidence_dirs = [candidate_dir.parent / "incumbent_evidence"]
            previous = candidate_dir.parent / "previous_candidate_evidence"
            if previous.exists():
                public_evidence_dirs.append(previous)
        else:
            public_input = candidate_dir.parent / "public_input.txt"
            public_evidence_dirs = [candidate_dir.parent / "public_evidence"]
        if require_public_input and (
            not public_input.is_file() or public_input.is_symlink() or public_input.stat().st_mode & 0o222
        ):
            raise ValueError("Claude revision requires a read-only regular public input")
        if require_public_input:
            for public_evidence in public_evidence_dirs:
                if self.public_evidence_format == "full_benchmark":
                    BenchmarkPublicEvidence.load(public_evidence)
                else:
                    PublicEvolutionEvidence.load(public_evidence)
                if any(path.stat().st_mode & 0o222 for path in (public_evidence, *public_evidence.rglob("*"))):
                    raise ValueError("Claude revision requires a read-only public evidence bundle")
        stdout = None
        stderr = None
        process = None
        ready_read = None
        ready_write = None
        key_material = None
        started_ns = time.time_ns()
        try:
            stdout = (log_dir / "claude.stdout.log").open("wb")
            stderr = (log_dir / "claude.stderr.log").open("wb")
            ready_read, ready_write = os.pipe()
            if relay is not None:
                key_material = relay.claude_key_material()
            command = sandbox_command(
                self._command(
                    prompt,
                    relay_enabled=relay is not None,
                ),
                cwd=candidate_dir,
                environment=claude_environment(
                    self.executable,
                    self.isolation_dir,
                    relay_socket=None if relay is None else "/claude-relay/api.sock",
                ),
                isolation_dir=self.isolation_dir,
                mounts=[
                    *executable_mounts(self.executable, include_prefix=False),
                    SandboxMount(candidate_dir, writable=True),
                    *([SandboxMount(public_input)] if public_input.is_file() else []),
                    *(
                        [SandboxMount(public_evidence) for public_evidence in public_evidence_dirs]
                        if require_public_input
                        else []
                    ),
                    *([SandboxMount(relay.socket_mount, target=Path("/claude-relay"))] if relay is not None else []),
                    *(
                        [
                            SandboxMount(key_material.helper, target=Path(CLAUDE_KEY_HELPER)),
                            SandboxMount(key_material.interpreter),
                            SandboxMount(key_material.shell, target=Path("/bin/sh")),
                            SandboxMount(key_material.trace, writable=True, target=Path(CLAUDE_AUTH_TRACE)),
                        ]
                        if key_material is not None
                        else []
                    ),
                    *isolated_state_mounts(self.isolation_dir),
                ],
                limits=self.sandbox_limits,
                ready_fd=ready_write,
            )
            process = subprocess.Popen(
                command,
                cwd=self.isolation_dir,
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                pass_fds=tuple(
                    value for value in (ready_write, None if key_material is None else key_material.descriptor)
                    if value is not None
                ),
            )
            register_owned_process(process, "claude")
            os.close(ready_write)
            ready_write = None
            if key_material is not None:
                os.close(key_material.descriptor)
                key_material = None
            ready, _, _ = select.select([ready_read], [], [], min(15.0, self.timeout_s))
            marker = os.read(ready_read, 1) if ready else b""
            os.close(ready_read)
            ready_read = None
            if marker != b"R":
                raise RuntimeError("Claude sandbox failed before executable launch")
            running = _RunningClaude(
                process,
                stdout,
                stderr,
                started_ns,
                "disabled" if relay is None else ("offline" if relay.offline else "online"),
            )
            self._write_process_record(log_dir, running, status="running", termination_reason=None)
            return running
        except BaseException:
            if process is not None:
                self._terminate_group(process)
                unregister_owned_process(process)
            if ready_write is not None:
                os.close(ready_write)
            if ready_read is not None:
                os.close(ready_read)
            if key_material is not None:
                os.close(key_material.descriptor)
            if stdout is not None:
                stdout.close()
            if stderr is not None:
                stderr.close()
            self._write_process_record(
                log_dir,
                None
                if process is None
                else _RunningClaude(
                    process,
                    stdout,
                    stderr,
                    started_ns,
                    "disabled" if relay is None else ("offline" if relay.offline else "online"),
                ),
                status="startup_failed",
                termination_reason="startup_failure",
            )
            raise

    def _write_process_record(
        self,
        log_dir: Path,
        running: _RunningClaude | None,
        *,
        status: str,
        termination_reason: str | None,
        interrupted: BaseException | None = None,
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        process = None if running is None else running.process
        _write_json(
            log_dir / "claude_process.json",
            {
                "schema_version": 1,
                "status": status,
                "pid": None if process is None else process.pid,
                "process_group": None if process is None else process.pid,
                "started_ns": None if running is None else running.started_ns,
                "finished_ns": None if status == "running" else time.time_ns(),
                "returncode": None if process is None else process.poll(),
                "termination_reason": termination_reason,
                "interrupted": None if interrupted is None else type(interrupted).__name__,
                "executable": {
                    "path": str(self.executable),
                    "sha256": self.executable_sha256,
                },
                "coding_model": self.coding_model,
                "timeout_s": self.timeout_s,
                "max_turns": self.max_turns,
                "tools": ["Read", "Edit"],
                "public_evidence_format": self.public_evidence_format,
                "network_isolated": True,
                "relay_mode": (
                    "online" if running is None and self.credential_dir is not None else
                    "disabled" if running is None else running.relay_mode
                ),
                "relay": relay_provenance(self.coding_model, self.relay_limits),
            },
        )

    @staticmethod
    def _terminate_group(process: subprocess.Popen[bytes], immediate: bool = False) -> None:
        if process.poll() is not None:
            return
        if immediate:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3.0)
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

    def revise(self, prompt: str, candidate_dir: Path, log_dir: Path, attempt_index: int) -> None:
        del attempt_index
        relay = None
        running = None
        if self.credential_dir is not None:
            relay = RelaySession(
                isolation_dir=self.isolation_dir,
                artifact_dir=log_dir,
                coding_model=self.coding_model,
                limits=self.relay_limits,
                credential_dir=self.credential_dir,
                offline=False,
            )
            try:
                relay.start()
                if relay.process is not None:
                    register_owned_process(relay.process, "claude-relay")
            except BaseException as exc:
                self._write_process_record(
                    log_dir,
                    None,
                    status="not_started",
                    termination_reason="relay_startup_failure",
                    interrupted=exc,
                )
                raise
        try:
            running = self._start(prompt, candidate_dir, log_dir, require_public_input=True, relay=relay)
            returncode = running.process.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            if running is not None:
                self._terminate_group(running.process)
            raise RuntimeError("Claude revision timed out") from exc
        except BaseException:
            if running is not None:
                self._terminate_group(running.process)
            raise
        finally:
            interrupted = sys.exc_info()[1]
            try:
                if running is not None:
                    if running.process.poll() is None:
                        self._terminate_group(running.process)
                    running.stdout.close()
                    running.stderr.close()
                    success = interrupted is None and running.process.returncode == 0
                    self._write_process_record(
                        log_dir,
                        running,
                        status="complete" if success else "failed",
                        termination_reason=None if success else "process_exit_or_interruption",
                        interrupted=interrupted,
                    )
                    unregister_owned_process(running.process)
            finally:
                if relay is not None:
                    try:
                        relay.stop("claude_exit" if interrupted is None else "claude_failure")
                    finally:
                        if relay.process is not None:
                            unregister_owned_process(relay.process)
        if returncode != 0:
            raise RuntimeError(f"Claude exited with code {returncode}")

    def launch_only(
        self,
        candidate_dir: Path,
        log_dir: Path,
        startup_wait_s: float = 0.1,
    ) -> LaunchCheck:
        running = self._start("Reply with READY and make no changes.", candidate_dir, log_dir)
        process, stdout, stderr = running.process, running.stdout, running.stderr
        deadline = time.monotonic() + startup_wait_s
        recorded = False
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Claude exited during launch check with code {process.returncode}")
                time.sleep(0.05)
            pid = process.pid
            self._terminate_group(process, immediate=True)
            self._write_process_record(
                log_dir,
                running,
                status="launch_checked",
                termination_reason="launch_check",
            )
            recorded = True
            return LaunchCheck(pid, True, int(process.returncode))
        finally:
            interrupted = sys.exc_info()[1]
            self._terminate_group(process, immediate=True)
            unregister_owned_process(process)
            stdout.close()
            stderr.close()
            if not recorded:
                self._write_process_record(
                    log_dir,
                    running,
                    status="launch_check_failed",
                    termination_reason="launch_check",
                    interrupted=interrupted,
                )

    def offline_probe(
        self,
        candidate_dir: Path,
        log_dir: Path,
        startup_timeout_s: float = 30.0,
    ) -> OfflineRelayProbe:
        if startup_timeout_s <= 0:
            raise ValueError("offline relay probe timeout must be positive")
        relay = RelaySession(
            isolation_dir=self.isolation_dir,
            artifact_dir=log_dir,
            coding_model=self.coding_model,
            limits=RelayLimits(
                max_requests=1,
                max_request_bytes=self.relay_limits.max_request_bytes,
                max_response_bytes=self.relay_limits.max_response_bytes,
                deadline_s=startup_timeout_s,
                provider_timeout_s=min(startup_timeout_s, self.relay_limits.provider_timeout_s),
            ),
            credential_dir=None,
            offline=True,
        )
        running = None
        state = None
        try:
            relay.start()
            if relay.process is not None:
                register_owned_process(relay.process, "claude-relay")
            running = self._start(
                "Reply with READY and make no changes.",
                candidate_dir,
                log_dir,
                relay=relay,
            )
            state = relay.wait_for_request(startup_timeout_s)
            request = state["requests"][0]
            if (
                request.get("method") != "POST"
                or request.get("path") != PROVIDER_PATH
                or request.get("model") != self.coding_model
                or request.get("status") != "offline_observed"
                or state.get("provider_contacted") is not False
                or state.get("network_isolated") is not True
            ):
                raise RuntimeError("offline Claude relay request differs from the pinned transport")
            self._terminate_group(running.process, immediate=True)
            result = OfflineRelayProbe(
                pid=running.process.pid,
                terminated_process_group=True,
                returncode=int(running.process.returncode),
                method=request["method"],
                path=request["path"],
                model=request["model"],
                body_bytes=int(request["body_bytes"]),
                provider_contacted=False,
                network_isolated=True,
            )
            _write_json(log_dir / "offline_probe.json", result.__dict__)
            return result
        except BaseException:
            if running is not None:
                self._terminate_group(running.process, immediate=True)
            raise
        finally:
            interrupted = sys.exc_info()[1]
            try:
                if running is not None:
                    if running.process.poll() is None:
                        self._terminate_group(running.process, immediate=True)
                    running.stdout.close()
                    running.stderr.close()
                    self._write_process_record(
                        log_dir,
                        running,
                        status="offline_probe_complete" if interrupted is None else "offline_probe_failed",
                        termination_reason="offline_probe",
                        interrupted=interrupted,
                    )
                    unregister_owned_process(running.process)
            finally:
                try:
                    relay.stop("offline_probe")
                finally:
                    if relay.process is not None:
                        unregister_owned_process(relay.process)
