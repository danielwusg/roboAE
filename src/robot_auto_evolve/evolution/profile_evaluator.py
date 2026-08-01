from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Mapping

import numpy as np

from robot_auto_evolve.agent import AgentProcessGateway, GatewayConfig, ToolEndpoint
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation import EpisodeExecution
from robot_auto_evolve.evaluation.simulator import SimulatorProcess
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import EpisodeKey
from robot_auto_evolve.services import ReplicaScheduler

from .evidence import PublicStepEvidence


POLICY_CALL_TIMEOUT_S = 600.0
TOOL_CALL_TIMEOUT_S = 900.0
AGENT_STEP_TIMEOUT_S = 3600.0
AGENT_START_TIMEOUT_S = 3600.0
SIMULATOR_START_TIMEOUT_S = 600.0
SIMULATOR_CALL_TIMEOUT_S = 600.0
ROBOTWIN2_SIMULATOR_START_TIMEOUT_S = 1800.0
ROBOTWIN2_SIMULATOR_CALL_TIMEOUT_S = 900.0
STOP_ON_FIRST_SUCCESS = "stop_on_first_success"
FULL_HORIZON_FINAL_SUCCESS = "full_horizon_final_success"

REUSE_SIM_SAFE_SUITE_PREFIXES = ("simpler_", "libero_pro_")
REUSE_SIM_SAFE_SUITES = frozenset({"robocerebra_public60", "vlabench_xvla_tracks_1_4"})


def reuse_sim_allowed(suite: str) -> bool:
    return suite in REUSE_SIM_SAFE_SUITES or suite.startswith(REUSE_SIM_SAFE_SUITE_PREFIXES)


def success_protocol(profile: Profile) -> str:
    if profile.environment.suite.startswith("simpler_"):
        return FULL_HORIZON_FINAL_SUCCESS
    return STOP_ON_FIRST_SUCCESS


def simulator_timeouts(suite: str) -> tuple[float, float]:
    if suite == "robotwin2_demo_clean":
        return ROBOTWIN2_SIMULATOR_START_TIMEOUT_S, ROBOTWIN2_SIMULATOR_CALL_TIMEOUT_S
    return SIMULATOR_START_TIMEOUT_S, SIMULATOR_CALL_TIMEOUT_S


def resolve_render_gpu_ids(profile: Profile, value: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if not isinstance(profile, Profile) or profile.policy.deployment_mode != "replicated":
        raise StrictSchemaError("render GPU assignment requires a replicated profile")
    default = tuple(item.identity.gpu_ids[0] for item in profile.policy.replicas)
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        raise StrictSchemaError("render GPU assignment must be an integer sequence")
    result = tuple(value)
    if len(result) != len(default) or any(type(item) is not int or item < 0 for item in result):
        raise StrictSchemaError("render GPU assignment must contain one nonnegative integer per policy replica")
    if not set(result) <= set(profile.resources.gpu_ids):
        raise StrictSchemaError("render GPU assignment falls outside the profile GPU pool")
    return result


VIDEO_FRAME_RATE = 20
VIDEO_CRF = "14"
VIDEO_PIXEL_FORMAT = "yuv444p"
VIDEO_PRESET = "veryfast"


def _round_list(values, places: int = 5) -> list:
    return [round(float(item), places) for item in values]


def _state_of(observation) -> dict:
    return {
        vector.spec.name: {
            "components": list(vector.spec.component_names),
            "values": _round_list(vector.values),
        }
        for vector in observation.proprioception.vectors
    }


def _depth_of(observation) -> dict:
    summary = {}
    for name in sorted(observation.cameras):
        camera = observation.cameras[name]
        if camera.depth_m is None or camera.depth_valid is None:
            continue
        depth = np.asarray(camera.depth_m, dtype=np.float64)
        valid = np.asarray(camera.depth_valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
        usable = depth[valid]
        summary[name] = {
            "valid_fraction": round(float(valid.mean()), 4),
            "min_m": None if not usable.size else round(float(usable.min()), 4),
            "median_m": None if not usable.size else round(float(np.median(usable)), 4),
            "max_m": None if not usable.size else round(float(usable.max()), 4),
        }
    return summary


def _readable_trace_bytes(
    steps,
    key: EpisodeKey,
    success: bool,
    termination: str,
    pictures: Mapping[str, list[str]] | None = None,
    picture_error: str | None = None,
) -> bytes:
    first = steps[0].observation
    stored = {} if pictures is None else pictures
    lines = [
        json.dumps(
            {
                "kind": "episode_trace",
                "picture_error": picture_error,
                "episode_id": key.artifact_id(),
                "task_id": key.task_id,
                "success": bool(success),
                "termination": termination,
                "n_action_steps": sum(1 for step in steps if step.action is not None),
                "cameras": {
                    name: {
                        "width": int(first.cameras[name].rgb.shape[1]),
                        "height": int(first.cameras[name].rgb.shape[0]),
                        "has_depth": first.cameras[name].depth_m is not None,
                        "pictures": stored.get(name, []),
                    }
                    for name in sorted(first.cameras)
                },
                "robot_state": [vector.spec.name for vector in first.proprioception.vectors],
            },
            sort_keys=True,
        )
    ]
    for step in steps:
        action_repr = None
        if step.action is not None:
            action_repr = {
                "values": [[round(float(x), 5) for x in row] for row in step.action.values.tolist()],
                "channels": list(step.action.spec.channel_names),
                "execution_count": step.action.execution_count,
            }
        lines.append(
            json.dumps(
                {
                    "step": step.observation.step_index,
                    "instruction": step.observation.instruction,
                    "action": action_repr,
                    "state": _state_of(step.observation),
                    "depth": _depth_of(step.observation),
                    "events": [
                        {
                            "type": event.event_type,
                            "status": event.status,
                            "detail": event.detail,
                            "capability": event.capability,
                            "result": None if event.result is None else dict(event.result),
                        }
                        for event in step.events
                    ],
                },
                sort_keys=True,
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _encode_video(frames, width: int, height: int) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "camera.mp4"
        command = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-framerate", str(VIDEO_FRAME_RATE),
            "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", VIDEO_PRESET, "-threads", "1",
            "-pix_fmt", VIDEO_PIXEL_FORMAT, "-crf", VIDEO_CRF,
            str(target),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            process.stdin.close()
            error = process.stderr.read()
            if process.wait(timeout=300) != 0:
                raise RuntimeError(error.decode("utf-8", "replace").strip()[:300] or "ffmpeg failed")
        except BaseException:
            process.kill()
            process.wait(timeout=30)
            raise
        finally:
            process.stderr.close()
        return target.read_bytes()


def _png(rgb) -> bytes:
    from PIL import Image
    import io

    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", compress_level=9)
    return output.getvalue()


def _frame_artifacts(steps) -> dict:
    frames: dict[str, bytes] = {}
    for position in sorted({0, len(steps) - 1}):
        obs = steps[position].observation
        if not obs.cameras:
            continue
        camera = sorted(obs.cameras)[0]
        frames[f"frame-{obs.step_index:08d}.png"] = _png(obs.cameras[camera].rgb)
    return frames


def _camera_artifacts(steps) -> tuple[dict, dict, str | None]:
    first = steps[0].observation
    if not first.cameras:
        return {}, {}, None
    try:
        artifacts: dict[str, bytes] = {}
        pictures: dict[str, list[str]] = {}
        for name in sorted(first.cameras):
            height, width = first.cameras[name].rgb.shape[:2]
            filename = f"camera-{name}.mp4"
            artifacts[filename] = _encode_video(
                (step.observation.cameras[name].rgb for step in steps), width, height
            )
            pictures[name] = [filename]
        return artifacts, pictures, None
    except Exception as exc:
        artifacts = _frame_artifacts(steps)
        fallback = sorted(first.cameras)[0]
        return artifacts, {fallback: sorted(artifacts)}, f"{type(exc).__name__}: {exc}"


class AgentGatewayPool:

    def __init__(self, pool_root: Path) -> None:
        self.pool_root = Path(pool_root)
        self._local = threading.local()
        self._all: list[AgentProcessGateway] = []
        self._lock = threading.Lock()

    def isolation_dir(self, replica_id: str) -> Path:
        return self.pool_root / f"t{threading.get_ident()}_{replica_id}"

    def acquire(self, replica_id: str, make_gateway) -> AgentProcessGateway:
        pool = getattr(self._local, "pool", None)
        if pool is None:
            pool = {}
            self._local.pool = pool
        gateway = pool.get(replica_id)
        if gateway is None:
            gateway = make_gateway()
            gateway.start()
            pool[replica_id] = gateway
            with self._lock:
                self._all.append(gateway)
        return gateway

    def evict(self, replica_id: str, gateway: "AgentProcessGateway") -> None:
        pool = getattr(self._local, "pool", None)
        if pool is not None and pool.get(replica_id) is gateway:
            del pool[replica_id]
        with self._lock:
            if gateway in self._all:
                self._all.remove(gateway)
        try:
            gateway.close(force=True)
        except Exception:
            pass

    def close_all(self) -> None:
        with self._lock:
            gateways = list(self._all)
            self._all.clear()
        for gateway in gateways:
            try:
                gateway.close(force=True)
            except Exception:
                pass
        shutil.rmtree(self.pool_root, ignore_errors=True)


class SimulatorProcessPool:

    def __init__(self, pool_root: Path) -> None:
        self.pool_root = Path(pool_root)
        self._local = threading.local()
        self._all: list["SimulatorProcess"] = []
        self._lock = threading.Lock()

    def runtime_dir(self, render_gpu_id: int) -> Path:
        return self.pool_root / f"t{threading.get_ident()}_g{render_gpu_id}"

    def acquire(self, render_gpu_id: int, episode, make_simulator) -> "SimulatorProcess":
        pool = getattr(self._local, "pool", None)
        if pool is None:
            pool = {}
            self._local.pool = pool
        simulator = pool.get(render_gpu_id)
        if simulator is None:
            simulator = make_simulator()
            simulator.start()
            pool[render_gpu_id] = simulator
            with self._lock:
                self._all.append(simulator)
        else:
            simulator.reinitialize(episode)
        return simulator

    def evict(self, render_gpu_id: int, simulator: "SimulatorProcess") -> None:
        pool = getattr(self._local, "pool", None)
        if pool is not None and pool.get(render_gpu_id) is simulator:
            del pool[render_gpu_id]
        with self._lock:
            if simulator in self._all:
                self._all.remove(simulator)
        try:
            simulator.close(force=True)
        except Exception:
            pass

    def close_all(self) -> None:
        with self._lock:
            simulators = list(self._all)
            self._all.clear()
        for simulator in simulators:
            try:
                simulator.close(force=True)
            except Exception:
                pass
        shutil.rmtree(self.pool_root, ignore_errors=True)


class ProfileEpisodeRunner:
    def __init__(
        self,
        profile: Profile,
        scaffold_dir: Path,
        scheduler: ReplicaScheduler,
        tool_endpoints: Mapping[str, ToolEndpoint],
        *,
        agent_python: Path,
        simulator_python: Path,
        simulator_source: Path | None,
        runtime_root: Path,
        replica_assignments: Mapping[str, str],
        render_gpu_assignments: Mapping[str, int],
        gateway_pool: "AgentGatewayPool | None" = None,
        simulator_pool: "SimulatorProcessPool | None" = None,
    ) -> None:
        self.profile = profile
        self.scaffold_path = Path(scaffold_dir).resolve() / "scaffold.py"
        self.scheduler = scheduler
        self.tool_endpoints = dict(tool_endpoints)
        self.agent_python = Path(agent_python).resolve()
        self.simulator_python = Path(simulator_python).resolve()
        self.simulator_source = None if simulator_source is None else Path(simulator_source).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.replica_assignments = dict(replica_assignments)
        self.render_gpu_assignments = dict(render_gpu_assignments)
        self.gateway_pool = gateway_pool
        self.simulator_pool = simulator_pool
        if set(self.render_gpu_assignments) != set(self.replica_assignments):
            raise StrictSchemaError("profile runner render and policy session assignments differ")
        self.success_protocol = success_protocol(profile)
        self.simulator_start_timeout_s, self.simulator_call_timeout_s = simulator_timeouts(
            profile.environment.suite
        )

    @staticmethod
    def _request_id(key: EpisodeKey, step_index: int) -> str:
        return hashlib.sha256(f"{key.artifact_id()}\0{step_index}".encode()).hexdigest()

    def _gateway_config(self, endpoints, isolation_dir: Path, stderr_path: Path) -> GatewayConfig:
        return GatewayConfig(
            scaffold_path=self.scaffold_path,
            endpoints=endpoints,
            agent_python=self.agent_python,
            isolation_dir=isolation_dir,
            expected_action_spec=self.profile.policy.action_spec,
            max_horizon=self.profile.policy.chunk_horizon,
            max_execution_count=self.profile.policy.execution_count,
            stderr_path=stderr_path,
            start_timeout_s=AGENT_START_TIMEOUT_S,
            call_timeout_s=AGENT_STEP_TIMEOUT_S,
        )

    def __call__(self, key: EpisodeKey) -> EpisodeExecution:
        session_id = key.artifact_id()
        runtime_dir = self.runtime_root / session_id
        runtime_dir.mkdir(parents=True, exist_ok=False)
        steps: list[PublicStepEvidence] = []
        success = False
        full_horizon = self.success_protocol == FULL_HORIZON_FINAL_SUCCESS
        private_metrics: dict[str, bool | float] | None = None
        with self.scheduler.session(
            session_id,
            preferred_replica_id=self.replica_assignments[session_id],
        ) as replica:
            render_gpu_id = self.render_gpu_assignments[session_id]
            endpoints = {
                "vla": ToolEndpoint(
                    replica.endpoint,
                    replica.identity,
                    True,
                    timeout_s=POLICY_CALL_TIMEOUT_S,
                ),
                **self.tool_endpoints,
            }
            if self.simulator_pool is not None:
                sim_pooled = True
                simulator = self.simulator_pool.acquire(
                    render_gpu_id,
                    key,
                    lambda: SimulatorProcess(
                        self.simulator_python,
                        self.profile,
                        key,
                        physical_gpu_id=render_gpu_id,
                        runtime_dir=self.simulator_pool.runtime_dir(render_gpu_id),
                        source_root=self.simulator_source,
                        start_timeout_s=self.simulator_start_timeout_s,
                        call_timeout_s=self.simulator_call_timeout_s,
                    ),
                )
            else:
                sim_pooled = False
                simulator = SimulatorProcess(
                    self.simulator_python,
                    self.profile,
                    key,
                    physical_gpu_id=render_gpu_id,
                    runtime_dir=runtime_dir / "simulator",
                    source_root=self.simulator_source,
                    start_timeout_s=self.simulator_start_timeout_s,
                    call_timeout_s=self.simulator_call_timeout_s,
                )
            if self.gateway_pool is not None:
                agent_pooled = True
                isolation_dir = self.gateway_pool.isolation_dir(replica.identity.replica_id)
                gateway = self.gateway_pool.acquire(
                    replica.identity.replica_id,
                    lambda: AgentProcessGateway(
                        self._gateway_config(endpoints, isolation_dir, isolation_dir / "stderr.log")
                    ),
                )
            else:
                agent_pooled = False
                gateway = AgentProcessGateway(
                    self._gateway_config(endpoints, runtime_dir / "agent", runtime_dir / "agent.stderr.log")
                )
            episode_ok = False
            try:
                if not sim_pooled:
                    simulator.start()
                if not agent_pooled:
                    gateway.start()
                simulator.reset()
                gateway.reset(session_id, key.policy_seed, key.task_id)
                for _ in range(key.horizon):
                    observation = self.profile.validate_observation(simulator.observe())
                    if observation.episode_id != session_id:
                        raise StrictSchemaError("profile runner: observation episode identity differs")
                    if not full_horizon and simulator.private_success():
                        success = True
                        steps.append(PublicStepEvidence(observation, None, ()))
                        break
                    result = gateway.act_with_events(
                        observation,
                        session_id,
                        self._request_id(key, observation.step_index),
                    )
                    action = self.profile.validate_agent_action_chunk(result.action)
                    steps.append(PublicStepEvidence(observation, action, result.events))
                    simulator.apply(action)
                    if not full_horizon and simulator.private_success():
                        success = True
                        break
                if full_horizon:
                    success = simulator.private_success()
                private_metrics = simulator.private_metrics()
                episode_ok = True
            finally:
                if not agent_pooled:
                    gateway.close(force=not episode_ok)
                elif episode_ok:
                    try:
                        gateway.end_session(session_id)
                    except Exception:
                        self.gateway_pool.evict(replica.identity.replica_id, gateway)
                else:
                    self.gateway_pool.evict(replica.identity.replica_id, gateway)
                if not sim_pooled:
                    simulator.close(force=not episode_ok)
                elif not episode_ok:
                    self.simulator_pool.evict(render_gpu_id, simulator)
        if not steps:
            raise RuntimeError("profile runner produced no public steps")
        if private_metrics is not None and "success" in private_metrics and private_metrics["success"] is not success:
            raise RuntimeError("profile runner private success and private metrics differ")
        termination = "horizon" if full_horizon else "success" if success else "horizon"
        pictures, picture_index, picture_error = _camera_artifacts(steps)
        artifacts = {
            "trace.jsonl": _readable_trace_bytes(
                steps, key, success, termination, picture_index, picture_error
            ),
        }
        artifacts.update(pictures)
        if private_metrics is not None:
            artifacts["private_metrics.json"] = (
                json.dumps(
                    {"schema_version": 1, "kind": "private_evaluator_metrics", "metrics": private_metrics},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        execution = EpisodeExecution(
            state="complete",
            success=success,
            steps=sum(item.action is not None for item in steps),
            artifacts=artifacts,
        )
        shutil.rmtree(runtime_dir, ignore_errors=True)
        return execution
