from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from robot_auto_evolve.agent import AgentProcessGateway, GatewayConfig, ToolEndpoint
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation import EpisodeExecution, EpisodeOutcome
from robot_auto_evolve.evaluation.simulator import SimulatorProcess
from robot_auto_evolve.protocol import StrictSchemaError, encode_message
from robot_auto_evolve.provenance import EpisodeKey
from robot_auto_evolve.services import ReplicaScheduler

from .evidence import PublicStepEvidence


POLICY_CALL_TIMEOUT_S = 120.0
TOOL_CALL_TIMEOUT_S = 300.0
AGENT_STEP_TIMEOUT_S = 600.0
SIMULATOR_START_TIMEOUT_S = 60.0
SIMULATOR_CALL_TIMEOUT_S = 120.0
ROBOTWIN2_SIMULATOR_START_TIMEOUT_S = 1800.0
ROBOTWIN2_SIMULATOR_CALL_TIMEOUT_S = 900.0
STOP_ON_FIRST_SUCCESS = "stop_on_first_success"
FULL_HORIZON_FINAL_SUCCESS = "full_horizon_final_success"


def success_protocol(profile: Profile) -> str:
    # SimplerEnv upstream (maniskill2_evaluator.py) runs the full horizon and reads
    # task success at the FINAL step (non-sticky) for BOTH robots/forks. This applies
    # to every simpler_* route (X-VLA google/widowx AND OpenVLA google), not just
    # OpenVLA. The former X-VLA sticky/stop-on-first path inflated the success rate.
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
        if set(self.render_gpu_assignments) != set(self.replica_assignments):
            raise StrictSchemaError("profile runner render and policy session assignments differ")
        self.success_protocol = success_protocol(profile)
        self.simulator_start_timeout_s, self.simulator_call_timeout_s = simulator_timeouts(
            profile.environment.suite
        )

    @staticmethod
    def _request_id(key: EpisodeKey, step_index: int) -> str:
        return hashlib.sha256(f"{key.artifact_id()}\0{step_index}".encode()).hexdigest()

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
            gateway = AgentProcessGateway(
                GatewayConfig(
                    scaffold_path=self.scaffold_path,
                    endpoints=endpoints,
                    agent_python=self.agent_python,
                    isolation_dir=runtime_dir / "agent",
                    expected_action_spec=self.profile.policy.action_spec,
                    max_horizon=self.profile.policy.chunk_horizon,
                    max_execution_count=self.profile.policy.execution_count,
                    stderr_path=runtime_dir / "agent.stderr.log",
                    call_timeout_s=AGENT_STEP_TIMEOUT_S,
                )
            )
            with simulator, gateway:
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
        if not steps:
            raise RuntimeError("profile runner produced no public steps")
        if private_metrics is not None and "success" in private_metrics and private_metrics["success"] is not success:
            raise RuntimeError("profile runner private success and private metrics differ")
        outcome = EpisodeOutcome(key, success)
        termination = "horizon" if full_horizon else "success" if success else "horizon"
        trace = {
            "outcome": outcome.to_mapping(),
            "termination": termination,
            "steps": [item.to_mapping() for item in steps],
            "error": None,
        }
        execution = {
            "schema_version": 2,
            "policy_service_name": replica.identity.service_name,
            "policy_replica_id": replica.identity.replica_id,
            "gpu_ids": list(replica.identity.gpu_ids),
            "render_gpu_id": render_gpu_id,
        }
        artifacts = {
            "execution.json": (json.dumps(execution, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            "trace.msgpack": encode_message(trace),
        }
        if private_metrics is not None:
            artifacts["private_metrics.json"] = (
                json.dumps(
                    {"schema_version": 1, "kind": "private_evaluator_metrics", "metrics": private_metrics},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        for source, name in (
            (runtime_dir / "agent.stderr.log", "agent.stderr.log"),
            (runtime_dir / "simulator" / "simulator.stderr.log", "simulator.stderr.log"),
        ):
            if source.is_file():
                artifacts[name] = source.read_bytes()
        return EpisodeExecution(
            state="complete",
            success=success,
            steps=sum(item.action is not None for item in steps),
            artifacts=artifacts,
        )
