from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from robot_auto_evolve.agent import AgentProcessGateway, GatewayConfig, ToolEndpoint
from robot_auto_evolve.benchmarks.openvla_simpler_worker import is_openvla_simpler_adapter
from robot_auto_evolve.benchmarks.render_integrity import rgb_integrity_evidence, validate_mujoco_rgb
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation import EpisodeExecution, EpisodeOutcome, evaluate_supplied_runner
from robot_auto_evolve.evaluation.simulator import SimulatorProcess
from robot_auto_evolve.protocol import StrictSchemaError, encode_message
from robot_auto_evolve.provenance import ArtifactRun, EpisodeKey, EpisodePlan
from robot_auto_evolve.runtime_paths import assert_run_runtime_path, project_root_from_package
from robot_auto_evolve.services import MsgpackServiceClient, ReplicaScheduler, ServiceReplica

from .evidence import PublicEvolutionEvidence, PublicStepEvidence
from .hashing import mapping_sha256, tree_hashes
from .models import EvaluationResult


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


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parallel_simulator_phase(
    simulators: tuple[SimulatorProcess, ...],
    phase: str,
    operation: Callable[[int, SimulatorProcess], object],
) -> tuple[object, ...]:
    results: list[object | None] = [None] * len(simulators)
    failures = []
    with ThreadPoolExecutor(max_workers=len(simulators), thread_name_prefix=f"render-preflight-{phase}") as executor:
        futures = {
            executor.submit(operation, index, simulator): index
            for index, simulator in enumerate(simulators)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                failures.append((index, exc))
    if failures:
        index, error = min(failures, key=lambda item: item[0])
        raise RuntimeError(
            f"render preflight {phase} failed for worker {index}: {type(error).__name__}: {error}"
        ) from error
    if any(result is None for result in results):
        raise RuntimeError(f"render preflight {phase} returned an incomplete result set")
    return tuple(result for result in results if result is not None)


@dataclass(frozen=True)
class _RenderPreflightContext:
    worker_index: int
    context_id: str
    key: EpisodeKey
    policy_replica_id: str


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


class ProfileEvaluator:
    def __init__(
        self,
        profile: Profile,
        plan: EpisodePlan,
        *,
        agent_python: Path,
        simulator_python: Path,
        simulator_source: Path | None = None,
        live_clients: Mapping[tuple[str, str], MsgpackServiceClient] | None = None,
        render_gpu_ids: tuple[int, ...] | list[int] | None = None,
        runtime_root: Path,
    ) -> None:
        if not isinstance(profile, Profile) or not isinstance(plan, EpisodePlan):
            raise StrictSchemaError("profile evaluator requires Profile and EpisodePlan")
        profile.validate(plan)
        if profile.policy.deployment_mode != "replicated":
            raise StrictSchemaError("profile evaluator currently requires a replicated policy")
        clients = {} if live_clients is None else dict(live_clients)
        replicas = []
        identities = []
        for item in profile.policy.replicas:
            key = (item.identity.service_name, item.identity.replica_id)
            client = clients.get(key) or MsgpackServiceClient(item.endpoint, item.identity, timeout=120.0)
            actual = client.validate_identity()
            identities.append(actual)
            replicas.append(ServiceReplica(item.endpoint, actual, client))
        tool_endpoints = {}
        for tool in profile.tools:
            if not tool.enabled or tool.service is None:
                continue
            item = tool.service
            key = (item.identity.service_name, item.identity.replica_id)
            client = clients.get(key) or MsgpackServiceClient(item.endpoint, item.identity, timeout=120.0)
            identities.append(client.validate_identity())
            tool_endpoints[tool.capability] = ToolEndpoint(
                item.endpoint,
                item.identity,
                tool.required,
                timeout_s=TOOL_CALL_TIMEOUT_S,
            )
        profile.validate_service_identities(identities)
        replica_count = len(replicas)
        sessions_per_replica = (profile.resources.workers + replica_count - 1) // replica_count
        self.profile = profile
        self.plan = plan
        self.agent_python = Path(agent_python).resolve()
        self.simulator_python = Path(simulator_python).resolve()
        self.simulator_source = None if simulator_source is None else Path(simulator_source).resolve()
        self.runtime_root = assert_run_runtime_path(project_root_from_package(), runtime_root)
        self.scheduler = ReplicaScheduler(replicas, max_sessions_per_replica=sessions_per_replica)
        self.tool_endpoints = tool_endpoints
        self.identities = tuple(sorted(identities, key=lambda item: (item.service_name, item.replica_id)))
        self.sessions_per_replica = sessions_per_replica
        self.render_gpu_ids = resolve_render_gpu_ids(profile, render_gpu_ids)
        self.render_gpu_by_replica = {
            replica.identity.replica_id: gpu_id
            for replica, gpu_id in zip(self.scheduler.replicas, self.render_gpu_ids, strict=True)
        }

    def _policy_replica_id(self, index: int) -> str:
        return self.scheduler.replicas[index % len(self.scheduler.replicas)].identity.replica_id

    def preflight_rendering(self, runtime_root: Path) -> dict[str, object]:
        root = Path(runtime_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        keys = self.plan.for_split("evolve")
        if not keys:
            raise RuntimeError("render preflight requires at least one evolution episode")
        contexts = tuple(
            _RenderPreflightContext(
                worker_index=index,
                context_id=f"worker-{index:04d}",
                key=keys[index % len(keys)],
                policy_replica_id=self._policy_replica_id(index),
            )
            for index in range(self.profile.resources.workers)
        )
        simulators = tuple(
            SimulatorProcess(
                self.simulator_python,
                self.profile,
                context.key,
                physical_gpu_id=self.render_gpu_by_replica[context.policy_replica_id],
                runtime_dir=root / context.context_id,
                source_root=self.simulator_source,
                start_timeout_s=simulator_timeouts(self.profile.environment.suite)[0],
                call_timeout_s=simulator_timeouts(self.profile.environment.suite)[1],
            )
            for context in contexts
        )
        failure: BaseException | None = None
        try:
            def start_simulator(_index: int, simulator: SimulatorProcess) -> object:
                simulator.start()
                return True

            _parallel_simulator_phase(
                simulators,
                "startup",
                start_simulator,
            )
            alive = tuple(
                simulator
                for simulator in simulators
                if simulator.process is not None and simulator.process.poll() is None
            )
            if len(alive) != self.profile.resources.workers:
                raise RuntimeError(
                    f"render preflight started {len(alive)}/{self.profile.resources.workers} simulator processes"
                )
            barrier = threading.Barrier(self.profile.resources.workers)

            def reset_observe(index: int, simulator: SimulatorProcess) -> object:
                context = contexts[index]
                barrier.wait(timeout=simulator_timeouts(self.profile.environment.suite)[1])
                simulator.reset()
                observation = self.profile.validate_observation(simulator.observe())
                if observation.episode_id != context.key.artifact_id() or observation.step_index != 0:
                    raise StrictSchemaError("render preflight observation identity or reset step differs")
                cameras = {}
                for name, camera in sorted(observation.cameras.items()):
                    rgb = validate_mujoco_rgb(camera.rgb, name)
                    evidence = rgb_integrity_evidence(rgb, name)
                    evidence["rgb_sha256"] = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
                    cameras[name] = evidence
                return {
                    "worker_index": index,
                    "context_id": context.context_id,
                    "policy_replica_id": context.policy_replica_id,
                    "render_gpu_id": self.render_gpu_by_replica[context.policy_replica_id],
                    "episode_id": context.key.artifact_id(),
                    "episode_key": context.key.to_mapping(),
                    "observation_step_index": observation.step_index,
                    "cameras": cameras,
                }

            records = _parallel_simulator_phase(simulators, "reset-observe", reset_observe)
            return {
                "schema_version": 2,
                "state": "complete",
                "suite": self.profile.environment.suite,
                "operation": "all-started-then-concurrent-reset-observe",
                "worker_count": self.profile.resources.workers,
                "policy_replica_count": len(self.scheduler.replicas),
                "max_sessions_per_policy_replica": self.sessions_per_replica,
                "render_gpu_ids": list(self.render_gpu_ids),
                "records": list(records),
            }
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                def close_simulator(_index: int, simulator: SimulatorProcess) -> object:
                    simulator.close(force=failure is not None)
                    return True

                _parallel_simulator_phase(
                    simulators,
                    "cleanup",
                    close_simulator,
                )
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"render preflight cleanup failed: {cleanup_error}")

    def evaluate(self, scaffold_dir: Path, split: str, output_dir: Path) -> EvaluationResult:
        scaffold = Path(scaffold_dir).resolve()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        code_hash = mapping_sha256(tree_hashes(scaffold))
        run = ArtifactRun.open(
            output / "artifacts",
            run_id=f"{self.profile.profile_id}-{split}-{output.name}-{code_hash[:12]}",
            profile_hash=self.profile.resolved_hash(),
            plan=self.plan,
            code_hash=code_hash,
            service_identities=self.identities,
            split=split,
        )
        replica_assignments = {
            key.artifact_id(): self._policy_replica_id(index)
            for index, key in enumerate(self.plan.for_split(split))
        }
        runner = ProfileEpisodeRunner(
            self.profile,
            scaffold,
            self.scheduler,
            self.tool_endpoints,
            agent_python=self.agent_python,
            simulator_python=self.simulator_python,
            simulator_source=self.simulator_source,
            runtime_root=self.runtime_root / "evaluations" / hashlib.sha256(str(output).encode()).hexdigest()[:24],
            replica_assignments=replica_assignments,
            render_gpu_assignments={
                session_id: self.render_gpu_by_replica[replica_id]
                for session_id, replica_id in replica_assignments.items()
            },
        )
        summary = evaluate_supplied_runner(
            self.plan,
            run,
            runner,
            split=split,
            workers=self.profile.resources.workers,
        )
        final = run.finalize()
        if not summary.complete or summary.metrics is None or final["state"] != "complete":
            raise RuntimeError(
                f"{split} evaluation incomplete: complete={summary.n_complete}/{summary.n_expected}, "
                f"partial={summary.n_partial}, error={summary.n_error}"
            )
        manifests = run.episode_manifests()
        outcomes = tuple(EpisodeOutcome.from_manifest(item) for item in manifests)
        evidence_hash = None
        evidence_count = 0
        if split == "evolve":
            records = tuple(
                (
                    EpisodeOutcome.from_manifest(manifest),
                    run.path / "episodes" / manifest.key.artifact_id() / "trace.msgpack",
                )
                for manifest in manifests
            )
            evidence = PublicEvolutionEvidence.create_from_trace_files(output / "public_evidence", records)
            evidence_hash = evidence.bundle_sha256
            evidence_count = len(evidence.episodes)
        result = EvaluationResult(
            split=split,
            outcomes=outcomes,
            metadata={
                "profile_sha256": self.profile.resolved_hash(),
                "episode_plan_sha256": self.plan.resolved_hash(),
                "artifact_manifest_sha256": final["manifest_sha256"],
                "requested_episode_workers": self.profile.resources.workers,
                "max_sessions_per_policy_replica": self.sessions_per_replica,
                "max_concurrent_inferences_per_policy_replica": 1,
                "policy_call_timeout_s": POLICY_CALL_TIMEOUT_S,
                "tool_call_timeout_s": TOOL_CALL_TIMEOUT_S,
                "agent_step_timeout_s": AGENT_STEP_TIMEOUT_S,
                "simulator_start_timeout_s": runner.simulator_start_timeout_s,
                "simulator_call_timeout_s": runner.simulator_call_timeout_s,
                "success_protocol": runner.success_protocol,
                "render_gpu_ids": list(self.render_gpu_ids),
                "metrics": summary.metrics.to_mapping(),
            },
            public_evidence_sha256=evidence_hash,
            public_evidence_episodes=evidence_count,
        )
        _write_json(output / "report.json", result.to_mapping())
        return result
