from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent import AgentProcessGateway, GatewayConfig, ToolEndpoint
from robot_auto_evolve.benchmarks.robolab120_batching import RoboLabBatch, build_robolab_batch_schedule
from robot_auto_evolve.benchmarks.robolab120_rpc import RoboLabActionBatch
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation import EpisodeOutcome, summarize_split
from robot_auto_evolve.evaluation.robolab120_simulator import RoboLabSimulatorProcess
from robot_auto_evolve.protocol import CanonicalActionChunk, StrictSchemaError, encode_message
from robot_auto_evolve.provenance import ArtifactRun, EpisodeKey, EpisodePlan
from robot_auto_evolve.services import MsgpackServiceClient, ServiceReplica

from .evidence import PublicEvolutionEvidence, PublicStepEvidence
from .hashing import mapping_sha256, tree_hashes
from .models import EvaluationResult
from .profile_evaluator import AGENT_STEP_TIMEOUT_S, POLICY_CALL_TIMEOUT_S, TOOL_CALL_TIMEOUT_S


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class RoboLab120ProfileEvaluator:
    def __init__(
        self,
        profile: Profile,
        plan: EpisodePlan,
        *,
        agent_python: Path,
        simulator_python: Path,
        simulator_source: Path,
        runtime_root: Path,
        live_clients: Mapping[tuple[str, str], MsgpackServiceClient] | None = None,
    ) -> None:
        if not isinstance(profile, Profile) or not isinstance(plan, EpisodePlan):
            raise StrictSchemaError("RoboLab evaluator requires Profile and EpisodePlan")
        profile.validate(plan)
        if profile.environment.suite != "robolab120_droid_jointpos":
            raise StrictSchemaError("RoboLab evaluator suite differs")
        if profile.policy.deployment_mode != "replicated":
            raise StrictSchemaError("RoboLab evaluator requires replicated policy services")
        clients = {} if live_clients is None else dict(live_clients)
        replicas = []
        identities = []
        for item in profile.policy.replicas:
            key = (item.identity.service_name, item.identity.replica_id)
            client = clients.get(key) or MsgpackServiceClient(item.endpoint, item.identity, timeout=120.0)
            actual = client.validate_identity()
            if len(actual.gpu_ids) != 1:
                raise StrictSchemaError("RoboLab policy replica must use one GPU")
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
        if profile.resources.workers % len(replicas):
            raise StrictSchemaError("RoboLab workers must divide evenly across policy replicas")
        vector_batch_size = profile.resources.workers // len(replicas)
        if vector_batch_size < 1:
            raise StrictSchemaError("RoboLab vector batch size is invalid")
        self.profile = profile
        self.plan = plan
        self.agent_python = Path(agent_python).resolve()
        self.simulator_python = Path(simulator_python).resolve()
        self.simulator_source = Path(simulator_source).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.replicas = tuple(replicas)
        self.tool_endpoints = tool_endpoints
        self.identities = tuple(sorted(identities, key=lambda item: (item.service_name, item.replica_id)))
        self.vector_batch_size = vector_batch_size
        self.simulators: tuple[RoboLabSimulatorProcess, ...] = ()

    def start(self) -> None:
        if self.simulators:
            raise RuntimeError("RoboLab evaluator is already started")
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        simulators = tuple(
            RoboLabSimulatorProcess(
                self.simulator_python,
                self.profile,
                physical_gpu_id=replica.identity.gpu_ids[0],
                vector_batch_size=self.vector_batch_size,
                runtime_dir=self.runtime_root / f"gpu{replica.identity.gpu_ids[0]}",
                source_root=self.simulator_source,
            )
            for replica in self.replicas
        )
        try:
            with ThreadPoolExecutor(max_workers=len(simulators)) as executor:
                futures = [executor.submit(item.start) for item in simulators]
                for future in futures:
                    future.result()
        except BaseException:
            for item in simulators:
                item.close(force=True)
            raise
        self.simulators = simulators

    def close(self, force: bool = False) -> None:
        simulators = self.simulators
        self.simulators = ()
        for simulator in simulators:
            simulator.close(force=force)

    @staticmethod
    def _request_id(key: EpisodeKey, step_index: int) -> str:
        return hashlib.sha256(f"{key.artifact_id()}\0{step_index}".encode()).hexdigest()

    def _gateway(
        self,
        scaffold: Path,
        replica: ServiceReplica,
        runtime: Path,
    ) -> AgentProcessGateway:
        endpoints = {
            "vla": ToolEndpoint(
                replica.endpoint,
                replica.identity,
                True,
                timeout_s=POLICY_CALL_TIMEOUT_S,
            ),
            **self.tool_endpoints,
        }
        return AgentProcessGateway(
            GatewayConfig(
                scaffold_path=scaffold / "scaffold.py",
                endpoints=endpoints,
                agent_python=self.agent_python,
                isolation_dir=runtime / "agent",
                expected_action_spec=self.profile.policy.action_spec,
                max_horizon=self.profile.policy.chunk_horizon,
                max_execution_count=self.profile.policy.execution_count,
                stderr_path=runtime / "agent.stderr.log",
                call_timeout_s=AGENT_STEP_TIMEOUT_S,
            )
        )

    def _noop(self, key: EpisodeKey, status: Any, execution_count: int) -> CanonicalActionChunk:
        start = min(status.step_index, key.horizon - execution_count)
        return CanonicalActionChunk(
            request_id=f"frozen-{key.artifact_id()}-{status.step_index}",
            session_id=key.artifact_id(),
            start_step=max(0, start),
            spec=self.profile.policy.action_spec,
            values=np.zeros((execution_count, 8), dtype=np.float32),
            execution_count=execution_count,
        )

    def _run_batch(
        self,
        scaffold: Path,
        output: Path,
        run: ArtifactRun,
        batch: RoboLabBatch,
        replica: ServiceReplica,
        simulator: RoboLabSimulatorProcess,
    ) -> None:
        runtime = output / "runtime" / batch.batch_id
        runtime.mkdir(parents=True, exist_ok=False)
        gateway = self._gateway(scaffold, replica, runtime)
        evidence: dict[str, list[PublicStepEvidence]] = {key.artifact_id(): [] for key in batch.episodes}
        started = {key.artifact_id(): time.time_ns() for key in batch.episodes}
        simulator.load_batch(batch)
        try:
            with gateway:
                for key in batch.episodes:
                    gateway.reset(key.artifact_id(), key.policy_seed, key.task_id)
                while True:
                    statuses = simulator.private_status_batch(batch).statuses
                    if all(item.frozen for item in statuses):
                        break
                    observations = simulator.observe_batch(batch).observations
                    actions: list[CanonicalActionChunk | None] = [None] * len(batch.episodes)
                    events: dict[int, tuple[Any, ...]] = {}
                    active_counts = set()
                    for index, (key, status, observation) in enumerate(
                        zip(batch.episodes, statuses, observations, strict=True)
                    ):
                        if status.frozen:
                            continue
                        result = gateway.act_with_events(
                            observation,
                            key.artifact_id(),
                            self._request_id(key, observation.step_index),
                        )
                        remaining = key.horizon - observation.step_index
                        action = self.profile.validate_agent_action_chunk(result.action)
                        if action.execution_count > remaining:
                            action = replace(action, execution_count=remaining)
                        actions[index] = action
                        events[index] = result.events
                        active_counts.add(action.execution_count)
                    if len(active_counts) != 1:
                        raise RuntimeError("RoboLab active sessions returned different execution counts")
                    execution_count = active_counts.pop()
                    for index, (key, status, observation) in enumerate(
                        zip(batch.episodes, statuses, observations, strict=True)
                    ):
                        if status.frozen:
                            actions[index] = self._noop(key, status, execution_count)
                        else:
                            evidence[key.artifact_id()].append(
                                PublicStepEvidence(observation, actions[index], events[index])
                            )
                    action_batch = RoboLabActionBatch(batch.batch_id, tuple(actions))
                    simulator.apply_batch(batch, action_batch)
            statuses = simulator.private_status_batch(batch).statuses
            if not all(item.frozen for item in statuses):
                raise RuntimeError("RoboLab batch stopped before every slot completed")
            simulator.finish_batch(batch)
        except BaseException:
            raise
        for key, status in zip(batch.episodes, statuses, strict=True):
            rows = evidence[key.artifact_id()]
            outcome = EpisodeOutcome(key, status.success)
            trace = {
                "outcome": outcome.to_mapping(),
                "termination": "success" if status.success else "horizon",
                "steps": [item.to_mapping() for item in rows],
                "error": None,
            }
            execution = {
                "schema_version": 1,
                "policy_service_name": replica.identity.service_name,
                "policy_replica_id": replica.identity.replica_id,
                "gpu_ids": list(replica.identity.gpu_ids),
                "robolab_batch_id": batch.batch_id,
                "vector_batch_size": len(batch.episodes),
                "vector_batch_capacity": self.vector_batch_size,
            }
            run.record_episode(
                key,
                state="complete",
                success=status.success,
                steps=len(rows),
                artifacts={
                    "execution.json": (
                        json.dumps(execution, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode(),
                    "trace.msgpack": encode_message(trace),
                },
                started_ns=started[key.artifact_id()],
                finished_ns=time.time_ns(),
            )

    def evaluate(self, scaffold_dir: Path, split: str, output_dir: Path) -> EvaluationResult:
        if not self.simulators:
            self.start()
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
        schedule = build_robolab_batch_schedule(
            self.plan,
            split=split,
            vector_batch_size=self.vector_batch_size,
        )
        lanes = [[] for _ in self.simulators]
        for index, batch in enumerate(schedule.batches):
            lanes[index % len(lanes)].append(batch)

        def execute_lane(index: int) -> None:
            for batch in lanes[index]:
                try:
                    self._run_batch(
                        scaffold,
                        output,
                        run,
                        batch,
                        self.replicas[index],
                        self.simulators[index],
                    )
                except Exception as exc:
                    for key in batch.episodes:
                        if key not in {item.key for item in run.episode_manifests()}:
                            run.record_episode(
                                key,
                                state="error",
                                success=None,
                                steps=0,
                                artifacts={"error.txt": f"{type(exc).__name__}: {exc}".encode()},
                                error=f"batch_exception:{type(exc).__name__}",
                            )
                    raise

        failures = []
        with ThreadPoolExecutor(max_workers=len(self.simulators)) as executor:
            futures = [executor.submit(execute_lane, index) for index in range(len(lanes))]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    failures.append(exc)
        if not failures:
            barrier = f"{split}-{output.name}-{code_hash[:12]}"
            for simulator in self.simulators:
                simulator.candidate_barrier(barrier)
        summary = summarize_split(self.plan, run.episode_manifests(), split)
        final = run.finalize()
        if failures or not summary.complete or summary.metrics is None or final["state"] != "complete":
            detail = type(failures[0]).__name__ if failures else "incomplete"
            raise RuntimeError(
                f"{split} RoboLab evaluation failed ({detail}): complete={summary.n_complete}/"
                f"{summary.n_expected}, partial={summary.n_partial}, error={summary.n_error}"
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
            public = PublicEvolutionEvidence.create_from_trace_files(output / "public_evidence", records)
            evidence_hash = public.bundle_sha256
            evidence_count = len(public.episodes)
        result = EvaluationResult(
            split=split,
            outcomes=outcomes,
            metadata={
                "profile_sha256": self.profile.resolved_hash(),
                "episode_plan_sha256": self.plan.resolved_hash(),
                "artifact_manifest_sha256": final["manifest_sha256"],
                "simulator_app_count": len(self.simulators),
                "vector_batch_size_per_app": self.vector_batch_size,
                "policy_call_timeout_s": POLICY_CALL_TIMEOUT_S,
                "tool_call_timeout_s": TOOL_CALL_TIMEOUT_S,
                "agent_step_timeout_s": AGENT_STEP_TIMEOUT_S,
                "metrics": summary.metrics.to_mapping(),
            },
            public_evidence_sha256=evidence_hash,
            public_evidence_episodes=evidence_count,
        )
        _write_json(output / "report.json", result.to_mapping())
        return result

    def __enter__(self) -> "RoboLab120ProfileEvaluator":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(force=exc is not None)
