from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent import AgentProcessGateway, GatewayConfig, ToolEndpoint
from robot_auto_evolve.benchmarks.robolab120_batching import (
    RoboLabBatch,
    build_robolab_benchmark_batches,
)
from robot_auto_evolve.benchmarks.robolab120_protocol import (
    ROBOLAB120_BENCHMARK_ID,
    ROBOLAB120_BENCHMARK_PROTOCOL,
    ROBOLAB120_ENVIRONMENT_SEED,
    ROBOLAB120_INSTRUCTION_TYPE,
    ROBOLAB120_MODEL_ROUTE,
    ROBOLAB120_POLICY_SEED_BASE,
    ROBOLAB120_SUITE,
    ROBOLAB120_TRIALS_PER_TASK,
    load_robolab120_catalog,
)
from robot_auto_evolve.benchmarks.robolab120_rpc import RoboLabActionBatch
from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation import EpisodeExecution
from robot_auto_evolve.evaluation.benchmark import (
    CanonicalBenchmarkEvaluator,
    _atomic_write,
    _load_json,
    _preserve_scaffold,
    _verify_episode_directory,
    verify_benchmark_output,
)
from robot_auto_evolve.evaluation.metrics import compute_task_macro_metrics
from robot_auto_evolve.evaluation.robolab120_simulator import RoboLabSimulatorProcess
from robot_auto_evolve.evolution.hashing import mapping_sha256, tree_hashes
from robot_auto_evolve.evolution.profile_evaluator import (
    AGENT_STEP_TIMEOUT_S,
    POLICY_CALL_TIMEOUT_S,
    TOOL_CALL_TIMEOUT_S,
)
from robot_auto_evolve.protocol import CanonicalActionChunk, StrictSchemaError, encode_message
from robot_auto_evolve.protocol.schema import boolean, fields, integer, sequence, sha256, string
from robot_auto_evolve.provenance import BenchmarkPlan, EpisodeKey, EpisodeManifest, canonical_json_bytes
from robot_auto_evolve.services import MsgpackServiceClient, ServiceReplica


METRIC = "equal_task_macro_success"

ROBOLAB120_BENCHMARK_ROUTES = {
    "molmobot_robolab120": {
        "policy_service_name": "molmobot_droid",
        "profile_path": "configs/molmobot_robolab120.json",
        "benchmark_id": ROBOLAB120_BENCHMARK_ID,
    },
    "openpi_pi05_droid_jointpos": {
        "policy_service_name": "openpi_pi05_droid_jointpos",
        "profile_path": "configs/openpi_pi05_droid_jointpos.json",
        "benchmark_id": "openpi_pi05_droid_jointpos_robolab120_vague_project_fixed_3_per_task_v1",
    },
    "openpi_pi0_fast_droid_jointpos": {
        "policy_service_name": "openpi_pi0_fast_droid_jointpos",
        "profile_path": "configs/openpi_pi0_fast_droid_jointpos.json",
        "benchmark_id": "openpi_pi0_fast_droid_jointpos_robolab120_vague_project_fixed_3_per_task_v1",
    },
}


def _relative_json(value: Any, name: str) -> str:
    path = PurePosixPath(string(value, name))
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
        raise StrictSchemaError(f"{name}: expected safe relative JSON path")
    return path.as_posix()


def robolab120_metrics(manifests: tuple[EpisodeManifest, ...]) -> dict[str, Any]:
    tasks = sorted({item.key.task_id for item in manifests})
    task_metrics = {
        task: compute_task_macro_metrics(tuple(item for item in manifests if item.key.task_id == task)).to_mapping()
        for task in tasks
    }
    overall = compute_task_macro_metrics(manifests).to_mapping()
    return {
        "metric": METRIC,
        "score": overall["macro_success"],
        "task_metrics": task_metrics,
        "all_task_macro": overall,
    }


def robolab120_runtime(
    manifests: tuple[EpisodeManifest, ...],
    episode_root: Path,
    *,
    expected_policy_service_name: str = "molmobot_droid",
) -> dict[str, Any]:
    expected_policy_service_name = string(
        expected_policy_service_name,
        "robolab120_benchmark.expected_policy_service_name",
    )
    records: list[dict[str, Any]] = []
    expected_fields = {
        "schema_version",
        "policy_service_name",
        "policy_replica_id",
        "gpu_ids",
        "robolab_batch_id",
        "vector_batch_size",
        "vector_batch_capacity",
        "simulator_app_id",
        "configured_workers_per_policy_replica",
        "simulator_apps_per_policy_replica",
    }
    for manifest in manifests:
        value = fields(
            _load_json(episode_root / manifest.key.artifact_id() / "execution.json"),
            expected_fields,
            path="robolab120_benchmark_execution",
        )
        if integer(value["schema_version"], "robolab120_benchmark_execution.schema_version") != 1:
            raise StrictSchemaError("RoboLab benchmark execution schema differs")
        if (
            string(value["policy_service_name"], "robolab120_benchmark_execution.policy_service_name")
            != expected_policy_service_name
        ):
            raise StrictSchemaError("RoboLab benchmark execution policy differs")
        replica_id = string(value["policy_replica_id"], "robolab120_benchmark_execution.policy_replica_id")
        gpu_ids = tuple(
            integer(item, "robolab120_benchmark_execution.gpu_ids", minimum=0)
            for item in sequence(value["gpu_ids"], "robolab120_benchmark_execution.gpu_ids")
        )
        if len(gpu_ids) != 1:
            raise StrictSchemaError("RoboLab benchmark execution GPU assignment differs")
        batch_size = integer(value["vector_batch_size"], "robolab120_benchmark_execution.vector_batch_size", minimum=1)
        capacity = integer(
            value["vector_batch_capacity"],
            "robolab120_benchmark_execution.vector_batch_capacity",
            minimum=1,
        )
        workers = integer(
            value["configured_workers_per_policy_replica"],
            "robolab120_benchmark_execution.configured_workers_per_policy_replica",
            minimum=1,
        )
        apps = integer(
            value["simulator_apps_per_policy_replica"],
            "robolab120_benchmark_execution.simulator_apps_per_policy_replica",
            minimum=1,
        )
        app_id = string(value["simulator_app_id"], "robolab120_benchmark_execution.simulator_app_id")
        if (
            batch_size > capacity
            or capacity * apps > workers
            or re.fullmatch(rf"gpu{gpu_ids[0]}-app(?:0|[1-9][0-9]*)", app_id) is None
            or replica_id != f"gpu{gpu_ids[0]}"
        ):
            raise StrictSchemaError("RoboLab benchmark execution concurrency differs")
        sha256(value["robolab_batch_id"], "robolab120_benchmark_execution.robolab_batch_id")
        records.append(
            {
                "batch_id": value["robolab_batch_id"],
                "batch_size": batch_size,
                "replica_id": replica_id,
                "gpu_id": gpu_ids[0],
                "app_id": app_id,
                "capacity": capacity,
                "workers": workers,
                "apps": apps,
            }
        )
    if not records:
        raise StrictSchemaError("RoboLab benchmark execution records are empty")
    batches: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        batches.setdefault(record["batch_id"], []).append(record)
    for batch_records in batches.values():
        first = batch_records[0]
        if (
            len(batch_records) != first["batch_size"]
            or any(record != first for record in batch_records)
        ):
            raise StrictSchemaError("RoboLab benchmark vector batch evidence differs")
    replicas = {(record["replica_id"], record["gpu_id"]) for record in records}
    capacities = {record["capacity"] for record in records}
    worker_counts = {record["workers"] for record in records}
    app_counts = {record["apps"] for record in records}
    if len(capacities) != 1 or len(worker_counts) != 1 or len(app_counts) != 1:
        raise StrictSchemaError("RoboLab benchmark execution concurrency is inconsistent")
    capacity = capacities.pop()
    workers = worker_counts.pop()
    apps_per_replica = app_counts.pop()
    app_ids = {record["app_id"] for record in records}
    for replica_id, gpu_id in replicas:
        expected = {f"gpu{gpu_id}-app{index}" for index in range(apps_per_replica)}
        actual = {
            record["app_id"]
            for record in records
            if record["replica_id"] == replica_id
        }
        if actual != expected:
            raise StrictSchemaError("RoboLab benchmark simulator application coverage differs")
    return {
        "policy_replica_count": len(replicas),
        "configured_workers_per_policy_replica": workers,
        "vector_batch_size_per_simulator_app": capacity,
        "actual_vector_batch_sizes": sorted({record["batch_size"] for record in records}),
        "simulator_apps_per_policy_replica": apps_per_replica,
        "simulator_app_count": len(app_ids),
        "planned_concurrent_episode_slots_per_policy_replica": capacity * apps_per_replica,
    }


@dataclass(frozen=True)
class RoboLab120BenchmarkConfig:
    benchmark_id: str
    model_route: str
    protocol: str
    trials_per_task: int
    instruction_type: str
    project_fixed_seeds: bool
    paper_seed_table_available: bool
    plan_path: str
    plan_sha256: str
    profile_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab120_benchmark.schema_version: expected 1")
        if self.protocol != ROBOLAB120_BENCHMARK_PROTOCOL:
            raise StrictSchemaError("RoboLab benchmark identity or protocol differs")
        model_route = string(self.model_route, "robolab120_benchmark.model_route")
        if model_route not in ROBOLAB120_BENCHMARK_ROUTES:
            raise StrictSchemaError("RoboLab benchmark model route differs")
        object.__setattr__(self, "model_route", model_route)
        if self.benchmark_id != self.route_binding["benchmark_id"]:
            raise StrictSchemaError("RoboLab benchmark identity does not match its model route")
        if integer(self.trials_per_task, "robolab120_benchmark.trials_per_task", minimum=1) != ROBOLAB120_TRIALS_PER_TASK:
            raise StrictSchemaError("RoboLab benchmark trial count differs")
        if self.instruction_type != ROBOLAB120_INSTRUCTION_TYPE:
            raise StrictSchemaError("RoboLab benchmark instruction type differs")
        if self.project_fixed_seeds is not True or self.paper_seed_table_available is not False:
            raise StrictSchemaError("RoboLab benchmark seed provenance differs")
        object.__setattr__(self, "plan_path", _relative_json(self.plan_path, "robolab120_benchmark.plan.path"))
        object.__setattr__(self, "plan_sha256", sha256(self.plan_sha256, "robolab120_benchmark.plan.sha256"))
        object.__setattr__(self, "profile_path", _relative_json(self.profile_path, "robolab120_benchmark.profile"))
        if self.profile_path != self.route_binding["profile_path"]:
            raise StrictSchemaError("RoboLab benchmark profile does not match its model route")

    @property
    def route_binding(self) -> dict[str, str]:
        return ROBOLAB120_BENCHMARK_ROUTES[self.model_route]

    @property
    def policy_service_name(self) -> str:
        return self.route_binding["policy_service_name"]

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLab120BenchmarkConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "model_route",
                "metric",
                "protocol",
                "trials_per_task",
                "instruction_type",
                "project_fixed_seeds",
                "paper_seed_table_available",
                "plan",
                "profile",
            },
            path="robolab120_benchmark",
        )
        if obj["metric"] != METRIC:
            raise StrictSchemaError("RoboLab benchmark metric differs")
        plan = fields(obj["plan"], {"path", "sha256"}, path="robolab120_benchmark.plan")
        return cls(
            schema_version=integer(obj["schema_version"], "robolab120_benchmark.schema_version"),
            benchmark_id=obj["benchmark_id"],
            model_route=obj["model_route"],
            protocol=obj["protocol"],
            trials_per_task=obj["trials_per_task"],
            instruction_type=obj["instruction_type"],
            project_fixed_seeds=boolean(obj["project_fixed_seeds"], "robolab120_benchmark.project_fixed_seeds"),
            paper_seed_table_available=boolean(
                obj["paper_seed_table_available"],
                "robolab120_benchmark.paper_seed_table_available",
            ),
            plan_path=plan["path"],
            plan_sha256=plan["sha256"],
            profile_path=obj["profile"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "RoboLab120BenchmarkConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def load_plan(self, project_root: str | Path) -> BenchmarkPlan:
        root = Path(project_root).resolve()
        source = root / self.plan_path
        if self.model_route == ROBOLAB120_MODEL_ROUTE:
            plan = BenchmarkPlan.load(source)
        else:
            overlay = fields(
                _load_json(source),
                {
                    "schema_version",
                    "kind",
                    "template",
                    "plan_id",
                    "model_route",
                    "resolved_sha256",
                },
                path="robolab120_benchmark_plan_overlay",
            )
            if (
                integer(
                    overlay["schema_version"],
                    "robolab120_benchmark_plan_overlay.schema_version",
                )
                != 1
                or overlay["kind"] != "robolab120_benchmark_plan_overlay"
                or overlay["plan_id"] != self.benchmark_id
                or overlay["model_route"] != self.model_route
                or sha256(
                    overlay["resolved_sha256"],
                    "robolab120_benchmark_plan_overlay.resolved_sha256",
                )
                != self.plan_sha256
            ):
                raise StrictSchemaError("RoboLab benchmark plan overlay identity differs")
            template = fields(
                overlay["template"],
                {"path", "sha256"},
                path="robolab120_benchmark_plan_overlay.template",
            )
            template_path = _relative_json(
                template["path"],
                "robolab120_benchmark_plan_overlay.template.path",
            )
            template_plan = BenchmarkPlan.load(root / template_path)
            if (
                template_plan.plan_id != ROBOLAB120_BENCHMARK_ID
                or template_plan.model_route != ROBOLAB120_MODEL_ROUTE
                or template_plan.resolved_hash()
                != sha256(
                    template["sha256"],
                    "robolab120_benchmark_plan_overlay.template.sha256",
                )
            ):
                raise StrictSchemaError("RoboLab benchmark plan overlay template differs")
            plan = BenchmarkPlan(self.benchmark_id, self.model_route, template_plan.episodes)
        if plan.resolved_hash() != self.plan_sha256:
            raise StrictSchemaError("RoboLab benchmark plan hash differs")
        self.validate_plan(plan, root)
        return plan

    def load_profile(self, project_root: str | Path) -> Profile:
        root = Path(project_root).resolve()
        profile = Profile.load(root / self.profile_path, project_root=root)
        if profile.environment.suite != ROBOLAB120_SUITE:
            raise StrictSchemaError("RoboLab benchmark profile suite differs")
        if {item.identity.service_name for item in profile.policy.replicas} != {self.policy_service_name}:
            raise StrictSchemaError("RoboLab benchmark policy differs")
        return profile

    def validate_plan(self, plan: BenchmarkPlan, project_root: str | Path) -> None:
        root = Path(project_root).resolve()
        catalog = load_robolab120_catalog(
            root / "external" / "robolab",
            root / "manifests" / "robolab120.json",
        )
        if (
            plan.plan_id != self.benchmark_id
            or plan.model_route != self.model_route
            or len(plan.episodes) != len(catalog) * ROBOLAB120_TRIALS_PER_TASK
        ):
            raise StrictSchemaError("RoboLab benchmark plan identity or count differs")
        catalog_index = {item.task_id: index for index, item in enumerate(catalog)}
        horizons = {item.task_id: item.horizon for item in catalog}
        counts = {item.task_id: 0 for item in catalog}
        for episode in plan.episodes:
            task_index = catalog_index.get(episode.task_id)
            if task_index is None:
                raise StrictSchemaError("RoboLab benchmark task differs")
            slot = episode.policy_seed - ROBOLAB120_POLICY_SEED_BASE - task_index * ROBOLAB120_TRIALS_PER_TASK
            if (
                episode.split != "benchmark"
                or episode.scenario_id != ROBOLAB120_INSTRUCTION_TYPE
                or episode.environment_seed != ROBOLAB120_ENVIRONMENT_SEED
                or episode.protocol != ROBOLAB120_BENCHMARK_PROTOCOL
                or episode.horizon != horizons[episode.task_id]
                or slot not in range(ROBOLAB120_TRIALS_PER_TASK)
                or episode.replicate_id != f"b-task{task_index:03d}-slot{slot:02d}"
            ):
                raise StrictSchemaError("RoboLab benchmark episode differs")
            counts[episode.task_id] += 1
        if set(counts.values()) != {ROBOLAB120_TRIALS_PER_TASK}:
            raise StrictSchemaError("RoboLab benchmark task coverage differs")


class RoboLab120BenchmarkEvaluator:
    def __init__(
        self,
        profile: Profile,
        plan: BenchmarkPlan,
        *,
        agent_python: Path,
        simulator_python: Path,
        simulator_source: Path,
        runtime_root: Path,
        live_clients: Mapping[tuple[str, str], MsgpackServiceClient],
    ) -> None:
        if not isinstance(profile, Profile) or not isinstance(plan, BenchmarkPlan):
            raise StrictSchemaError("RoboLab benchmark evaluator requires Profile and BenchmarkPlan")
        if profile.environment.suite != ROBOLAB120_SUITE or profile.policy.deployment_mode != "replicated":
            raise StrictSchemaError("RoboLab benchmark profile differs")
        clients = dict(live_clients)
        replicas = []
        identities = []
        for item in profile.policy.replicas:
            key = (item.identity.service_name, item.identity.replica_id)
            if key not in clients:
                raise StrictSchemaError("RoboLab benchmark policy client is absent")
            actual = clients[key].validate_identity()
            if len(actual.gpu_ids) != 1:
                raise StrictSchemaError("RoboLab benchmark policy replica must use one GPU")
            identities.append(actual)
            replicas.append(ServiceReplica(item.endpoint, actual, clients[key]))
        tool_endpoints = {}
        for tool in profile.tools:
            if not tool.enabled or tool.service is None:
                continue
            item = tool.service
            key = (item.identity.service_name, item.identity.replica_id)
            if key not in clients:
                raise StrictSchemaError("RoboLab benchmark tool client is absent")
            identities.append(clients[key].validate_identity())
            tool_endpoints[tool.capability] = ToolEndpoint(
                item.endpoint,
                item.identity,
                tool.required,
                timeout_s=TOOL_CALL_TIMEOUT_S,
            )
        profile.validate_service_identities(identities)
        policy_service_names = {item.identity.service_name for item in replicas}
        if len(policy_service_names) != 1:
            raise StrictSchemaError("RoboLab benchmark policy service names differ")
        if profile.resources.workers % len(replicas):
            raise StrictSchemaError("RoboLab benchmark workers must divide across policy replicas")
        vector_batch_size = profile.resources.workers // len(replicas)
        planned_batches = build_robolab_benchmark_batches(plan, vector_batch_size=vector_batch_size)
        simulator_batch_capacity = max(len(batch.episodes) for batch in planned_batches)
        useful_apps_per_replica = max(1, (len(planned_batches) + len(replicas) - 1) // len(replicas))
        simulator_apps_per_replica = min(
            max(1, vector_batch_size // simulator_batch_capacity),
            useful_apps_per_replica,
        )
        self.profile = profile
        self.plan = plan
        self.agent_python = Path(agent_python).resolve()
        self.simulator_python = Path(simulator_python).resolve()
        self.simulator_source = Path(simulator_source).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.replicas = tuple(replicas)
        self.tool_endpoints = tool_endpoints
        self.identities = tuple(sorted(identities, key=lambda item: (item.service_name, item.replica_id)))
        self.policy_service_name = policy_service_names.pop()
        self.vector_batch_size = vector_batch_size
        self.simulator_batch_capacity = simulator_batch_capacity
        self.simulator_apps_per_replica = simulator_apps_per_replica
        self.episode_manifest_validator = None
        self.simulators: tuple[RoboLabSimulatorProcess, ...] = ()
        self.simulator_replicas: tuple[ServiceReplica, ...] = ()
        self.simulator_ids: tuple[str, ...] = ()

    def start(self) -> None:
        if self.simulators:
            raise RuntimeError("RoboLab benchmark evaluator is already started")
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        pairs = tuple(
            (replica, app_index)
            for replica in self.replicas
            for app_index in range(self.simulator_apps_per_replica)
        )
        simulators = tuple(
            RoboLabSimulatorProcess(
                self.simulator_python,
                self.profile,
                physical_gpu_id=replica.identity.gpu_ids[0],
                vector_batch_size=self.simulator_batch_capacity,
                runtime_dir=self.runtime_root / f"gpu{replica.identity.gpu_ids[0]}-app{app_index}",
                source_root=self.simulator_source,
            )
            for replica, app_index in pairs
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
        self.simulator_replicas = tuple(replica for replica, _ in pairs)
        self.simulator_ids = tuple(
            f"gpu{replica.identity.gpu_ids[0]}-app{app_index}"
            for replica, app_index in pairs
        )

    def close(self, force: bool = False) -> None:
        simulators = self.simulators
        self.simulators = ()
        self.simulator_replicas = ()
        self.simulator_ids = ()
        for simulator in simulators:
            simulator.close(force=force)

    @staticmethod
    def _request_id(key: EpisodeKey, step_index: int) -> str:
        return hashlib.sha256(f"{key.artifact_id()}\0{step_index}".encode()).hexdigest()

    def _gateway(self, scaffold: Path, replica: ServiceReplica, runtime: Path) -> AgentProcessGateway:
        return AgentProcessGateway(
            GatewayConfig(
                scaffold_path=scaffold / "scaffold.py",
                endpoints={
                    "vla": ToolEndpoint(
                        replica.endpoint,
                        replica.identity,
                        True,
                        timeout_s=POLICY_CALL_TIMEOUT_S,
                    ),
                    **self.tool_endpoints,
                },
                agent_python=self.agent_python,
                isolation_dir=runtime / "agent",
                expected_action_spec=self.profile.policy.action_spec,
                max_horizon=self.profile.policy.chunk_horizon,
                max_execution_count=self.profile.policy.execution_count,
                stderr_path=runtime / "agent.stderr.log",
                call_timeout_s=AGENT_STEP_TIMEOUT_S,
            )
        )

    def _noop(self, key: EpisodeKey, step_index: int, execution_count: int) -> CanonicalActionChunk:
        start = min(step_index, key.horizon - execution_count)
        return CanonicalActionChunk(
            request_id=f"frozen-{key.artifact_id()}-{step_index}",
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
        staging: Path,
        batch: RoboLabBatch,
        replica: ServiceReplica,
        simulator: RoboLabSimulatorProcess,
        simulator_app_id: str,
    ) -> None:
        runtime = self.runtime_root / "batches" / batch.batch_id
        runtime.mkdir(parents=True, exist_ok=False)
        gateway = self._gateway(scaffold, replica, runtime)
        traces: dict[str, list[dict[str, Any]]] = {key.artifact_id(): [] for key in batch.episodes}
        started = {key.artifact_id(): time.time_ns() for key in batch.episodes}
        simulator.load_batch(batch)
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
                    action = self.profile.validate_agent_action_chunk(result.action)
                    remaining = key.horizon - observation.step_index
                    if action.execution_count > remaining:
                        action = replace(action, execution_count=remaining)
                    actions[index] = action
                    events[index] = result.events
                    active_counts.add(action.execution_count)
                if len(active_counts) != 1:
                    raise RuntimeError("RoboLab benchmark active sessions returned different execution counts")
                execution_count = active_counts.pop()
                for index, (key, status, observation) in enumerate(
                    zip(batch.episodes, statuses, observations, strict=True)
                ):
                    if status.frozen:
                        actions[index] = self._noop(key, status.step_index, execution_count)
                    else:
                        traces[key.artifact_id()].append(
                            {
                                "observation": observation.to_mapping(),
                                "action": actions[index].to_mapping(),
                                "events": [item.to_mapping() for item in events[index]],
                            }
                        )
                simulator.apply_batch(batch, RoboLabActionBatch(batch.batch_id, tuple(actions)))
        statuses = simulator.private_status_batch(batch).statuses
        if not all(item.frozen for item in statuses):
            raise RuntimeError("RoboLab benchmark batch stopped before completion")
        simulator.finish_batch(batch)
        for key, status in zip(batch.episodes, statuses, strict=True):
            rows = traces[key.artifact_id()]
            execution = EpisodeExecution(
                state="complete",
                success=status.success,
                steps=len(rows),
                artifacts={
                    "execution.json": (
                        json.dumps(
                            {
                                "schema_version": 1,
                                "policy_service_name": replica.identity.service_name,
                                "policy_replica_id": replica.identity.replica_id,
                                "gpu_ids": list(replica.identity.gpu_ids),
                                "robolab_batch_id": batch.batch_id,
                                "vector_batch_size": len(batch.episodes),
                                "vector_batch_capacity": self.simulator_batch_capacity,
                                "simulator_app_id": simulator_app_id,
                                "configured_workers_per_policy_replica": self.vector_batch_size,
                                "simulator_apps_per_policy_replica": self.simulator_apps_per_replica,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode(),
                    "trace.msgpack": encode_message(
                        {
                            "outcome": {"key": key.to_mapping(), "success": status.success},
                            "termination": "success" if status.success else "horizon",
                            "steps": rows,
                            "error": None,
                        }
                    ),
                },
            )
            CanonicalBenchmarkEvaluator._record_episode(
                self,
                output,
                staging,
                key,
                execution,
                started[key.artifact_id()],
            )

    def _header(self, code_hash: str, created_ns: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark_plan": self.plan.to_mapping(),
            "benchmark_plan_sha256": self.plan.resolved_hash(),
            "profile_sha256": {ROBOLAB120_SUITE: self.profile.resolved_hash()},
            "code_sha256": code_hash,
            "service_identities": [item.to_mapping() for item in self.identities],
            "created_ns": created_ns,
        }

    def _prepare(self, output: Path, code_hash: str) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        (output / "episodes").mkdir(exist_ok=True)
        path = output / "run.json"
        if not path.exists():
            value = self._header(code_hash, time.time_ns())
            _atomic_write(path, canonical_json_bytes(value))
            return value
        value = _load_json(path)
        if not isinstance(value, Mapping):
            raise StrictSchemaError("RoboLab benchmark run header differs")
        expected = self._header(
            code_hash,
            integer(value.get("created_ns"), "robolab120_benchmark_run.created_ns", minimum=0),
        )
        if value != expected:
            raise StrictSchemaError("RoboLab benchmark run invariant differs")
        return dict(value)

    def _load_manifests(self, output: Path) -> tuple[EpisodeManifest, ...]:
        key_by_id = {key.artifact_id(): key for key in self.plan.episodes}
        directories = {path.name: path for path in (output / "episodes").iterdir()}
        if set(directories) - set(key_by_id):
            raise StrictSchemaError("RoboLab benchmark output contains an unknown episode")
        return tuple(
            _verify_episode_directory(directories[artifact_id], key)
            for artifact_id, key in key_by_id.items()
            if artifact_id in directories
        )

    def _report(self, manifests: tuple[EpisodeManifest, ...], errors: int) -> dict[str, Any]:
        complete = len(manifests) == len(self.plan.episodes) and errors == 0
        return {
            "schema_version": 1,
            "benchmark_plan_sha256": self.plan.resolved_hash(),
            "n_expected": len(self.plan.episodes),
            "n_complete": len(manifests),
            "n_pending": len(self.plan.episodes) - len(manifests),
            "errors_this_invocation": errors,
            "complete": complete,
            "metrics": robolab120_metrics(manifests) if complete else None,
            "runtime": {
                "policy_replica_count": len(self.replicas),
                "configured_workers_per_policy_replica": self.vector_batch_size,
                "vector_batch_size_per_simulator_app": self.simulator_batch_capacity,
                "actual_vector_batch_sizes": sorted(
                    {
                        len(batch.episodes)
                        for batch in build_robolab_benchmark_batches(
                            self.plan,
                            vector_batch_size=self.vector_batch_size,
                        )
                    }
                ),
                "simulator_apps_per_policy_replica": self.simulator_apps_per_replica,
                "simulator_app_count": len(self.replicas) * self.simulator_apps_per_replica,
                "planned_concurrent_episode_slots_per_policy_replica": (
                    self.simulator_batch_capacity * self.simulator_apps_per_replica
                ),
            },
            "updated_ns": time.time_ns(),
        }

    def evaluate(self, scaffold_dir: Path, output_dir: Path, invocation_dir: Path) -> dict[str, Any]:
        scaffold = Path(scaffold_dir).resolve()
        output = Path(output_dir).resolve()
        invocation = Path(invocation_dir).resolve()
        invocation.mkdir(parents=True, exist_ok=False)
        code_hash = mapping_sha256(tree_hashes(scaffold))
        self._prepare(output, code_hash)
        _preserve_scaffold(output, scaffold, code_hash)
        if (output / "final.json").is_file():
            verify_benchmark_output(
                output,
                plan_validator=lambda value: self._validate_resumed_plan(value),
                metric_function=robolab120_metrics,
                runtime_function=lambda manifests, episode_root: robolab120_runtime(
                    manifests,
                    episode_root,
                    expected_policy_service_name=self.policy_service_name,
                ),
                profile_suites=(ROBOLAB120_SUITE,),
            )
            return dict(_load_json(output / "report.json"))
        if not self.simulators:
            self.start()
        if not (
            len(self.simulator_replicas) == len(self.simulators) == len(self.simulator_ids)
        ):
            raise RuntimeError("RoboLab benchmark simulator-to-policy assignment differs")
        batches = build_robolab_benchmark_batches(self.plan, vector_batch_size=self.vector_batch_size)
        existing = {item.key for item in self._load_manifests(output)}
        lanes: list[list[RoboLabBatch]] = [[] for _ in self.simulators]
        for index, batch in enumerate(batches):
            present = [key in existing for key in batch.episodes]
            if any(present) and not all(present):
                raise StrictSchemaError("RoboLab benchmark has a partially recorded vector batch")
            if not all(present):
                lanes[index % len(lanes)].append(batch)
        staging = invocation / "episode_staging"
        errors = []

        def execute_lane(index: int) -> None:
            for batch in lanes[index]:
                try:
                    self._run_batch(
                        scaffold,
                        output,
                        staging,
                        batch,
                        self.simulator_replicas[index],
                        self.simulators[index],
                        self.simulator_ids[index],
                    )
                except Exception as exc:
                    for key in batch.episodes:
                        CanonicalBenchmarkEvaluator._write_error(invocation / "errors", key, exc)
                    raise

        with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
            futures = [executor.submit(execute_lane, index) for index in range(len(lanes))]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
        if not errors:
            barrier = f"benchmark-{self.plan.resolved_hash()[:16]}"
            for simulator in self.simulators:
                simulator.candidate_barrier(barrier)
        manifests = self._load_manifests(output)
        report = self._report(manifests, len(errors))
        _atomic_write(output / "report.json", canonical_json_bytes(report))
        if report["complete"]:
            manifest_hashes = {
                item.key.artifact_id(): hashlib.sha256(
                    (output / "episodes" / item.key.artifact_id() / "episode.json").read_bytes()
                ).hexdigest()
                for item in manifests
            }
            final = {
                "schema_version": 1,
                "benchmark_plan_sha256": self.plan.resolved_hash(),
                "run_sha256": hashlib.sha256((output / "run.json").read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
                "episode_manifest_sha256": manifest_hashes,
                "n_complete": len(manifests),
                "finalized_ns": time.time_ns(),
            }
            final["manifest_sha256"] = mapping_sha256(final)
            _atomic_write(output / "final.json", canonical_json_bytes(final))
        if errors:
            raise RuntimeError(f"RoboLab benchmark invocation had {len(errors)} lane errors")
        if not report["complete"]:
            raise RuntimeError("RoboLab benchmark invocation is incomplete")
        return report

    def _validate_resumed_plan(self, plan: BenchmarkPlan) -> None:
        if plan.to_mapping() != self.plan.to_mapping():
            raise StrictSchemaError("RoboLab resumed benchmark plan differs")


def verify_robolab120_benchmark_output(
    path: str | Path,
    config: RoboLab120BenchmarkConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    return verify_benchmark_output(
        path,
        plan_validator=lambda plan: config.validate_plan(plan, project_root),
        metric_function=robolab120_metrics,
        runtime_function=lambda manifests, episode_root: robolab120_runtime(
            manifests,
            episode_root,
            expected_policy_service_name=config.policy_service_name,
        ),
        profile_suites=(ROBOLAB120_SUITE,),
    )
