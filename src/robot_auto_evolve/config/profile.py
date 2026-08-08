from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from robot_auto_evolve.protocol import CanonicalActionChunk, CanonicalActionSpec, FairObservation, RobotStateSpec
from robot_auto_evolve.protocol.observation import OPTICAL_CONVENTIONS
from robot_auto_evolve.protocol.schema import (
    StrictSchemaError,
    boolean,
    enum,
    fields,
    integer,
    json_object,
    number,
    reject_json_constant,
    sequence,
    string,
)
from robot_auto_evolve.provenance import EpisodePlan
from robot_auto_evolve.services.identity import ServiceIdentity


def _url(value: Any, path: str) -> str:
    result = string(value, path).rstrip("/")
    parsed = urlparse(result)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
        raise StrictSchemaError(f"{path}: expected HTTP service base URL")
    if parsed.params or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise StrictSchemaError(f"{path}: invalid service base URL")
    return result


@dataclass(frozen=True)
class CameraSpec:
    name: str
    frame_id: str
    optical_convention: str
    width: int
    height: int
    has_depth: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", string(self.name, "camera_spec.name"))
        object.__setattr__(self, "frame_id", string(self.frame_id, "camera_spec.frame_id"))
        object.__setattr__(
            self,
            "optical_convention",
            enum(self.optical_convention, OPTICAL_CONVENTIONS, "camera_spec.optical_convention"),
        )
        object.__setattr__(self, "width", integer(self.width, "camera_spec.width", minimum=1))
        object.__setattr__(self, "height", integer(self.height, "camera_spec.height", minimum=1))
        object.__setattr__(self, "has_depth", boolean(self.has_depth, "camera_spec.has_depth"))

    @classmethod
    def from_mapping(cls, value: Any) -> "CameraSpec":
        obj = fields(
            value,
            {"name", "frame_id", "optical_convention", "width", "height", "has_depth"},
            path="camera_spec",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frame_id": self.frame_id,
            "optical_convention": self.optical_convention,
            "width": self.width,
            "height": self.height,
            "has_depth": self.has_depth,
        }


@dataclass(frozen=True)
class EnvironmentProfile:
    adapter: str
    suite: str
    embodiment: str
    cameras: tuple[CameraSpec, ...]
    robot_state: tuple[RobotStateSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter", string(self.adapter, "environment.adapter"))
        object.__setattr__(self, "suite", string(self.suite, "environment.suite"))
        object.__setattr__(self, "embodiment", string(self.embodiment, "environment.embodiment"))
        cameras = tuple(self.cameras)
        if not cameras or any(not isinstance(item, CameraSpec) for item in cameras):
            raise StrictSchemaError("environment.cameras: expected nonempty CameraSpec sequence")
        if [item.name for item in cameras] != sorted(item.name for item in cameras):
            raise StrictSchemaError("environment.cameras: expected sorted unique names")
        if len({item.name for item in cameras}) != len(cameras):
            raise StrictSchemaError("environment.cameras: duplicate names")
        robot_state = tuple(self.robot_state)
        if not robot_state or any(not isinstance(item, RobotStateSpec) for item in robot_state):
            raise StrictSchemaError("environment.robot_state: expected nonempty RobotStateSpec sequence")
        if [item.name for item in robot_state] != sorted(item.name for item in robot_state):
            raise StrictSchemaError("environment.robot_state: expected sorted unique names")
        if len({item.name for item in robot_state}) != len(robot_state):
            raise StrictSchemaError("environment.robot_state: duplicate names")
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "robot_state", robot_state)

    @classmethod
    def from_mapping(cls, value: Any) -> "EnvironmentProfile":
        obj = fields(value, {"adapter", "suite", "embodiment", "cameras", "robot_state"}, path="environment")
        return cls(
            adapter=obj["adapter"],
            suite=obj["suite"],
            embodiment=obj["embodiment"],
            cameras=tuple(CameraSpec.from_mapping(item) for item in sequence(obj["cameras"], "environment.cameras")),
            robot_state=tuple(
                RobotStateSpec.from_mapping(item)
                for item in sequence(obj["robot_state"], "environment.robot_state")
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "suite": self.suite,
            "embodiment": self.embodiment,
            "cameras": [item.to_mapping() for item in self.cameras],
            "robot_state": [item.to_mapping() for item in self.robot_state],
        }


@dataclass(frozen=True)
class ServiceEndpointProfile:
    endpoint: str
    identity: ServiceIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _url(self.endpoint, "service.endpoint"))
        if not isinstance(self.identity, ServiceIdentity):
            raise StrictSchemaError("service.identity: expected ServiceIdentity")

    @classmethod
    def from_mapping(cls, value: Any) -> "ServiceEndpointProfile":
        obj = fields(value, {"endpoint", "identity"}, path="service")
        return cls(endpoint=obj["endpoint"], identity=ServiceIdentity.from_mapping(obj["identity"]))

    def to_mapping(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "identity": self.identity.to_mapping()}


@dataclass(frozen=True)
class PolicyProfile:
    adapter: str
    action_spec: CanonicalActionSpec
    chunk_horizon: int
    execution_count: int
    deployment_mode: str
    replicas: tuple[ServiceEndpointProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter", string(self.adapter, "policy.adapter"))
        if not isinstance(self.action_spec, CanonicalActionSpec):
            raise StrictSchemaError("policy.action_spec: expected CanonicalActionSpec")
        horizon = integer(self.chunk_horizon, "policy.chunk_horizon", minimum=1)
        count = integer(self.execution_count, "policy.execution_count", minimum=1, maximum=horizon)
        mode = enum(self.deployment_mode, {"replicated", "tensor_parallel", "fixture"}, "policy.deployment_mode")
        replicas = tuple(self.replicas)
        if any(not isinstance(item, ServiceEndpointProfile) for item in replicas):
            raise StrictSchemaError("policy.replicas: expected ServiceEndpointProfile entries")
        identities = [item.identity for item in replicas]
        if any(item.service_kind != "policy" for item in identities):
            raise StrictSchemaError("policy.replicas: service_kind must be policy")
        if mode == "replicated":
            if not replicas or any(len(item.gpu_ids) != 1 for item in identities):
                raise StrictSchemaError("policy.replicas: replicated mode requires one GPU per replica")
            if any(not identities[0].same_model_as(item) for item in identities[1:]):
                raise StrictSchemaError("policy.replicas: model identities differ")
        elif mode == "tensor_parallel":
            if len(replicas) != 1 or len(identities[0].gpu_ids) < 2:
                raise StrictSchemaError("policy.replicas: tensor_parallel mode requires one multi-GPU service")
        elif replicas:
            raise StrictSchemaError("policy.replicas: fixture mode requires no services")
        if len({item.identity.replica_id for item in replicas}) != len(replicas):
            raise StrictSchemaError("policy.replicas: duplicate replica_id")
        object.__setattr__(self, "chunk_horizon", horizon)
        object.__setattr__(self, "execution_count", count)
        object.__setattr__(self, "deployment_mode", mode)
        object.__setattr__(self, "replicas", replicas)

    @classmethod
    def from_mapping(cls, value: Any) -> "PolicyProfile":
        obj = fields(
            value,
            {"adapter", "action_spec", "chunk_horizon", "execution_count", "deployment_mode", "replicas"},
            path="policy",
        )
        return cls(
            adapter=obj["adapter"],
            action_spec=CanonicalActionSpec.from_mapping(obj["action_spec"]),
            chunk_horizon=obj["chunk_horizon"],
            execution_count=obj["execution_count"],
            deployment_mode=obj["deployment_mode"],
            replicas=tuple(
                ServiceEndpointProfile.from_mapping(item)
                for item in sequence(obj["replicas"], "policy.replicas")
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "action_spec": self.action_spec.to_mapping(),
            "chunk_horizon": self.chunk_horizon,
            "execution_count": self.execution_count,
            "deployment_mode": self.deployment_mode,
            "replicas": [item.to_mapping() for item in self.replicas],
        }


@dataclass(frozen=True)
class ToolProfile:
    capability: str
    enabled: bool
    required: bool
    availability: str
    blocker: str | None
    service: ServiceEndpointProfile | None

    def __post_init__(self) -> None:
        capability = enum(
            self.capability,
            {"language", "vision", "detection", "segmentation", "pointing", "grasp"},
            "tool.capability",
        )
        enabled = boolean(self.enabled, "tool.enabled")
        required = boolean(self.required, "tool.required")
        availability = enum(self.availability, {"available", "blocked", "unavailable"}, "tool.availability")
        blocker = None if self.blocker is None else string(self.blocker, "tool.blocker")
        if self.service is not None and not isinstance(self.service, ServiceEndpointProfile):
            raise StrictSchemaError("tool.service: expected ServiceEndpointProfile or null")
        if availability == "available":
            if self.service is None or blocker is not None:
                raise StrictSchemaError("tool: available requires service and null blocker")
            if self.service.identity.service_kind != capability:
                raise StrictSchemaError("tool: service identity mismatch")
        elif self.service is not None or blocker is None:
            raise StrictSchemaError("tool: blocked or unavailable requires null service and blocker")
        if enabled and availability != "available":
            raise StrictSchemaError("tool: enabled tool is not available")
        if required and not enabled:
            raise StrictSchemaError("tool: required tool is not enabled")
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "blocker", blocker)

    @classmethod
    def from_mapping(cls, value: Any) -> "ToolProfile":
        obj = fields(value, {"capability", "enabled", "required", "availability", "blocker", "service"}, path="tool")
        return cls(
            capability=obj["capability"],
            enabled=obj["enabled"],
            required=obj["required"],
            availability=obj["availability"],
            blocker=obj["blocker"],
            service=None if obj["service"] is None else ServiceEndpointProfile.from_mapping(obj["service"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "enabled": self.enabled,
            "required": self.required,
            "availability": self.availability,
            "blocker": self.blocker,
            "service": None if self.service is None else self.service.to_mapping(),
        }


@dataclass(frozen=True)
class EpisodePlanReference:
    path: str

    def __post_init__(self) -> None:
        path = string(self.path, "episode_plan.path")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or pure.suffix.lower() != ".json":
            raise StrictSchemaError("episode_plan.path: expected safe relative JSON path")
        object.__setattr__(self, "path", pure.as_posix())

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodePlanReference":
        obj = fields(value, {"path"}, path="episode_plan")
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {"path": self.path}

    def load(self, base_dir: str | Path) -> EpisodePlan:
        root = Path(base_dir).resolve()
        source = (root / self.path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise StrictSchemaError("episode_plan.path: resolved path escapes project root") from exc
        return EpisodePlan.load(source)


@dataclass(frozen=True)
class MetaLoopProfile:
    candidate_budget: int
    coding_backend: str
    coding_model: str | None
    max_turns: int
    timeout_s: int
    api_request_budget: int
    api_request_max_bytes: int
    api_response_max_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_budget",
            integer(self.candidate_budget, "meta_loop.candidate_budget", minimum=1),
        )
        backend = enum(self.coding_backend, {"claude", "fixture"}, "meta_loop.coding_backend")
        object.__setattr__(
            self,
            "coding_backend",
            backend,
        )
        model = None if self.coding_model is None else string(self.coding_model, "meta_loop.coding_model")
        if model is not None and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model) is None:
            raise StrictSchemaError("meta_loop.coding_model: invalid model identifier")
        object.__setattr__(self, "max_turns", integer(self.max_turns, "meta_loop.max_turns", minimum=1))
        object.__setattr__(self, "timeout_s", integer(self.timeout_s, "meta_loop.timeout_s", minimum=1))
        request_budget = integer(self.api_request_budget, "meta_loop.api_request_budget", minimum=0, maximum=1024)
        request_bytes = integer(
            self.api_request_max_bytes,
            "meta_loop.api_request_max_bytes",
            minimum=0,
            maximum=256 * 1024**2,
        )
        response_bytes = integer(
            self.api_response_max_bytes,
            "meta_loop.api_response_max_bytes",
            minimum=0,
            maximum=256 * 1024**2,
        )
        if backend == "claude" and (model is None or min(request_budget, request_bytes, response_bytes) < 1):
            raise StrictSchemaError("meta_loop: Claude backend requires a model and positive API limits")
        if backend == "fixture" and (model is not None or any((request_budget, request_bytes, response_bytes))):
            raise StrictSchemaError("meta_loop: fixture backend requires null model and zero API limits")
        object.__setattr__(self, "coding_model", model)
        object.__setattr__(self, "api_request_budget", request_budget)
        object.__setattr__(self, "api_request_max_bytes", request_bytes)
        object.__setattr__(self, "api_response_max_bytes", response_bytes)

    @classmethod
    def from_mapping(cls, value: Any) -> "MetaLoopProfile":
        obj = fields(
            value,
            {
                "candidate_budget",
                "coding_backend",
                "coding_model",
                "max_turns",
                "timeout_s",
                "api_request_budget",
                "api_request_max_bytes",
                "api_response_max_bytes",
            },
            path="meta_loop",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_budget": self.candidate_budget,
            "coding_backend": self.coding_backend,
            "coding_model": self.coding_model,
            "max_turns": self.max_turns,
            "timeout_s": self.timeout_s,
            "api_request_budget": self.api_request_budget,
            "api_request_max_bytes": self.api_request_max_bytes,
            "api_response_max_bytes": self.api_response_max_bytes,
        }


@dataclass(frozen=True)
class ResourceProfile:
    mode: str
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        mode = enum(self.mode, {"two_gpu", "multi_gpu", "fixture"}, "resources.mode")
        if isinstance(self.gpu_ids, (str, bytes)):
            raise StrictSchemaError("resources.gpu_ids: expected sequence")
        gpu_ids = tuple(integer(item, f"resources.gpu_ids[{index}]", minimum=0) for index, item in enumerate(self.gpu_ids))
        if gpu_ids != tuple(sorted(set(gpu_ids))):
            raise StrictSchemaError("resources.gpu_ids: expected sorted unique values")
        if mode == "two_gpu" and len(gpu_ids) != 2:
            raise StrictSchemaError("resources.gpu_ids: two_gpu requires exactly two values")
        if mode == "multi_gpu" and len(gpu_ids) < 2:
            raise StrictSchemaError("resources.gpu_ids: multi_gpu requires at least two values")
        if mode == "fixture" and gpu_ids:
            raise StrictSchemaError("resources.gpu_ids: fixture requires no values")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "gpu_ids", gpu_ids)

    @classmethod
    def from_mapping(cls, value: Any) -> "ResourceProfile":
        obj = fields(value, {"mode", "gpu_ids"}, path="resources")
        return cls(mode=obj["mode"], gpu_ids=tuple(sequence(obj["gpu_ids"], "resources.gpu_ids")))

    def to_mapping(self) -> dict[str, Any]:
        return {"mode": self.mode, "gpu_ids": list(self.gpu_ids)}


@dataclass(frozen=True)
class Profile:
    profile_id: str
    environment: EnvironmentProfile
    policy: PolicyProfile
    tools: tuple[ToolProfile, ...]
    episode_plan: EpisodePlanReference
    meta_loop: MetaLoopProfile
    resources: ResourceProfile
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("profile.schema_version: expected 1")
        object.__setattr__(self, "profile_id", string(self.profile_id, "profile.profile_id"))
        expected_types = {
            "environment": EnvironmentProfile,
            "policy": PolicyProfile,
            "episode_plan": EpisodePlanReference,
            "meta_loop": MetaLoopProfile,
            "resources": ResourceProfile,
        }
        for name, expected in expected_types.items():
            if not isinstance(getattr(self, name), expected):
                raise StrictSchemaError(f"profile.{name}: expected {expected.__name__}")
        tools = tuple(self.tools)
        if any(not isinstance(tool, ToolProfile) for tool in tools):
            raise StrictSchemaError("profile.tools: expected ToolProfile entries")
        if [tool.capability for tool in tools] != sorted(tool.capability for tool in tools):
            raise StrictSchemaError("profile.tools: expected sorted unique capabilities")
        if len({tool.capability for tool in tools}) != len(tools):
            raise StrictSchemaError("profile.tools: duplicate capabilities")
        if self.policy.deployment_mode == "fixture" and self.resources.mode != "fixture":
            raise StrictSchemaError("profile: fixture policy requires fixture resources")
        if self.policy.deployment_mode != "fixture" and self.resources.mode not in {"two_gpu", "multi_gpu"}:
            raise StrictSchemaError("profile: deployed policy requires GPU resources")
        if self.policy.deployment_mode == "replicated":
            policy_gpu_ids = tuple(item.identity.gpu_ids[0] for item in self.policy.replicas)
            counts = {gpu_id: policy_gpu_ids.count(gpu_id) for gpu_id in self.resources.gpu_ids}
            if (
                set(policy_gpu_ids) != set(self.resources.gpu_ids)
                or len(set(counts.values())) != 1
                or policy_gpu_ids != tuple(sorted(policy_gpu_ids))
            ):
                raise StrictSchemaError(
                    "profile: policy replicas must be the same number of copies on each resource GPU, in GPU order"
                )
        if self.policy.deployment_mode == "tensor_parallel":
            if self.policy.replicas[0].identity.gpu_ids != self.resources.gpu_ids:
                raise StrictSchemaError("profile: tensor-parallel GPUs differ from resource GPUs")
        if self.policy.deployment_mode != "fixture":
            services = list(self.policy.replicas)
            services.extend(
                tool.service
                for tool in tools
                if tool.enabled and tool.service is not None
            )
            if len({item.endpoint for item in services}) != len(services):
                raise StrictSchemaError("profile: duplicate service endpoint")
            resource_gpus = set(self.resources.gpu_ids)
            if any(not set(item.identity.gpu_ids) <= resource_gpus for item in services):
                raise StrictSchemaError("profile: service GPU falls outside resource GPUs")
        object.__setattr__(self, "tools", tools)

    @classmethod
    def from_mapping(cls, value: Any) -> "Profile":
        obj = fields(
            value,
            {"schema_version", "profile_id", "environment", "policy", "tools", "episode_plan", "meta_loop", "resources"},
            path="profile",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "profile.schema_version"),
            profile_id=obj["profile_id"],
            environment=EnvironmentProfile.from_mapping(obj["environment"]),
            policy=PolicyProfile.from_mapping(obj["policy"]),
            tools=tuple(ToolProfile.from_mapping(item) for item in sequence(obj["tools"], "profile.tools")),
            episode_plan=EpisodePlanReference.from_mapping(obj["episode_plan"]),
            meta_loop=MetaLoopProfile.from_mapping(obj["meta_loop"]),
            resources=ResourceProfile.from_mapping(obj["resources"]),
        )

    @classmethod
    def load(cls, path: str | Path, *, project_root: str | Path | None = None) -> "Profile":
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise StrictSchemaError("profile: expected .json file")
        try:
            if project_root is None:
                config_parent = next((parent for parent in source.parents if parent.name == "configs"), None)
                root = config_parent.parent if config_parent is not None else source.parent
            else:
                root = Path(project_root)
            resolved_root = root.resolve()
            resolved_source = source.resolve()
            try:
                resolved_source.relative_to(resolved_root)
            except ValueError as exc:
                raise StrictSchemaError("profile: source escapes project root") from exc
            with source.open("r", encoding="utf-8") as stream:
                profile = cls.from_mapping(
                    json.load(stream, object_pairs_hook=json_object, parse_constant=reject_json_constant)
                )
            profile.validate(profile.episode_plan.load(resolved_root))
            return profile
        except StrictSchemaError:
            raise
        except Exception as exc:
            raise StrictSchemaError(f"profile: failed to load {source}: {exc}") from exc

    def validate(self, episode_plan: EpisodePlan | None = None) -> "Profile":
        Profile.from_mapping(self.to_mapping())
        if episode_plan is not None and not isinstance(episode_plan, EpisodePlan):
            raise StrictSchemaError("profile.episode_plan: expected EpisodePlan")
        return self

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "environment": self.environment.to_mapping(),
            "policy": self.policy.to_mapping(),
            "tools": [tool.to_mapping() for tool in self.tools],
            "episode_plan": self.episode_plan.to_mapping(),
            "meta_loop": self.meta_loop.to_mapping(),
            "resources": self.resources.to_mapping(),
        }

    def validate_observation(self, observation: FairObservation) -> FairObservation:
        if not isinstance(observation, FairObservation):
            raise StrictSchemaError("profile.observation: expected FairObservation")
        expected_cameras = {item.name: item for item in self.environment.cameras}
        if set(observation.cameras) != set(expected_cameras):
            raise StrictSchemaError("profile.observation: camera names differ")
        for name, expected in expected_cameras.items():
            actual = observation.cameras[name]
            if actual.frame_id != expected.frame_id or actual.optical_convention != expected.optical_convention:
                raise StrictSchemaError(f"profile.observation.cameras.{name}: frame convention differs")
            if actual.rgb.shape[:2] != (expected.height, expected.width):
                raise StrictSchemaError(f"profile.observation.cameras.{name}: image shape differs")
            if (actual.depth_m is not None) != expected.has_depth:
                raise StrictSchemaError(f"profile.observation.cameras.{name}: depth availability differs")
        expected_state = tuple(item.to_mapping() for item in self.environment.robot_state)
        actual_state = tuple(item.spec.to_mapping() for item in observation.proprioception.vectors)
        if actual_state != expected_state:
            raise StrictSchemaError("profile.observation: robot state specification differs")
        return observation

    def _validate_action_spec(self, chunk: CanonicalActionChunk) -> None:
        if not isinstance(chunk, CanonicalActionChunk):
            raise StrictSchemaError("profile.action: expected CanonicalActionChunk")
        if chunk.spec != self.policy.action_spec:
            raise StrictSchemaError("profile.action: action specification differs")

    def validate_agent_action_chunk(self, chunk: CanonicalActionChunk) -> CanonicalActionChunk:
        self._validate_action_spec(chunk)
        if chunk.horizon > self.policy.chunk_horizon:
            raise StrictSchemaError("profile.agent_action: chunk horizon exceeds limit")
        if chunk.execution_count > self.policy.execution_count:
            raise StrictSchemaError("profile.agent_action: execution count exceeds limit")
        return chunk

    def validate_action_chunk(self, chunk: CanonicalActionChunk) -> CanonicalActionChunk:
        return self.validate_agent_action_chunk(chunk)
