from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from robot_auto_evolve.agent.framing import read_frame, write_frame
from robot_auto_evolve.protocol import CanonicalActionChunk, FairObservation, StrictSchemaError
from robot_auto_evolve.protocol.schema import boolean, enum, fields, integer, mapping, sequence, sha256, string
from robot_auto_evolve.provenance import mapping_sha256

from .robolab120_batching import RoboLabBatch


ROBOLAB_RPC_OPERATIONS = frozenset(
    {
        "initialize_app",
        "load_batch",
        "observe_batch",
        "apply_batch",
        "private_status_batch",
        "finish_batch",
        "candidate_barrier",
        "close",
    }
)


def _json_value(value: Any, path: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StrictSchemaError(f"{path}: expected finite JSON number")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise StrictSchemaError(f"{path}: expected string keys")
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise StrictSchemaError(f"{path}: expected JSON value")


def _git_commit(value: Any, path: str) -> str:
    result = string(value, path)
    if re.fullmatch(r"[0-9a-f]{40}", result) is None:
        raise StrictSchemaError(f"{path}: expected lowercase Git commit")
    return result


def _episode_ids(value: Any, path: str) -> tuple[str, ...]:
    result = tuple(string(item, f"{path}[{index}]") for index, item in enumerate(sequence(value, path)))
    if not result:
        raise StrictSchemaError(f"{path}: empty sequence")
    if len(set(result)) != len(result):
        raise StrictSchemaError(f"{path}: duplicate episode ids")
    return result


def _batch_reference(value: Any, path: str) -> dict[str, Any]:
    obj = fields(value, {"batch_id"}, path=path)
    return {"batch_id": sha256(obj["batch_id"], f"{path}.batch_id")}


@dataclass(frozen=True)
class RoboLabAppConfig:
    static_profile: Mapping[str, Any]
    static_profile_sha256: str
    source_commit: str
    asset_lock_sha256: str
    simulator_gpu_id: int
    vector_batch_size: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab_app_config.schema_version: expected 1")
        profile = _json_value(mapping(self.static_profile, "robolab_app_config.static_profile"), "robolab_app_config.static_profile")
        profile_hash = sha256(self.static_profile_sha256, "robolab_app_config.static_profile_sha256")
        if mapping_sha256(profile) != profile_hash:
            raise StrictSchemaError("robolab_app_config.static_profile_sha256: hash mismatch")
        object.__setattr__(self, "static_profile", profile)
        object.__setattr__(self, "static_profile_sha256", profile_hash)
        object.__setattr__(self, "source_commit", _git_commit(self.source_commit, "robolab_app_config.source_commit"))
        object.__setattr__(
            self,
            "asset_lock_sha256",
            sha256(self.asset_lock_sha256, "robolab_app_config.asset_lock_sha256"),
        )
        object.__setattr__(
            self,
            "simulator_gpu_id",
            integer(self.simulator_gpu_id, "robolab_app_config.simulator_gpu_id", minimum=0),
        )
        object.__setattr__(
            self,
            "vector_batch_size",
            integer(self.vector_batch_size, "robolab_app_config.vector_batch_size", minimum=1),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabAppConfig":
        obj = fields(
            value,
            {
                "schema_version",
                "static_profile",
                "static_profile_sha256",
                "source_commit",
                "asset_lock_sha256",
                "simulator_gpu_id",
                "vector_batch_size",
            },
            path="robolab_app_config",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "robolab_app_config.schema_version"),
            static_profile=obj["static_profile"],
            static_profile_sha256=obj["static_profile_sha256"],
            source_commit=obj["source_commit"],
            asset_lock_sha256=obj["asset_lock_sha256"],
            simulator_gpu_id=obj["simulator_gpu_id"],
            vector_batch_size=obj["vector_batch_size"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "static_profile": dict(self.static_profile),
            "static_profile_sha256": self.static_profile_sha256,
            "source_commit": self.source_commit,
            "asset_lock_sha256": self.asset_lock_sha256,
            "simulator_gpu_id": self.simulator_gpu_id,
            "vector_batch_size": self.vector_batch_size,
        }


@dataclass(frozen=True)
class RoboLabObservationBatch:
    batch_id: str
    observations: tuple[FairObservation, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab_observation_batch.schema_version: expected 1")
        batch_id = sha256(self.batch_id, "robolab_observation_batch.batch_id")
        observations = tuple(self.observations)
        if not observations or any(not isinstance(item, FairObservation) for item in observations):
            raise StrictSchemaError("robolab_observation_batch.observations: expected nonempty FairObservation sequence")
        ids = [item.episode_id for item in observations]
        if len(set(ids)) != len(ids):
            raise StrictSchemaError("robolab_observation_batch.observations: duplicate episode ids")
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "observations", observations)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabObservationBatch":
        obj = fields(
            value,
            {"schema_version", "batch_id", "observations"},
            path="robolab_observation_batch",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "robolab_observation_batch.schema_version"),
            batch_id=obj["batch_id"],
            observations=tuple(
                FairObservation.from_mapping(item)
                for item in sequence(obj["observations"], "robolab_observation_batch.observations")
            ),
        )

    def validate_batch(self, batch: RoboLabBatch) -> None:
        if not isinstance(batch, RoboLabBatch):
            raise StrictSchemaError("robolab_observation_batch.batch: expected RoboLabBatch")
        if self.batch_id != batch.batch_id or tuple(item.episode_id for item in self.observations) != batch.episode_ids:
            raise StrictSchemaError("robolab_observation_batch: batch identity or episode order differs")
        if any(item.step_index > episode.horizon for item, episode in zip(self.observations, batch.episodes)):
            raise StrictSchemaError("robolab_observation_batch: step exceeds episode horizon")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "observations": [item.to_mapping() for item in self.observations],
        }


@dataclass(frozen=True)
class RoboLabActionBatch:
    batch_id: str
    actions: tuple[CanonicalActionChunk, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab_action_batch.schema_version: expected 1")
        batch_id = sha256(self.batch_id, "robolab_action_batch.batch_id")
        actions = tuple(self.actions)
        if not actions or any(not isinstance(item, CanonicalActionChunk) for item in actions):
            raise StrictSchemaError("robolab_action_batch.actions: expected nonempty CanonicalActionChunk sequence")
        ids = [item.session_id for item in actions]
        if len(set(ids)) != len(ids):
            raise StrictSchemaError("robolab_action_batch.actions: duplicate session ids")
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "actions", actions)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabActionBatch":
        obj = fields(value, {"schema_version", "batch_id", "actions"}, path="robolab_action_batch")
        return cls(
            schema_version=integer(obj["schema_version"], "robolab_action_batch.schema_version"),
            batch_id=obj["batch_id"],
            actions=tuple(
                CanonicalActionChunk.from_mapping(item)
                for item in sequence(obj["actions"], "robolab_action_batch.actions")
            ),
        )

    def validate_batch(self, batch: RoboLabBatch) -> None:
        if not isinstance(batch, RoboLabBatch):
            raise StrictSchemaError("robolab_action_batch.batch: expected RoboLabBatch")
        if self.batch_id != batch.batch_id or tuple(item.session_id for item in self.actions) != batch.episode_ids:
            raise StrictSchemaError("robolab_action_batch: batch identity or episode order differs")
        if any(
            item.start_step + item.execution_count > episode.horizon
            for item, episode in zip(self.actions, batch.episodes)
        ):
            raise StrictSchemaError("robolab_action_batch: execution exceeds episode horizon")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "actions": [item.to_mapping() for item in self.actions],
        }


@dataclass(frozen=True)
class RoboLabPrivateStatus:
    episode_id: str
    step_index: int
    terminated: bool
    truncated: bool
    success: bool
    frozen: bool

    def __post_init__(self) -> None:
        episode_id = string(self.episode_id, "robolab_private_status.episode_id")
        step_index = integer(self.step_index, "robolab_private_status.step_index", minimum=0)
        terminated = boolean(self.terminated, "robolab_private_status.terminated")
        truncated = boolean(self.truncated, "robolab_private_status.truncated")
        success = boolean(self.success, "robolab_private_status.success")
        frozen = boolean(self.frozen, "robolab_private_status.frozen")
        if frozen != (terminated or truncated):
            raise StrictSchemaError("robolab_private_status.frozen: termination state differs")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "frozen", frozen)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabPrivateStatus":
        obj = fields(
            value,
            {"episode_id", "step_index", "terminated", "truncated", "success", "frozen"},
            path="robolab_private_status",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "frozen": self.frozen,
        }


@dataclass(frozen=True)
class RoboLabPrivateStatusBatch:
    batch_id: str
    statuses: tuple[RoboLabPrivateStatus, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StrictSchemaError("robolab_private_status_batch.schema_version: expected 1")
        batch_id = sha256(self.batch_id, "robolab_private_status_batch.batch_id")
        statuses = tuple(self.statuses)
        if not statuses or any(not isinstance(item, RoboLabPrivateStatus) for item in statuses):
            raise StrictSchemaError("robolab_private_status_batch.statuses: expected nonempty status sequence")
        ids = [item.episode_id for item in statuses]
        if len(set(ids)) != len(ids):
            raise StrictSchemaError("robolab_private_status_batch.statuses: duplicate episode ids")
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "statuses", statuses)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabPrivateStatusBatch":
        obj = fields(
            value,
            {"schema_version", "batch_id", "statuses"},
            path="robolab_private_status_batch",
        )
        return cls(
            schema_version=integer(obj["schema_version"], "robolab_private_status_batch.schema_version"),
            batch_id=obj["batch_id"],
            statuses=tuple(
                RoboLabPrivateStatus.from_mapping(item)
                for item in sequence(obj["statuses"], "robolab_private_status_batch.statuses")
            ),
        )

    def validate_batch(self, batch: RoboLabBatch) -> None:
        if not isinstance(batch, RoboLabBatch):
            raise StrictSchemaError("robolab_private_status_batch.batch: expected RoboLabBatch")
        if self.batch_id != batch.batch_id or tuple(item.episode_id for item in self.statuses) != batch.episode_ids:
            raise StrictSchemaError("robolab_private_status_batch: batch identity or episode order differs")
        if any(item.step_index > episode.horizon for item, episode in zip(self.statuses, batch.episodes)):
            raise StrictSchemaError("robolab_private_status_batch: step exceeds episode horizon")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "statuses": [item.to_mapping() for item in self.statuses],
        }


def _request_payload(operation: str, value: Any) -> dict[str, Any]:
    if operation == "initialize_app":
        return RoboLabAppConfig.from_mapping(value).to_mapping()
    if operation == "load_batch":
        return RoboLabBatch.from_mapping(value).to_mapping()
    if operation in {"observe_batch", "private_status_batch", "finish_batch"}:
        return _batch_reference(value, f"robolab_request.{operation}")
    if operation == "apply_batch":
        return RoboLabActionBatch.from_mapping(value).to_mapping()
    if operation == "candidate_barrier":
        obj = fields(value, {"barrier_id"}, path="robolab_request.candidate_barrier")
        return {"barrier_id": string(obj["barrier_id"], "robolab_request.candidate_barrier.barrier_id")}
    if operation == "close":
        fields(value, set(), path="robolab_request.close")
        return {}
    raise StrictSchemaError("robolab_request.operation: unknown operation")


def _ack(value: Any, operation: str, flag: str) -> dict[str, Any]:
    obj = fields(value, {"batch_id", "episode_ids", flag}, path=f"robolab_response.{operation}")
    if boolean(obj[flag], f"robolab_response.{operation}.{flag}") is not True:
        raise StrictSchemaError(f"robolab_response.{operation}.{flag}: expected true")
    return {
        "batch_id": sha256(obj["batch_id"], f"robolab_response.{operation}.batch_id"),
        "episode_ids": list(_episode_ids(obj["episode_ids"], f"robolab_response.{operation}.episode_ids")),
        flag: True,
    }


def _response_result(operation: str, value: Any) -> dict[str, Any]:
    if operation == "initialize_app":
        obj = fields(value, {"ready"}, path="robolab_response.initialize_app")
        if boolean(obj["ready"], "robolab_response.initialize_app.ready") is not True:
            raise StrictSchemaError("robolab_response.initialize_app.ready: expected true")
        return {"ready": True}
    if operation == "load_batch":
        return _ack(value, operation, "loaded")
    if operation == "observe_batch":
        return RoboLabObservationBatch.from_mapping(value).to_mapping()
    if operation == "apply_batch":
        return _ack(value, operation, "applied")
    if operation == "private_status_batch":
        return RoboLabPrivateStatusBatch.from_mapping(value).to_mapping()
    if operation == "finish_batch":
        return _ack(value, operation, "finished")
    if operation == "candidate_barrier":
        obj = fields(value, {"barrier_id", "ready"}, path="robolab_response.candidate_barrier")
        if boolean(obj["ready"], "robolab_response.candidate_barrier.ready") is not True:
            raise StrictSchemaError("robolab_response.candidate_barrier.ready: expected true")
        return {
            "barrier_id": string(obj["barrier_id"], "robolab_response.candidate_barrier.barrier_id"),
            "ready": True,
        }
    if operation == "close":
        obj = fields(value, {"closed"}, path="robolab_response.close")
        if boolean(obj["closed"], "robolab_response.close.closed") is not True:
            raise StrictSchemaError("robolab_response.close.closed: expected true")
        return {"closed": True}
    raise StrictSchemaError("robolab_response.operation: unknown operation")


@dataclass(frozen=True)
class RoboLabRpcRequest:
    sequence: int
    operation: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        sequence_number = integer(self.sequence, "robolab_request.sequence", minimum=1)
        operation = enum(self.operation, ROBOLAB_RPC_OPERATIONS, "robolab_request.operation")
        payload = _request_payload(operation, self.payload)
        object.__setattr__(self, "sequence", sequence_number)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabRpcRequest":
        obj = fields(value, {"sequence", "operation", "payload"}, path="robolab_request")
        return cls(sequence=obj["sequence"], operation=obj["operation"], payload=obj["payload"])

    def to_mapping(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "operation": self.operation, "payload": dict(self.payload)}


@dataclass(frozen=True)
class RoboLabRpcResponse:
    sequence: int
    operation: str
    ok: bool
    result: Mapping[str, Any] | None
    error: str | None

    def __post_init__(self) -> None:
        sequence_number = integer(self.sequence, "robolab_response.sequence", minimum=1)
        operation = enum(self.operation, ROBOLAB_RPC_OPERATIONS, "robolab_response.operation")
        ok = boolean(self.ok, "robolab_response.ok")
        if ok:
            if self.error is not None:
                raise StrictSchemaError("robolab_response.error: expected null")
            result = _response_result(operation, self.result)
            error = None
        else:
            if self.result is not None:
                raise StrictSchemaError("robolab_response.result: expected null")
            result = None
            error = string(self.error, "robolab_response.error")
        object.__setattr__(self, "sequence", sequence_number)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "error", error)

    @classmethod
    def from_mapping(cls, value: Any) -> "RoboLabRpcResponse":
        obj = fields(value, {"sequence", "operation", "ok", "result", "error"}, path="robolab_response")
        return cls(
            sequence=obj["sequence"],
            operation=obj["operation"],
            ok=obj["ok"],
            result=obj["result"],
            error=obj["error"],
        )

    @classmethod
    def success(cls, request: RoboLabRpcRequest, result: Mapping[str, Any]) -> "RoboLabRpcResponse":
        if not isinstance(request, RoboLabRpcRequest):
            raise StrictSchemaError("robolab_response.request: expected RoboLabRpcRequest")
        return cls(request.sequence, request.operation, True, result, None)

    @classmethod
    def failure(cls, request: RoboLabRpcRequest, error: str) -> "RoboLabRpcResponse":
        if not isinstance(request, RoboLabRpcRequest):
            raise StrictSchemaError("robolab_response.request: expected RoboLabRpcRequest")
        return cls(request.sequence, request.operation, False, None, error)

    def validate_request(self, request: RoboLabRpcRequest) -> None:
        if not isinstance(request, RoboLabRpcRequest):
            raise StrictSchemaError("robolab_response.request: expected RoboLabRpcRequest")
        if self.sequence != request.sequence or self.operation != request.operation:
            raise StrictSchemaError("robolab_response: request identity differs")
        if not self.ok:
            return
        result = self.result
        if self.operation == "load_batch":
            batch = RoboLabBatch.from_mapping(request.payload)
            if result["batch_id"] != batch.batch_id or tuple(result["episode_ids"]) != batch.episode_ids:
                raise StrictSchemaError("robolab_response.load_batch: loaded batch differs")
        elif self.operation in {"observe_batch", "private_status_batch", "finish_batch"}:
            if result["batch_id"] != request.payload["batch_id"]:
                raise StrictSchemaError(f"robolab_response.{self.operation}: batch differs")
        elif self.operation == "apply_batch":
            actions = RoboLabActionBatch.from_mapping(request.payload)
            if result["batch_id"] != actions.batch_id or tuple(result["episode_ids"]) != tuple(
                item.session_id for item in actions.actions
            ):
                raise StrictSchemaError("robolab_response.apply_batch: applied batch differs")
        elif self.operation == "candidate_barrier" and result["barrier_id"] != request.payload["barrier_id"]:
            raise StrictSchemaError("robolab_response.candidate_barrier: barrier differs")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "operation": self.operation,
            "ok": self.ok,
            "result": None if self.result is None else dict(self.result),
            "error": self.error,
        }


def read_robolab_request(fd: int, timeout_s: float | None = None) -> RoboLabRpcRequest:
    return RoboLabRpcRequest.from_mapping(read_frame(fd, timeout_s))


def write_robolab_request(fd: int, request: RoboLabRpcRequest) -> None:
    if not isinstance(request, RoboLabRpcRequest):
        raise StrictSchemaError("robolab_request: expected RoboLabRpcRequest")
    write_frame(fd, request.to_mapping())


def read_robolab_response(fd: int, timeout_s: float | None = None) -> RoboLabRpcResponse:
    return RoboLabRpcResponse.from_mapping(read_frame(fd, timeout_s))


def write_robolab_response(fd: int, response: RoboLabRpcResponse) -> None:
    if not isinstance(response, RoboLabRpcResponse):
        raise StrictSchemaError("robolab_response: expected RoboLabRpcResponse")
    write_frame(fd, response.to_mapping())
