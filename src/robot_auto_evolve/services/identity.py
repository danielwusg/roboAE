from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_auto_evolve.protocol.schema import StrictSchemaError, boolean, fields, integer, sequence, sha256, string


class ServiceIdentityMismatch(StrictSchemaError):
    pass


@dataclass(frozen=True)
class ServiceIdentity:
    service_name: str
    service_kind: str
    service_version: str
    protocol_version: int
    model_id: str
    checkpoint_revision: str
    config_sha256: str
    stateful: bool
    replica_id: str
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_name", string(self.service_name, "identity.service_name"))
        object.__setattr__(self, "service_kind", string(self.service_kind, "identity.service_kind"))
        object.__setattr__(self, "service_version", string(self.service_version, "identity.service_version"))
        object.__setattr__(
            self, "protocol_version", integer(self.protocol_version, "identity.protocol_version", minimum=1)
        )
        object.__setattr__(self, "model_id", string(self.model_id, "identity.model_id"))
        object.__setattr__(
            self, "checkpoint_revision", string(self.checkpoint_revision, "identity.checkpoint_revision")
        )
        object.__setattr__(self, "config_sha256", sha256(self.config_sha256, "identity.config_sha256"))
        object.__setattr__(self, "stateful", boolean(self.stateful, "identity.stateful"))
        object.__setattr__(self, "replica_id", string(self.replica_id, "identity.replica_id"))
        if isinstance(self.gpu_ids, (str, bytes)):
            raise StrictSchemaError("identity.gpu_ids: expected sequence")
        gpu_ids = tuple(integer(item, f"identity.gpu_ids[{index}]", minimum=0) for index, item in enumerate(self.gpu_ids))
        if tuple(sorted(set(gpu_ids))) != gpu_ids:
            raise StrictSchemaError("identity.gpu_ids: expected sorted unique values")
        object.__setattr__(self, "gpu_ids", gpu_ids)

    @classmethod
    def from_mapping(cls, value: Any) -> "ServiceIdentity":
        obj = fields(
            value,
            {
                "service_name",
                "service_kind",
                "service_version",
                "protocol_version",
                "model_id",
                "checkpoint_revision",
                "config_sha256",
                "stateful",
                "replica_id",
                "gpu_ids",
            },
            path="identity",
        )
        return cls(**{**obj, "gpu_ids": tuple(sequence(obj["gpu_ids"], "identity.gpu_ids"))})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "service_kind": self.service_kind,
            "service_version": self.service_version,
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "checkpoint_revision": self.checkpoint_revision,
            "config_sha256": self.config_sha256,
            "stateful": self.stateful,
            "replica_id": self.replica_id,
            "gpu_ids": list(self.gpu_ids),
        }

    @property
    def gpu_id(self) -> int:
        if len(self.gpu_ids) != 1:
            raise StrictSchemaError("identity.gpu_id: identity does not own exactly one GPU")
        return self.gpu_ids[0]

    def validate_exact(self, actual: "ServiceIdentity") -> None:
        if not isinstance(actual, ServiceIdentity):
            raise ServiceIdentityMismatch("identity: expected ServiceIdentity")
        expected_map = self.to_mapping()
        actual_map = actual.to_mapping()
        differences = {
            key: {"expected": expected_map[key], "actual": actual_map[key]}
            for key in expected_map
            if expected_map[key] != actual_map[key]
        }
        if differences:
            detail = ", ".join(
                f"{key}={item['actual']!r} expected {item['expected']!r}"
                for key, item in sorted(differences.items())
            )
            raise ServiceIdentityMismatch(f"identity mismatch: {detail}")

    def same_model_as(self, other: "ServiceIdentity") -> bool:
        keys = (
            "service_name",
            "service_kind",
            "service_version",
            "protocol_version",
            "model_id",
            "checkpoint_revision",
            "config_sha256",
            "stateful",
        )
        return all(getattr(self, key) == getattr(other, key) for key in keys)
