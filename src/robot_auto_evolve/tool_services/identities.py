from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from robot_auto_evolve.services import ServiceIdentity


TOOL_MODEL_STATUSES = frozenset(
    {
        "backend_implemented",
        "gpu_smoke_verified",
        "unavailable_pending_source_test",
    }
)
REASON_REQUIRED_TOOL_STATUSES = frozenset({"unavailable_pending_source_test"})


@dataclass(frozen=True)
class ToolModelSpec:
    service_name: str
    capability: str
    model_id: str
    checkpoint_revision: str
    status: str = "backend_implemented"
    reason: str | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in TOOL_MODEL_STATUSES:
            raise ValueError(f"unsupported tool model status: {self.status}")
        if (self.status in REASON_REQUIRED_TOOL_STATUSES) != (self.reason is not None):
            raise ValueError("guarded tool status requires one reason")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("tool model reason must be nonempty")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint SHA-256 must be lowercase hexadecimal")


MODEL_SPECS = {
    "qwen_language": ToolModelSpec(
        "qwen-language",
        "language",
        "Qwen/Qwen2.5-32B-Instruct",
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        "gpu_smoke_verified",
    ),
    "openai_language": ToolModelSpec(
        "openai-compatible-language",
        "language",
        "configured-openai-compatible-model",
        "configured-server-revision",
    ),
    "qwen_vision": ToolModelSpec(
        "qwen-vision",
        "vision",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "gpu_smoke_verified",
    ),
    "molmo2_vision": ToolModelSpec(
        "molmo2-vision",
        "vision",
        "allenai/Molmo2-8B",
        "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b",
        "gpu_smoke_verified",
    ),
    "molmo2_pointing": ToolModelSpec(
        "molmo2-pointing",
        "pointing",
        "allenai/Molmo2-8B",
        "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b",
        "gpu_smoke_verified",
    ),
    "grounding_dino": ToolModelSpec(
        "grounding-dino",
        "detection",
        "IDEA-Research/grounding-dino-base",
        "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "gpu_smoke_verified",
    ),
    "sam3": ToolModelSpec(
        "sam3",
        "segmentation",
        "AEmotionStudio/sam3",
        "5eac5d508135b2f19adc3ef095efb7d393236f75",
        "gpu_smoke_verified",
        checkpoint_sha256="127037a7a11169c63a210b8e3e9caad24a66abdd65d3a78bbc3a7d8577d57026",
    ),
    "robopoint": ToolModelSpec(
        "robopoint",
        "pointing",
        "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
        "508ed28687caeeb923e1d7d5905e37dccc16c991",
        "unavailable_pending_source_test",
        "The official RoboPoint source adapter is implemented but has not passed its pinned GPU smoke.",
    ),
    "graspgen": ToolModelSpec(
        "graspgen",
        "grasp",
        "adithyamurali/GraspGenModels",
        "ec1ccbb5eec0680db669246ac312a3636f16ee43",
        "gpu_smoke_verified",
    ),
}


def config_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def identity_for(
    name: str,
    *,
    gpu_id: int,
    replica_id: str,
    runtime_config: Mapping[str, Any],
    model_id: str | None = None,
    checkpoint_revision: str | None = None,
) -> ServiceIdentity:
    spec = MODEL_SPECS[name]
    return ServiceIdentity(
        service_name=spec.service_name,
        service_kind=spec.capability,
        service_version="1",
        protocol_version=1,
        model_id=model_id or spec.model_id,
        checkpoint_revision=checkpoint_revision or spec.checkpoint_revision,
        config_sha256=config_hash(runtime_config),
        stateful=False,
        replica_id=replica_id,
        gpu_ids=(gpu_id,),
    )
