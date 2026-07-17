from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from robot_auto_evolve.agent import AgentEvent
from robot_auto_evolve.evaluation import EpisodeOutcome
from robot_auto_evolve.protocol import CanonicalActionChunk, FairObservation, StrictSchemaError, decode_message


TERMINATIONS = frozenset({"success", "horizon", "infrastructure_error"})
BUNDLE_TERMINATIONS = frozenset({"success", "horizon"})
EVIDENCE_SCHEMA_VERSION = 2
MAX_EPISODES = 256
MAX_SAMPLED_STEPS_PER_EPISODE = 8
MAX_EVENTS_PER_SAMPLED_STEP = 16
MAX_EVENT_EXCERPTS_PER_EPISODE = 8
MAX_IDENTIFIER_BYTES = 256
MAX_TEXT_BYTES = 1024
MAX_ACTION_SAMPLE_ROWS = 4
MAX_PROPRIO_VALUES = 64
MAX_FRAME_CANDIDATES = 64
MAX_FRAME_FILES = 32
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_TOTAL_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 2 * 1024 * 1024
MAX_BUNDLE_BYTES = 20 * 1024 * 1024


def evidence_limits() -> dict[str, int]:
    return {
        "max_episodes": MAX_EPISODES,
        "max_sampled_steps_per_episode": MAX_SAMPLED_STEPS_PER_EPISODE,
        "max_events_per_sampled_step": MAX_EVENTS_PER_SAMPLED_STEP,
        "max_event_excerpts_per_episode": MAX_EVENT_EXCERPTS_PER_EPISODE,
        "max_identifier_bytes": MAX_IDENTIFIER_BYTES,
        "max_text_bytes": MAX_TEXT_BYTES,
        "max_action_sample_rows": MAX_ACTION_SAMPLE_ROWS,
        "max_proprio_values": MAX_PROPRIO_VALUES,
        "max_frame_candidates": MAX_FRAME_CANDIDATES,
        "max_frame_files": MAX_FRAME_FILES,
        "max_frame_bytes": MAX_FRAME_BYTES,
        "max_frame_total_bytes": MAX_FRAME_TOTAL_BYTES,
        "max_index_bytes": MAX_INDEX_BYTES,
        "max_bundle_bytes": MAX_BUNDLE_BYTES,
    }


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _strict_json(payload: bytes, path: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise StrictSchemaError(f"{path}: duplicate or invalid JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(StrictSchemaError(f"{path}: invalid constant {value}")),
        )
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"{path}: invalid JSON: {exc}") from exc


def _mapping(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise StrictSchemaError(f"{path}: invalid fields")
    return value


def _integer(value: Any, path: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise StrictSchemaError(f"{path}: invalid integer")
    return value


def _number(value: Any, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise StrictSchemaError(f"{path}: invalid number")
    return float(value)


def _identifier(value: Any, path: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value) or len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise StrictSchemaError(f"{path}: invalid bounded string")
    return value


def _sha256(value: Any, path: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StrictSchemaError(f"{path}: invalid sha256")
    return value


def _bounded_prefix(value: str, maximum: int) -> str:
    payload = value.encode("utf-8")
    if len(payload) <= maximum:
        return value
    return payload[:maximum].decode("utf-8", errors="ignore")


def _text_record(value: str) -> dict[str, Any]:
    if type(value) is not str:
        raise StrictSchemaError("public evidence text must be a string")
    payload = value.encode("utf-8")
    text = _bounded_prefix(value, MAX_TEXT_BYTES)
    return {
        "text": text,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "utf8_bytes": len(payload),
        "truncated": len(payload) > len(text.encode("utf-8")),
    }


def _validate_text_record(value: Any, path: str) -> None:
    obj = _mapping(value, {"text", "sha256", "utf8_bytes", "truncated"}, path)
    if type(obj["text"]) is not str or len(obj["text"].encode("utf-8")) > MAX_TEXT_BYTES:
        raise StrictSchemaError(f"{path}.text: exceeds byte limit")
    _sha256(obj["sha256"], f"{path}.sha256")
    size = _integer(obj["utf8_bytes"], f"{path}.utf8_bytes")
    if type(obj["truncated"]) is not bool:
        raise StrictSchemaError(f"{path}.truncated: expected bool")
    encoded = obj["text"].encode("utf-8")
    if len(encoded) > size or obj["truncated"] != (len(encoded) < size):
        raise StrictSchemaError(f"{path}: inconsistent text length")
    if not obj["truncated"] and hashlib.sha256(encoded).hexdigest() != obj["sha256"]:
        raise StrictSchemaError(f"{path}: text hash mismatch")


@dataclass(frozen=True)
class PublicStepEvidence:
    observation: FairObservation
    action: CanonicalActionChunk | None
    events: tuple[AgentEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation, FairObservation):
            raise StrictSchemaError("public_step.observation: expected FairObservation")
        if self.action is not None:
            if not isinstance(self.action, CanonicalActionChunk):
                raise StrictSchemaError("public_step.action: expected CanonicalActionChunk or null")
            if self.action.start_step != self.observation.step_index:
                raise StrictSchemaError("public_step.action: start step mismatch")
        events = tuple(self.events)
        if any(not isinstance(event, AgentEvent) for event in events):
            raise StrictSchemaError("public_step.events: expected AgentEvent entries")
        if any(event.step_index != self.observation.step_index for event in events):
            raise StrictSchemaError("public_step.events: step mismatch")
        object.__setattr__(self, "events", events)

    @classmethod
    def from_mapping(cls, value: Any) -> "PublicStepEvidence":
        obj = _mapping(value, {"observation", "action", "events"}, "public_step")
        if not isinstance(obj["events"], list):
            raise StrictSchemaError("public_step.events: expected list")
        return cls(
            observation=FairObservation.from_mapping(obj["observation"]),
            action=None if obj["action"] is None else CanonicalActionChunk.from_mapping(obj["action"]),
            events=tuple(AgentEvent.from_mapping(event) for event in obj["events"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_mapping(),
            "action": None if self.action is None else self.action.to_mapping(),
            "events": [event.to_mapping() for event in self.events],
        }


@dataclass(frozen=True)
class PublicEpisodeEvidence:
    outcome: EpisodeOutcome
    termination: str
    steps: tuple[PublicStepEvidence, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EpisodeOutcome) or self.outcome.key.split != "evolve":
            raise StrictSchemaError("public_episode.outcome: expected evolve EpisodeOutcome")
        if self.termination not in TERMINATIONS:
            raise StrictSchemaError("public_episode.termination: unsupported value")
        steps = tuple(self.steps)
        if not steps or any(not isinstance(step, PublicStepEvidence) for step in steps):
            raise StrictSchemaError("public_episode.steps: expected nonempty PublicStepEvidence entries")
        indices = [step.observation.step_index for step in steps]
        if indices != sorted(set(indices)):
            raise StrictSchemaError("public_episode.steps: expected sorted unique indices")
        episode_id = self.outcome.key.artifact_id()
        if any(step.observation.episode_id != episode_id for step in steps):
            raise StrictSchemaError("public_episode.steps: episode identity mismatch")
        instructions = {step.observation.instruction for step in steps}
        if len(instructions) != 1 or not next(iter(instructions)):
            raise StrictSchemaError("public_episode.steps: instruction changed or is empty")
        if self.outcome.success and self.termination != "success":
            raise StrictSchemaError("public_episode.termination: successful outcome requires success")
        if not self.outcome.success and self.termination == "success":
            raise StrictSchemaError("public_episode.termination: failed outcome cannot terminate with success")
        if self.termination == "infrastructure_error" and not self.error:
            raise StrictSchemaError("public_episode.error: required for infrastructure error")
        if self.termination != "infrastructure_error" and self.error is not None:
            raise StrictSchemaError("public_episode.error: expected null")
        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_mapping(cls, value: Any) -> "PublicEpisodeEvidence":
        obj = _mapping(value, {"outcome", "termination", "steps", "error"}, "public_episode")
        if not isinstance(obj["steps"], list):
            raise StrictSchemaError("public_episode.steps: expected list")
        return cls(
            outcome=EpisodeOutcome.from_mapping(obj["outcome"]),
            termination=obj["termination"],
            steps=tuple(PublicStepEvidence.from_mapping(step) for step in obj["steps"]),
            error=obj["error"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.to_mapping(),
            "termination": self.termination,
            "steps": [step.to_mapping() for step in self.steps],
            "error": self.error,
        }


@dataclass
class _FrameCandidate:
    rank: str
    outcome: EpisodeOutcome
    step_index: int
    camera_name: str
    rgb_sha256: str
    camera_record: dict[str, Any]


def _sample_indices(length: int) -> tuple[int, ...]:
    if length <= MAX_SAMPLED_STEPS_PER_EPISODE:
        return tuple(range(length))
    denominator = MAX_SAMPLED_STEPS_PER_EPISODE - 1
    return tuple((index * (length - 1) + denominator // 2) // denominator for index in range(MAX_SAMPLED_STEPS_PER_EPISODE))


def _matrix(value: np.ndarray | None) -> list[list[float]] | None:
    return None if value is None else np.asarray(value, dtype=np.float64).tolist()


def _camera_record(camera: Any) -> dict[str, Any]:
    rgb = camera.rgb
    depth = None
    if camera.depth_m is not None:
        valid = camera.depth_valid
        values = camera.depth_m[valid]
        depth = {
            "shape": list(camera.depth_m.shape),
            "sha256": hashlib.sha256(camera.depth_m.tobytes()).hexdigest(),
            "valid_mask_sha256": hashlib.sha256(valid.tobytes()).hexdigest(),
            "valid_fraction": float(valid.mean()),
            "valid_min": None if values.size == 0 else float(values.min()),
            "valid_max": None if values.size == 0 else float(values.max()),
            "valid_mean": None if values.size == 0 else float(values.mean()),
        }
    return {
        "frame_id": _identifier(camera.frame_id, "camera.frame_id"),
        "optical_convention": _identifier(camera.optical_convention, "camera.optical_convention"),
        "rgb_shape": list(rgb.shape),
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "rgb_mean": [float(value) for value in rgb.mean(axis=(0, 1))],
        "depth": depth,
        "intrinsics": _matrix(camera.intrinsics),
        "camera_to_world": _matrix(camera.camera_to_world),
        "diagnostic_frame": None,
    }


def _proprioception_record(vector: Any) -> dict[str, Any]:
    values = vector.values
    count = int(values.size)
    prefix = values[:MAX_PROPRIO_VALUES]
    spec = vector.spec
    return {
        "name": _identifier(spec.name, "proprioception.name"),
        "quantity": _identifier(spec.quantity, "proprioception.quantity"),
        "frame_id": _identifier(spec.frame_id, "proprioception.frame_id"),
        "reference_frame": _identifier(spec.reference_frame, "proprioception.reference_frame"),
        "representation": _identifier(spec.representation, "proprioception.representation"),
        "quaternion_order": _identifier(spec.quaternion_order, "proprioception.quaternion_order"),
        "component_count": len(spec.component_names),
        "components": [_identifier(item, "proprioception.component") for item in spec.component_names[:MAX_PROPRIO_VALUES]],
        "units": [_identifier(item, "proprioception.unit") for item in spec.units[:MAX_PROPRIO_VALUES]],
        "value_count": count,
        "values": [float(value) for value in prefix],
        "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "truncated": count > MAX_PROPRIO_VALUES,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def _action_record(action: CanonicalActionChunk) -> dict[str, Any]:
    values = action.values
    spec = action.spec.to_mapping()
    return {
        "spec_sha256": hashlib.sha256(_canonical_json(spec)).hexdigest(),
        "channel_names": list(action.spec.channel_names),
        "channel_semantics": list(action.spec.channel_semantics),
        "coordinate_frame": action.spec.coordinate_frame,
        "shape": list(values.shape),
        "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sampled_values": values[:MAX_ACTION_SAMPLE_ROWS].tolist(),
        "sampled_rows": min(values.shape[0], MAX_ACTION_SAMPLE_ROWS),
        "execution_count": action.execution_count,
    }


def _event_record(event: AgentEvent) -> dict[str, Any]:
    return {
        "step_index": event.step_index,
        "event_type": _text_record(event.event_type),
        "status": event.status,
        "detail": _text_record(event.detail),
        "capability": event.capability,
    }


def _bounded_step_events(events: tuple[AgentEvent, ...]) -> tuple[AgentEvent, ...]:
    if len(events) <= MAX_EVENTS_PER_SAMPLED_STEP:
        return events
    return (*events[: MAX_EVENTS_PER_SAMPLED_STEP - 1], events[-1])


def _summarize_episode(episode: PublicEpisodeEvidence) -> tuple[dict[str, Any], list[_FrameCandidate]]:
    if episode.termination not in BUNDLE_TERMINATIONS or episode.error is not None:
        raise StrictSchemaError("public evidence bundle accepts complete robot outcomes only")
    sampled = set(_sample_indices(len(episode.steps)))
    sampled_steps: list[dict[str, Any]] = []
    candidates: list[_FrameCandidate] = []
    event_counts: dict[str, int] = {}
    event_excerpts: list[tuple[str, dict[str, Any]]] = []
    action_spec_sha256: str | None = None
    action_names: list[str] | None = None
    action_rows = 0
    action_sum: np.ndarray | None = None
    action_minimum: np.ndarray | None = None
    action_maximum: np.ndarray | None = None
    n_actions = 0
    for position, step in enumerate(episode.steps):
        for event_index, event in enumerate(step.events):
            count_key = f"{event.status}:{event.capability or 'none'}"
            event_counts[count_key] = event_counts.get(count_key, 0) + 1
            record = _event_record(event)
            rank = hashlib.sha256(
                f"{episode.outcome.key.artifact_id()}\0{position}\0{event_index}\0{event.status}\0{event.capability}".encode()
            ).hexdigest()
            event_excerpts.append((rank, record))
            event_excerpts = sorted(event_excerpts, key=lambda item: item[0])[:MAX_EVENT_EXCERPTS_PER_EPISODE]
        if step.action is not None:
            n_actions += 1
            record = _action_record(step.action)
            if action_spec_sha256 is None:
                action_spec_sha256 = record["spec_sha256"]
                action_names = record["channel_names"]
            elif action_spec_sha256 != record["spec_sha256"]:
                raise StrictSchemaError("public evidence action spec changed within an episode")
            values = step.action.values.astype(np.float64)
            rows = values.shape[0]
            action_rows += rows
            row_sum = values.sum(axis=0)
            row_minimum = values.min(axis=0)
            row_maximum = values.max(axis=0)
            action_sum = row_sum if action_sum is None else action_sum + row_sum
            action_minimum = row_minimum if action_minimum is None else np.minimum(action_minimum, row_minimum)
            action_maximum = row_maximum if action_maximum is None else np.maximum(action_maximum, row_maximum)
        if position not in sampled:
            continue
        camera_records: dict[str, Any] = {}
        for camera_name, camera in sorted(step.observation.cameras.items()):
            _identifier(camera_name, "camera.name")
            camera_summary = _camera_record(camera)
            camera_records[camera_name] = camera_summary
            rank = hashlib.sha256(
                f"{episode.outcome.key.artifact_id()}\0{step.observation.step_index}\0{camera_name}\0{camera_summary['rgb_sha256']}".encode()
            ).hexdigest()
            candidates.append(
                _FrameCandidate(
                    rank,
                    episode.outcome,
                    step.observation.step_index,
                    camera_name,
                    camera_summary["rgb_sha256"],
                    camera_summary,
                )
            )
        sampled_steps.append(
            {
                "step_index": step.observation.step_index,
                "cameras": camera_records,
                "proprioception": [_proprioception_record(vector) for vector in step.observation.proprioception.vectors],
                "action": None if step.action is None else _action_record(step.action),
                "event_count": len(step.events),
                "events": [_event_record(event) for event in _bounded_step_events(step.events)],
            }
        )
    action_statistics = None
    if action_rows:
        if action_sum is None or action_minimum is None or action_maximum is None or action_names is None:
            raise RuntimeError("public evidence action statistics are incomplete")
        action_statistics = {
            "spec_sha256": action_spec_sha256,
            "channel_names": action_names,
            "rows": action_rows,
            "minimum": action_minimum.tolist(),
            "maximum": action_maximum.tolist(),
            "mean": (action_sum / action_rows).tolist(),
        }
    first = episode.steps[0]
    summary = {
        "outcome": episode.outcome.to_mapping(),
        "termination": episode.termination,
        "error": None if episode.error is None else _text_record(episode.error),
        "instruction": _text_record(first.observation.instruction),
        "n_observations": len(episode.steps),
        "n_actions": n_actions,
        "sampled_steps": sampled_steps,
        "event_counts": dict(sorted(event_counts.items())),
        "event_excerpts": [record for _, record in event_excerpts],
        "action_statistics": action_statistics,
    }
    return summary, candidates


def _png(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", compress_level=6, optimize=False)
    return output.getvalue()


def _descriptor(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _validate_descriptor(value: Any, path: str) -> dict[str, Any]:
    obj = dict(_mapping(value, {"path", "sha256", "size_bytes"}, path))
    relative = Path(_identifier(obj["path"], f"{path}.path"))
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != obj["path"]:
        raise StrictSchemaError(f"{path}.path: unsafe relative path")
    _sha256(obj["sha256"], f"{path}.sha256")
    _integer(obj["size_bytes"], f"{path}.size_bytes", maximum=MAX_FRAME_BYTES)
    return obj


def _validate_matrix(value: Any, shape: tuple[int, int], path: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise StrictSchemaError(f"{path}: invalid matrix")
    for row in value:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise StrictSchemaError(f"{path}: invalid matrix row")
        for item in row:
            _number(item, path)


def _validate_event(value: Any, path: str) -> None:
    obj = _mapping(value, {"step_index", "event_type", "status", "detail", "capability"}, path)
    _integer(obj["step_index"], f"{path}.step_index")
    _validate_text_record(obj["event_type"], f"{path}.event_type")
    if obj["status"] not in {"started", "ok", "optional_error", "infrastructure_error", "triggered", "skipped"}:
        raise StrictSchemaError(f"{path}.status: unsupported")
    _validate_text_record(obj["detail"], f"{path}.detail")
    if obj["capability"] is not None:
        _identifier(obj["capability"], f"{path}.capability")


def _validate_camera(
    value: Any,
    frames: Mapping[str, Mapping[str, Any]],
    decoded_frames: Mapping[str, tuple[tuple[int, ...], str]] | None,
    path: str,
) -> None:
    obj = _mapping(
        value,
        {
            "frame_id",
            "optical_convention",
            "rgb_shape",
            "rgb_sha256",
            "rgb_mean",
            "depth",
            "intrinsics",
            "camera_to_world",
            "diagnostic_frame",
        },
        path,
    )
    _identifier(obj["frame_id"], f"{path}.frame_id")
    _identifier(obj["optical_convention"], f"{path}.optical_convention")
    if not isinstance(obj["rgb_shape"], list) or len(obj["rgb_shape"]) != 3 or obj["rgb_shape"][2] != 3:
        raise StrictSchemaError(f"{path}.rgb_shape: invalid")
    for size in obj["rgb_shape"]:
        _integer(size, f"{path}.rgb_shape", minimum=1)
    _sha256(obj["rgb_sha256"], f"{path}.rgb_sha256")
    if not isinstance(obj["rgb_mean"], list) or len(obj["rgb_mean"]) != 3:
        raise StrictSchemaError(f"{path}.rgb_mean: invalid")
    for item in obj["rgb_mean"]:
        if not 0.0 <= _number(item, f"{path}.rgb_mean") <= 255.0:
            raise StrictSchemaError(f"{path}.rgb_mean: out of range")
    if obj["depth"] is not None:
        depth = _mapping(
            obj["depth"],
            {"shape", "sha256", "valid_mask_sha256", "valid_fraction", "valid_min", "valid_max", "valid_mean"},
            f"{path}.depth",
        )
        if not isinstance(depth["shape"], list) or len(depth["shape"]) != 2:
            raise StrictSchemaError(f"{path}.depth.shape: invalid")
        for size in depth["shape"]:
            _integer(size, f"{path}.depth.shape", minimum=1)
        _sha256(depth["sha256"], f"{path}.depth.sha256")
        _sha256(depth["valid_mask_sha256"], f"{path}.depth.valid_mask_sha256")
        if not 0.0 <= _number(depth["valid_fraction"], f"{path}.depth.valid_fraction") <= 1.0:
            raise StrictSchemaError(f"{path}.depth.valid_fraction: out of range")
        for name in ("valid_min", "valid_max", "valid_mean"):
            if depth[name] is not None:
                _number(depth[name], f"{path}.depth.{name}")
    _validate_matrix(obj["intrinsics"], (3, 3), f"{path}.intrinsics")
    _validate_matrix(obj["camera_to_world"], (4, 4), f"{path}.camera_to_world")
    if obj["diagnostic_frame"] is not None:
        descriptor = _validate_descriptor(obj["diagnostic_frame"], f"{path}.diagnostic_frame")
        if frames.get(descriptor["path"]) != descriptor:
            raise StrictSchemaError(f"{path}.diagnostic_frame: differs from manifest")
        if decoded_frames is not None and decoded_frames.get(descriptor["path"]) != (
            tuple(obj["rgb_shape"]),
            obj["rgb_sha256"],
        ):
            raise StrictSchemaError(f"{path}.diagnostic_frame: PNG pixels differ from RGB summary")


def _validate_proprioception(value: Any, path: str) -> None:
    obj = _mapping(
        value,
        {
            "name",
            "quantity",
            "frame_id",
            "reference_frame",
            "representation",
            "quaternion_order",
            "component_count",
            "components",
            "units",
            "value_count",
            "values",
            "values_sha256",
            "truncated",
            "minimum",
            "maximum",
            "mean",
        },
        path,
    )
    for name in ("name", "quantity", "frame_id", "reference_frame", "representation", "quaternion_order"):
        _identifier(obj[name], f"{path}.{name}")
    count = _integer(obj["component_count"], f"{path}.component_count", minimum=1)
    value_count = _integer(obj["value_count"], f"{path}.value_count", minimum=1)
    if count != value_count:
        raise StrictSchemaError(f"{path}: component and value counts differ")
    for name in ("components", "units", "values"):
        if not isinstance(obj[name], list) or len(obj[name]) != min(count, MAX_PROPRIO_VALUES):
            raise StrictSchemaError(f"{path}.{name}: invalid bounded list")
    for item in obj["components"]:
        _identifier(item, f"{path}.components")
    for item in obj["units"]:
        _identifier(item, f"{path}.units")
    for item in obj["values"]:
        _number(item, f"{path}.values")
    _sha256(obj["values_sha256"], f"{path}.values_sha256")
    if type(obj["truncated"]) is not bool or obj["truncated"] != (count > MAX_PROPRIO_VALUES):
        raise StrictSchemaError(f"{path}.truncated: inconsistent")
    for name in ("minimum", "maximum", "mean"):
        _number(obj[name], f"{path}.{name}")


def _validate_action(value: Any, path: str) -> None:
    obj = _mapping(
        value,
        {
            "spec_sha256",
            "channel_names",
            "channel_semantics",
            "coordinate_frame",
            "shape",
            "values_sha256",
            "sampled_values",
            "sampled_rows",
            "execution_count",
        },
        path,
    )
    _sha256(obj["spec_sha256"], f"{path}.spec_sha256")
    if not isinstance(obj["channel_names"], list) or not obj["channel_names"]:
        raise StrictSchemaError(f"{path}.channel_names: invalid")
    if not isinstance(obj["channel_semantics"], list) or len(obj["channel_semantics"]) != len(obj["channel_names"]):
        raise StrictSchemaError(f"{path}.channel_semantics: invalid")
    for name in (*obj["channel_names"], *obj["channel_semantics"]):
        _identifier(name, path)
    _identifier(obj["coordinate_frame"], f"{path}.coordinate_frame")
    if not isinstance(obj["shape"], list) or len(obj["shape"]) != 2:
        raise StrictSchemaError(f"{path}.shape: invalid")
    rows = _integer(obj["shape"][0], f"{path}.shape.rows", minimum=1)
    width = _integer(obj["shape"][1], f"{path}.shape.width", minimum=1)
    if width != len(obj["channel_names"]):
        raise StrictSchemaError(f"{path}.shape: channel width mismatch")
    sampled_rows = _integer(obj["sampled_rows"], f"{path}.sampled_rows", minimum=1, maximum=MAX_ACTION_SAMPLE_ROWS)
    if sampled_rows != min(rows, MAX_ACTION_SAMPLE_ROWS):
        raise StrictSchemaError(f"{path}.sampled_rows: inconsistent")
    if not isinstance(obj["sampled_values"], list) or len(obj["sampled_values"]) != sampled_rows:
        raise StrictSchemaError(f"{path}.sampled_values: invalid")
    for row in obj["sampled_values"]:
        if not isinstance(row, list) or len(row) != width:
            raise StrictSchemaError(f"{path}.sampled_values: wrong width")
        for item in row:
            _number(item, f"{path}.sampled_values")
    _sha256(obj["values_sha256"], f"{path}.values_sha256")
    _integer(obj["execution_count"], f"{path}.execution_count", minimum=1, maximum=rows)


def _validate_action_statistics(value: Any, path: str) -> None:
    obj = _mapping(value, {"spec_sha256", "channel_names", "rows", "minimum", "maximum", "mean"}, path)
    _sha256(obj["spec_sha256"], f"{path}.spec_sha256")
    if not isinstance(obj["channel_names"], list) or not obj["channel_names"]:
        raise StrictSchemaError(f"{path}.channel_names: invalid")
    for item in obj["channel_names"]:
        _identifier(item, f"{path}.channel_names")
    _integer(obj["rows"], f"{path}.rows", minimum=1)
    for name in ("minimum", "maximum", "mean"):
        if not isinstance(obj[name], list) or len(obj[name]) != len(obj["channel_names"]):
            raise StrictSchemaError(f"{path}.{name}: invalid")
        for item in obj[name]:
            _number(item, f"{path}.{name}")


def _validate_index(
    value: Any,
    frames: Mapping[str, Mapping[str, Any]],
    decoded_frames: Mapping[str, tuple[tuple[int, ...], str]] | None = None,
) -> tuple[EpisodeOutcome, ...]:
    obj = _mapping(value, {"schema_version", "kind", "limits", "n_episodes", "episodes"}, "public_evidence.index")
    if obj["schema_version"] != EVIDENCE_SCHEMA_VERSION or obj["kind"] != "public_evolve_evidence":
        raise StrictSchemaError("public_evidence.index: wrong schema or kind")
    if obj["limits"] != evidence_limits():
        raise StrictSchemaError("public_evidence.index: limits differ")
    if not isinstance(obj["episodes"], list):
        raise StrictSchemaError("public_evidence.index.episodes: expected list")
    count = _integer(obj["n_episodes"], "public_evidence.index.n_episodes", minimum=1, maximum=MAX_EPISODES)
    if len(obj["episodes"]) != count:
        raise StrictSchemaError("public_evidence.index: episode count mismatch")
    outcomes: list[EpisodeOutcome] = []
    referenced_frames: set[str] = set()
    for episode_index, value_episode in enumerate(obj["episodes"]):
        path = f"public_evidence.index.episodes[{episode_index}]"
        episode = _mapping(
            value_episode,
            {
                "outcome",
                "termination",
                "error",
                "instruction",
                "n_observations",
                "n_actions",
                "sampled_steps",
                "event_counts",
                "event_excerpts",
                "action_statistics",
            },
            path,
        )
        outcome = EpisodeOutcome.from_mapping(episode["outcome"])
        if outcome.key.split != "evolve" or outcome.to_mapping() != episode["outcome"]:
            raise StrictSchemaError(f"{path}.outcome: expected canonical evolve outcome")
        if episode["termination"] not in BUNDLE_TERMINATIONS:
            raise StrictSchemaError(f"{path}.termination: invalid")
        if outcome.success != (episode["termination"] == "success"):
            raise StrictSchemaError(f"{path}.termination: inconsistent with outcome")
        if episode["error"] is not None:
            raise StrictSchemaError(f"{path}.error: forbidden")
        _validate_text_record(episode["instruction"], f"{path}.instruction")
        n_observations = _integer(episode["n_observations"], f"{path}.n_observations", minimum=1)
        n_actions = _integer(episode["n_actions"], f"{path}.n_actions", maximum=n_observations)
        if not isinstance(episode["sampled_steps"], list) or not 1 <= len(episode["sampled_steps"]) <= MAX_SAMPLED_STEPS_PER_EPISODE:
            raise StrictSchemaError(f"{path}.sampled_steps: invalid count")
        indices: list[int] = []
        for step_offset, value_step in enumerate(episode["sampled_steps"]):
            step_path = f"{path}.sampled_steps[{step_offset}]"
            step = _mapping(value_step, {"step_index", "cameras", "proprioception", "action", "event_count", "events"}, step_path)
            indices.append(_integer(step["step_index"], f"{step_path}.step_index"))
            if not isinstance(step["cameras"], Mapping) or not step["cameras"]:
                raise StrictSchemaError(f"{step_path}.cameras: invalid")
            for camera_name, camera in step["cameras"].items():
                _identifier(camera_name, f"{step_path}.camera_name")
                _validate_camera(camera, frames, decoded_frames, f"{step_path}.cameras.{camera_name}")
                if camera["diagnostic_frame"] is not None:
                    referenced_frames.add(camera["diagnostic_frame"]["path"])
            if not isinstance(step["proprioception"], list) or not step["proprioception"]:
                raise StrictSchemaError(f"{step_path}.proprioception: invalid")
            for vector_index, vector in enumerate(step["proprioception"]):
                _validate_proprioception(vector, f"{step_path}.proprioception[{vector_index}]")
            if step["action"] is not None:
                _validate_action(step["action"], f"{step_path}.action")
            event_count = _integer(step["event_count"], f"{step_path}.event_count")
            if not isinstance(step["events"], list) or len(step["events"]) > MAX_EVENTS_PER_SAMPLED_STEP or len(step["events"]) > event_count:
                raise StrictSchemaError(f"{step_path}.events: invalid bounded list")
            for event_index, event in enumerate(step["events"]):
                _validate_event(event, f"{step_path}.events[{event_index}]")
        if indices != sorted(set(indices)):
            raise StrictSchemaError(f"{path}.sampled_steps: indices are not sorted and unique")
        if not isinstance(episode["event_counts"], Mapping) or len(episode["event_counts"]) > 64:
            raise StrictSchemaError(f"{path}.event_counts: invalid")
        for key, value_count in episode["event_counts"].items():
            _identifier(key, f"{path}.event_counts key")
            _integer(value_count, f"{path}.event_counts.{key}")
        if not isinstance(episode["event_excerpts"], list) or len(episode["event_excerpts"]) > MAX_EVENT_EXCERPTS_PER_EPISODE:
            raise StrictSchemaError(f"{path}.event_excerpts: invalid")
        for event_index, event in enumerate(episode["event_excerpts"]):
            _validate_event(event, f"{path}.event_excerpts[{event_index}]")
        if episode["action_statistics"] is None:
            if n_actions != 0:
                raise StrictSchemaError(f"{path}.action_statistics: absent with actions")
        else:
            if n_actions == 0:
                raise StrictSchemaError(f"{path}.action_statistics: present without actions")
            _validate_action_statistics(episode["action_statistics"], f"{path}.action_statistics")
        outcomes.append(outcome)
    if outcomes != sorted(outcomes, key=lambda item: item.key) or len({item.key for item in outcomes}) != len(outcomes):
        raise StrictSchemaError("public_evidence.index: outcomes are not sorted and unique")
    if referenced_frames != set(frames):
        raise StrictSchemaError("public_evidence.index: frame references differ from manifest")
    return tuple(outcomes)


class PublicEvolutionEvidence:
    def __init__(self, root: Path, index: Mapping[str, Any], manifest: Mapping[str, Any], outcomes: tuple[EpisodeOutcome, ...]) -> None:
        self.root = Path(root).resolve()
        self.index = dict(index)
        self.manifest = dict(manifest)
        self.outcomes = outcomes

    @property
    def episodes(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.index["episodes"])

    @property
    def bundle_sha256(self) -> str:
        return self.manifest["bundle_sha256"]

    @property
    def payload_bytes(self) -> int:
        return self.manifest["payload_bytes"]

    @classmethod
    def create(
        cls,
        root: Path,
        outcomes: Sequence[EpisodeOutcome],
        loader: Callable[[EpisodeOutcome], PublicEpisodeEvidence],
    ) -> "PublicEvolutionEvidence":
        target = Path(root).resolve()
        checked_outcomes = tuple(sorted(outcomes, key=lambda item: item.key))
        if not checked_outcomes or len(checked_outcomes) > MAX_EPISODES:
            raise StrictSchemaError("public evidence requires 1..256 outcomes")
        if any(not isinstance(item, EpisodeOutcome) or item.key.split != "evolve" for item in checked_outcomes):
            raise StrictSchemaError("public evidence accepts evolve outcomes only")
        if len({item.key for item in checked_outcomes}) != len(checked_outcomes):
            raise StrictSchemaError("public evidence outcomes are duplicated")
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
        staging.mkdir()
        (staging / "frames").mkdir()
        try:
            summaries: list[dict[str, Any]] = []
            candidates: list[_FrameCandidate] = []
            for outcome in checked_outcomes:
                episode = loader(outcome)
                if not isinstance(episode, PublicEpisodeEvidence) or episode.outcome != outcome:
                    raise StrictSchemaError("public evidence loader returned a different outcome")
                summary, episode_candidates = _summarize_episode(episode)
                summaries.append(summary)
                candidates.extend(episode_candidates)
            candidate_groups: dict[tuple[str, bool], list[_FrameCandidate]] = {}
            for candidate in candidates:
                candidate_groups.setdefault(
                    (candidate.outcome.key.task_id, candidate.outcome.success),
                    [],
                ).append(candidate)
            for group in candidate_groups.values():
                group.sort(key=lambda item: (item.rank, item.outcome.key, item.step_index, item.camera_name))
            selected_candidates: list[_FrameCandidate] = []
            group_offset = 0
            group_keys = sorted(candidate_groups)
            while len(selected_candidates) < MAX_FRAME_CANDIDATES:
                added = False
                for group_key in group_keys:
                    group = candidate_groups[group_key]
                    if group_offset < len(group):
                        selected_candidates.append(group[group_offset])
                        added = True
                        if len(selected_candidates) == MAX_FRAME_CANDIDATES:
                            break
                if not added:
                    break
                group_offset += 1
            selection_order = {candidate.rank: index for index, candidate in enumerate(selected_candidates)}
            by_outcome: dict[EpisodeOutcome, list[_FrameCandidate]] = {}
            for candidate in selected_candidates:
                by_outcome.setdefault(candidate.outcome, []).append(candidate)
            encoded: list[tuple[_FrameCandidate, bytes]] = []
            for outcome, outcome_candidates in sorted(by_outcome.items(), key=lambda item: item[0].key):
                episode = loader(outcome)
                if not isinstance(episode, PublicEpisodeEvidence) or episode.outcome != outcome:
                    raise StrictSchemaError("public evidence loader changed between passes")
                steps = {item.observation.step_index: item for item in episode.steps}
                for candidate in outcome_candidates:
                    try:
                        rgb = steps[candidate.step_index].observation.cameras[candidate.camera_name].rgb
                    except KeyError as exc:
                        raise StrictSchemaError("public evidence diagnostic frame disappeared") from exc
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != candidate.rgb_sha256:
                        raise StrictSchemaError("public evidence diagnostic frame changed between passes")
                    payload = _png(rgb)
                    if len(payload) <= MAX_FRAME_BYTES:
                        encoded.append((candidate, payload))
            frame_descriptors: list[dict[str, Any]] = []
            frame_total = 0
            for candidate, payload in sorted(encoded, key=lambda item: selection_order[item[0].rank]):
                if len(frame_descriptors) >= MAX_FRAME_FILES or frame_total + len(payload) > MAX_FRAME_TOTAL_BYTES:
                    break
                relative = f"frames/{candidate.rank[:32]}.png"
                path = staging / relative
                if path.exists():
                    raise RuntimeError("public evidence diagnostic frame path collision")
                path.write_bytes(payload)
                descriptor = _descriptor(relative, payload)
                candidate.camera_record["diagnostic_frame"] = descriptor
                frame_descriptors.append(descriptor)
                frame_total += len(payload)
            index = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "kind": "public_evolve_evidence",
                "limits": evidence_limits(),
                "n_episodes": len(summaries),
                "episodes": summaries,
            }
            _validate_index(index, {item["path"]: item for item in frame_descriptors})
            index_payload = _canonical_json(index)
            if len(index_payload) > MAX_INDEX_BYTES:
                raise StrictSchemaError(f"public evidence index exceeds {MAX_INDEX_BYTES} bytes")
            index_descriptor = _descriptor("index.json", index_payload)
            manifest_without_digest = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "kind": "public_evolve_evidence_bundle",
                "limits": evidence_limits(),
                "index": index_descriptor,
                "frames": sorted(frame_descriptors, key=lambda item: item["path"]),
                "payload_bytes": len(index_payload) + frame_total,
            }
            manifest = {
                **manifest_without_digest,
                "bundle_sha256": hashlib.sha256(_canonical_json(manifest_without_digest)).hexdigest(),
            }
            manifest_payload = _canonical_json(manifest)
            if manifest["payload_bytes"] + len(manifest_payload) > MAX_BUNDLE_BYTES:
                raise StrictSchemaError(f"public evidence bundle exceeds {MAX_BUNDLE_BYTES} bytes")
            (staging / "index.json").write_bytes(index_payload)
            (staging / "manifest.json").write_bytes(manifest_payload)
            for path in staging.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            staging.chmod(0o555)
            os.rename(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls.load(target)

    @classmethod
    def create_from_trace_files(
        cls,
        root: Path,
        records: Sequence[tuple[EpisodeOutcome, Path]],
    ) -> "PublicEvolutionEvidence":
        paths: dict[EpisodeOutcome, Path] = {}
        for outcome, path in records:
            if outcome in paths:
                raise StrictSchemaError("public evidence trace outcome is duplicated")
            paths[outcome] = Path(path).resolve()

        def load(outcome: EpisodeOutcome) -> PublicEpisodeEvidence:
            path = paths[outcome]
            if not path.is_file() or path.is_symlink():
                raise StrictSchemaError("public evidence trace must be a regular file")
            return PublicEpisodeEvidence.from_mapping(decode_message(path.read_bytes()))

        return cls.create(root, tuple(paths), load)

    @classmethod
    def load(cls, root: Path) -> "PublicEvolutionEvidence":
        path = Path(root).resolve()
        if not path.is_dir() or path.is_symlink():
            raise StrictSchemaError("public evidence root must be a regular directory")
        manifest_path = path / "manifest.json"
        index_path = path / "index.json"
        if not manifest_path.is_file() or manifest_path.is_symlink() or not index_path.is_file() or index_path.is_symlink():
            raise StrictSchemaError("public evidence manifest and index must be regular files")
        manifest_payload = manifest_path.read_bytes()
        manifest = _mapping(
            _strict_json(manifest_payload, "public_evidence.manifest"),
            {"schema_version", "kind", "limits", "index", "frames", "payload_bytes", "bundle_sha256"},
            "public_evidence.manifest",
        )
        if _canonical_json(manifest) != manifest_payload:
            raise StrictSchemaError("public evidence manifest is not canonical JSON")
        if manifest["schema_version"] != EVIDENCE_SCHEMA_VERSION or manifest["kind"] != "public_evolve_evidence_bundle":
            raise StrictSchemaError("public evidence manifest has wrong schema or kind")
        if manifest["limits"] != evidence_limits():
            raise StrictSchemaError("public evidence manifest limits differ")
        index_descriptor = _validate_descriptor(manifest["index"], "public_evidence.manifest.index")
        if index_descriptor["path"] != "index.json" or index_descriptor["size_bytes"] > MAX_INDEX_BYTES:
            raise StrictSchemaError("public evidence index descriptor is invalid")
        if not isinstance(manifest["frames"], list) or len(manifest["frames"]) > MAX_FRAME_FILES:
            raise StrictSchemaError("public evidence frame descriptor count is invalid")
        frame_descriptors: dict[str, dict[str, Any]] = {}
        for index, value in enumerate(manifest["frames"]):
            descriptor = _validate_descriptor(value, f"public_evidence.manifest.frames[{index}]")
            if not descriptor["path"].startswith("frames/") or descriptor["path"] in frame_descriptors:
                raise StrictSchemaError("public evidence frame path is invalid or duplicated")
            frame_descriptors[descriptor["path"]] = descriptor
        if list(frame_descriptors) != sorted(frame_descriptors):
            raise StrictSchemaError("public evidence frame descriptors are not sorted")
        expected_files = {"manifest.json", "index.json", *frame_descriptors}
        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        for child in path.rglob("*"):
            if child.is_symlink():
                raise StrictSchemaError("public evidence contains a symlink")
            if child.is_file():
                actual_files.add(child.relative_to(path).as_posix())
            elif child.is_dir():
                actual_directories.add(child.relative_to(path).as_posix())
            else:
                raise StrictSchemaError("public evidence contains a non-regular entry")
        if actual_files != expected_files or actual_directories != {"frames"}:
            raise StrictSchemaError("public evidence contains unexpected or missing files")
        index_payload = index_path.read_bytes()
        if len(index_payload) != index_descriptor["size_bytes"] or hashlib.sha256(index_payload).hexdigest() != index_descriptor["sha256"]:
            raise StrictSchemaError("public evidence index descriptor mismatch")
        if len(index_payload) > MAX_INDEX_BYTES:
            raise StrictSchemaError("public evidence index exceeds byte limit")
        frame_total = 0
        decoded_frames: dict[str, tuple[tuple[int, ...], str]] = {}
        for relative, descriptor in frame_descriptors.items():
            frame_path = path / relative
            payload = frame_path.read_bytes()
            if len(payload) != descriptor["size_bytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
                raise StrictSchemaError("public evidence frame descriptor mismatch")
            if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                raise StrictSchemaError("public evidence diagnostic frame is not PNG")
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.mode != "RGB":
                        raise StrictSchemaError("public evidence diagnostic frame is not RGB")
                    rgb = np.asarray(image, dtype=np.uint8)
            except StrictSchemaError:
                raise
            except Exception as exc:
                raise StrictSchemaError(f"public evidence diagnostic frame is invalid: {exc}") from exc
            if _png(rgb) != payload:
                raise StrictSchemaError("public evidence diagnostic frame is not canonical PNG")
            decoded_frames[relative] = (tuple(rgb.shape), hashlib.sha256(rgb.tobytes()).hexdigest())
            frame_total += len(payload)
        if frame_total > MAX_FRAME_TOTAL_BYTES:
            raise StrictSchemaError("public evidence frames exceed total byte limit")
        payload_bytes = _integer(manifest["payload_bytes"], "public_evidence.manifest.payload_bytes")
        if payload_bytes != len(index_payload) + frame_total:
            raise StrictSchemaError("public evidence payload byte count differs")
        stable_manifest = dict(manifest)
        digest = _sha256(stable_manifest.pop("bundle_sha256"), "public_evidence.manifest.bundle_sha256")
        if hashlib.sha256(_canonical_json(stable_manifest)).hexdigest() != digest:
            raise StrictSchemaError("public evidence bundle digest mismatch")
        if payload_bytes + len(manifest_payload) > MAX_BUNDLE_BYTES:
            raise StrictSchemaError("public evidence bundle exceeds byte limit")
        index = _strict_json(index_payload, "public_evidence.index")
        if _canonical_json(index) != index_payload:
            raise StrictSchemaError("public evidence index is not canonical JSON")
        outcomes = _validate_index(index, frame_descriptors, decoded_frames)
        return cls(path, index, manifest, outcomes)

    def validate_outcomes(self, outcomes: tuple[EpisodeOutcome, ...]) -> None:
        expected = tuple(sorted(outcomes, key=lambda item: item.key))
        if self.outcomes != expected:
            raise StrictSchemaError("public evidence outcomes differ from evolve report")

    def prompt_text(self) -> str:
        payload = _canonical_json(self.index)
        if len(payload) > MAX_INDEX_BYTES:
            raise StrictSchemaError("public evidence prompt index exceeds byte limit")
        return payload.decode("utf-8")
