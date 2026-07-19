from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from robot_auto_evolve.agent import AgentEvent
from robot_auto_evolve.protocol import CanonicalActionChunk, FairObservation, StrictSchemaError


def _mapping(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise StrictSchemaError(f"{path}: invalid fields")
    return value


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
