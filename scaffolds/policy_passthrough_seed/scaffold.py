from __future__ import annotations

from robot_auto_evolve.agent import AgentRequest, ToolboxProtocol, VLARequest
from robot_auto_evolve.protocol import CanonicalActionChunk


# The bare-policy seed. It REQUIRES only the frozen policy (`vla`) and declares every other
# tool as OPTIONAL so they are wired and callable -- the coding agent can add perception,
# grounding, or language planning if (and only if) that raises the score. By default this seed
# uses none of them: it feeds the frozen policy its ORIGINAL task instruction, unchanged, one
# action per step. It is the neutral starting point for evolution (contrast scaffolds/
# volo_harness_seed, which ships a designed plan-and-perceive loop that rewrites the instruction
# into a sub-goal every step -- useful as a strong prior, but off-distribution for policies
# trained on the benchmark's own instruction phrasing).
SCAFFOLD_CONFIG = {
    "required_capabilities": ("vla",),
    "optional_capabilities": ("language", "vision", "detection", "segmentation", "pointing", "grasp"),
}


class PolicyPassthrough:
    def reset(self, session_id: str) -> None:
        return None

    def act(self, request: AgentRequest, tools: ToolboxProtocol) -> CanonicalActionChunk:
        action = tools.vla(
            VLARequest(
                request_id=request.request_id,
                session_id=request.session_id,
                observation=request.observation,
                instruction=request.observation.instruction,
                context=(),
                refresh=request.observation.step_index == 0,
            )
        )
        if action.horizon != 1 or action.execution_count != 1:
            raise RuntimeError("native policy service must return exactly one executable action per step")
        return action


def create_scaffold() -> PolicyPassthrough:
    return PolicyPassthrough()
