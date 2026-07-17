from __future__ import annotations

from robot_auto_evolve.agent import AgentRequest, ToolboxProtocol, VLARequest
from robot_auto_evolve.protocol import CanonicalActionChunk


SCAFFOLD_CONFIG = {
    "required_capabilities": ("vla",),
    "optional_capabilities": (),
}


class VlaPassthrough:
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
            raise RuntimeError("native policy service must return one executable action per step")
        return action


def create_scaffold() -> VlaPassthrough:
    return VlaPassthrough()
