"""Router mode: one small routing call picks a specialist, which handles the message.

    START ─▶ guard_in ─▶ route ─▶ policy | leave | pay | people | general ─▶ guard_out ─▶ END

Decisions:
- Routing is a structured-output call (one field, a fixed set of names), not free text:
  the answer is always a valid branch.
- The router sees recent user/assistant text only, not tool traffic: enough to resolve
  follow-ups ("and next week?") at a few hundred tokens.
- If routing fails (API error, odd output), the "general" specialist with all of the
  user's tools takes the message: router mode is never less capable than single mode.
- Each specialist is its own node (a subgraph), so traces and logs show who answered.
- Guardrails, confirmation and limits are the same code as the single agent (graph.py).
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from hrm.agents.graph import (
    DEFAULT_LIMITS,
    AgentLimits,
    AgentState,
    agent_loop,
    guarded_graph,
    recent_history,
    require_checkpointer,
)
from hrm.agents.specialists import FALLBACK, SPECIALISTS, Specialist, SpecialistName
from hrm.core.logging import get_logger

log = get_logger(__name__)

ROUTER_HISTORY_TOKENS = 600


class RouteDecision(BaseModel):
    """Which specialist should handle the user's latest message."""

    specialist: SpecialistName = Field(description="The best specialist for the message.")


def build_router_agent(
    llm: BaseChatModel,
    router_llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    limits: AgentLimits = DEFAULT_LIMITS,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    specialists: Sequence[Specialist] = SPECIALISTS,
) -> CompiledStateGraph[AgentState]:
    require_checkpointer(tools, checkpointer)
    decide = router_llm.with_structured_output(RouteDecision)
    router_prompt = _router_prompt(specialists)
    names: list[str] = [s.name for s in specialists]

    def route(state: AgentState) -> AgentState:
        try:
            decision = decide.invoke([SystemMessage(content=router_prompt), *_dialogue(state)])
            choice = decision.specialist if isinstance(decision, RouteDecision) else FALLBACK
        except Exception as exc:  # any routing failure: the generalist can still answer
            log.warning("agent.route_failed", error=type(exc).__name__)
            choice = FALLBACK
        log.info("agent.route", specialist=choice)
        return {"specialist": choice if choice in names else FALLBACK}

    graph = guarded_graph(first="route")
    graph.add_node("route", route)
    for specialist in specialists:
        own = _tools_for(specialist, tools)
        prompt = f"{system_prompt}\n{specialist.focus}"
        graph.add_node(specialist.name, agent_loop(llm, own, prompt, limits))
        graph.add_edge(specialist.name, "guard_out")
    graph.add_conditional_edges("route", lambda state: state.get("specialist") or FALLBACK, names)
    return graph.compile(checkpointer=checkpointer)


def _tools_for(specialist: Specialist, tools: Sequence[BaseTool]) -> list[BaseTool]:
    if specialist.tools is None:
        return list(tools)
    return [t for t in tools if t.name in specialist.tools]


def _dialogue(state: AgentState) -> list[AnyMessage]:
    """Recent user questions and assistant answers, without tool calls and results."""
    recent = recent_history(state["messages"], ROUTER_HISTORY_TOKENS)
    return [
        m
        for m in recent
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]


def _router_prompt(specialists: Sequence[Specialist]) -> str:
    options = "\n".join(f"- {s.name}: {s.description}" for s in specialists)
    return (
        "You route messages in an HR assistant. Pick the one specialist that should handle "
        "the user's latest message, using the conversation for context.\n\n"
        f"Specialists:\n{options}"
    )
