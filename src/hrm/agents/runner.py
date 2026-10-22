"""One user message in, one reply out: everything the API needs from the agent.

Keeps LangGraph details (thread config, interrupts, Command) out of the API layer.
A reply is either an answer, or a request to confirm actions that change data.

The confirmation text is built here from the exact tool arguments, not taken from the
model: the user approves what will actually run, whatever the model wrote around it.
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel

from hrm.agents.graph import AgentState
from hrm.core.errors import HRMError

# One line per tool that changes data; a test checks every write tool has one.
ACTION_SUMMARIES = {
    "request_leave": "Request {leave_type} leave from {start_date} to {end_date}",
    "cancel_my_leave_request": "Cancel your leave request #{request_id}",
    "submit_expense_claim": (
        "Submit an expense claim: {description}, {amount} {currency}, bought {purchase_date}"
    ),
    "decide_leave_request": "{decision} leave request #{request_id}",
    "decide_expense_claim": "{decision} expense claim #{claim_id}",
}


class NothingToConfirmError(HRMError):
    """resume() was called but no action is waiting for confirmation."""


class PendingAction(BaseModel):
    tool: str
    args: dict[str, Any]
    summary: str


class AgentReply(BaseModel):
    kind: Literal["answer", "confirm"]
    text: str
    actions: list[PendingAction] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class AgentRunner:
    def __init__(self, agent: CompiledStateGraph[AgentState]) -> None:
        self._agent = agent

    def ask(self, thread_id: str, message: str) -> AgentReply:
        if self._is_waiting(thread_id):
            # The user wrote something new instead of answering: treat it as "no",
            # so an old action can never be approved by a later, unrelated message.
            self._run(Command(resume={"approved": False}), thread_id)
        return self._run({"messages": [HumanMessage(content=message)]}, thread_id)

    def resume(self, thread_id: str, *, approved: bool) -> AgentReply:
        if not self._is_waiting(thread_id):
            raise NothingToConfirmError("There is nothing waiting for confirmation.")
        return self._run(Command(resume={"approved": approved}), thread_id)

    def history(self, thread_id: str) -> list[ChatMessage]:
        """What the user saw: their messages and the assistant's answers, no tool traffic."""
        messages = self._agent.get_state(_config(thread_id)).values.get("messages", [])
        return [
            ChatMessage(role="user" if isinstance(m, HumanMessage) else "assistant", text=m.text)
            for m in messages
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
        ]

    def pending(self, thread_id: str) -> AgentReply | None:
        """The confirmation this conversation is waiting for, if any."""
        return _confirmation(self._agent.get_state(_config(thread_id)))

    def _run(self, payload: AgentState | Command[Any], thread_id: str) -> AgentReply:
        config = _config(thread_id)
        self._agent.invoke(payload, config)
        state = self._agent.get_state(config)
        return _confirmation(state) or AgentReply(
            kind="answer", text=state.values["messages"][-1].text
        )

    def _is_waiting(self, thread_id: str) -> bool:
        return bool(self._agent.get_state(_config(thread_id)).interrupts)


def _confirmation(state: StateSnapshot) -> AgentReply | None:
    if not state.interrupts:
        return None
    actions = [_pending(a) for a in state.interrupts[0].value["actions"]]
    lines = "\n".join(f"- {a.summary}" for a in actions)
    return AgentReply(kind="confirm", text=f"Please confirm:\n{lines}", actions=actions)


def describe_action(tool: str, args: dict[str, Any]) -> str:
    template = ACTION_SUMMARIES.get(tool)
    try:
        summary = template.format(**args) if template else None
    except KeyError:  # the model left out an argument; the tool itself will reject it
        summary = None
    if summary is None:
        return f"Run {tool} with {args}"
    summary = summary[0].upper() + summary[1:]  # "approve ..." -> "Approve ..."; keeps "EUR"
    extra = args.get("reason") or args.get("note")
    return f"{summary} ({extra})" if extra else summary


def _pending(action: dict[str, Any]) -> PendingAction:
    return PendingAction(
        tool=action["tool"],
        args=action["args"],
        summary=describe_action(action["tool"], action["args"]),
    )


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}
