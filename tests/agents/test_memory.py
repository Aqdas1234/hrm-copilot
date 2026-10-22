"""Conversation memory: ownership, persistence, history trimming, and the factory."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hrm.agents.factory import AgentFactory
from hrm.agents.graph import recent_history
from hrm.agents.memory import InvalidConversationError, open_memory, thread_id
from hrm.agents.runner import AgentRunner
from hrm.core.config import Settings
from hrm.db.models import LeaveRequest
from hrm.integrations.payroll import PayrollClient
from tests.agents.helpers import TODAY, FakePolicies, user_ctx
from tests.fakes import ScriptedChatModel, StubHTTP, tool_call, wants_tools


def factory(
    llm: ScriptedChatModel, saver: BaseCheckpointSaver[str], **settings: Any
) -> AgentFactory:
    return AgentFactory(
        settings=Settings(_env_file=None, **settings),
        llm=llm,
        policies=FakePolicies(),
        http=StubHTTP(),
        payroll=PayrollClient(StubHTTP(), base_url="http://payroll.test"),
        checkpointer=saver,
    )


def runner(
    db: Session, name: str, llm: ScriptedChatModel, saver: BaseCheckpointSaver[str]
) -> AgentRunner:
    return factory(llm, saver).runner(db, user_ctx(db, name), TODAY)


# --- ownership -----------------------------------------------------------------------------


def test_the_same_conversation_id_is_a_different_thread_per_user(db: Session) -> None:
    alice, bob = user_ctx(db, "alice.park"), user_ctx(db, "bob.chen")
    assert thread_id(alice, "1") != thread_id(bob, "1")


@pytest.mark.parametrize("bad", ["", "a b", "../x", "user-1:2", "x" * 65, "é"])
def test_unsafe_conversation_ids_are_rejected(db: Session, bad: str) -> None:
    with pytest.raises(InvalidConversationError):
        thread_id(user_ctx(db, "alice.park"), bad)


def test_users_never_see_each_others_conversations(db: Session) -> None:
    saver = InMemorySaver()
    alice_llm = ScriptedChatModel(script=[AIMessage(content="noted")])
    runner(db, "alice.park", alice_llm, saver).ask(
        thread_id(user_ctx(db, "alice.park"), "1"), "My secret plan is to resign"
    )

    bob_llm = ScriptedChatModel(script=[AIMessage(content="hi")])
    runner(db, "bob.chen", bob_llm, saver).ask(thread_id(user_ctx(db, "bob.chen"), "1"), "hello")

    seen_by_bob = " ".join(m.text for m in bob_llm.seen[0])
    assert "resign" not in seen_by_bob


# --- memory --------------------------------------------------------------------------------


def test_follow_up_questions_see_the_earlier_turns(db: Session) -> None:
    saver = InMemorySaver()
    llm = ScriptedChatModel(
        script=[AIMessage(content="PTO needs 5 business days' notice."), AIMessage(content="Yes.")]
    )
    agent = runner(db, "alice.park", llm, saver)
    thread = thread_id(user_ctx(db, "alice.park"), "1")
    agent.ask(thread, "How much notice for PTO?")
    agent.ask(thread, "Does that include weekends?")

    texts = [m.text for m in llm.seen[1]]
    assert "How much notice for PTO?" in texts
    assert "PTO needs 5 business days' notice." in texts


def test_a_waiting_confirmation_survives_a_restart(db: Session, tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    thread = thread_id(user_ctx(db, "alice.park"), "1")
    pto = {"leave_type": "pto", "start_date": "2026-11-10", "end_date": "2026-11-11"}
    with open_memory(path) as saver:
        ask = ScriptedChatModel(script=[wants_tools(tool_call("request_leave", **pto))])
        assert runner(db, "alice.park", ask, saver).ask(thread, "Book it").kind == "confirm"

    before = db.scalar(select(func.count()).select_from(LeaveRequest))
    with open_memory(path) as saver:  # a new process: only the file is shared
        done = ScriptedChatModel(script=[AIMessage(content="Filed.")])
        reply = runner(db, "alice.park", done, saver).resume(thread, approved=True)

    assert reply.text == "Filed."
    assert db.scalar(select(func.count()).select_from(LeaveRequest)) == (before or 0) + 1


def test_the_factory_builds_the_prompt_from_the_employee_record(db: Session) -> None:
    llm = ScriptedChatModel(script=[AIMessage(content="hi")])
    runner(db, "hira.khan", llm, InMemorySaver()).ask("t", "hello")
    assert "Hira Khan" in llm.seen[0][0].text


def test_router_mode_routes_before_answering(db: Session) -> None:
    llm = ScriptedChatModel(
        script=[
            wants_tools(tool_call("RouteDecision", specialist="policy")),
            AIMessage(content="hi"),
        ]
    )
    runner = factory(llm, InMemorySaver(), agent_mode="router").runner(
        db, user_ctx(db, "alice.park"), TODAY
    )
    assert runner.ask("t", "hello").text == "hi"
    assert len(llm.seen) == 2  # one routing call, one specialist call


# --- history trimming ----------------------------------------------------------------------


def _turn(n: int, words: int = 50) -> list[AnyMessage]:
    call = tool_call("get_my_profile")
    return [
        HumanMessage(content=f"question {n} " + "word " * words),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="data " * words, tool_call_id=call["id"]),
        AIMessage(content=f"answer {n}"),
    ]


def test_short_histories_are_sent_whole() -> None:
    messages = [*_turn(1), *_turn(2), HumanMessage(content="now?")]
    assert recent_history(messages, 10_000) == messages


def test_old_turns_are_dropped_and_history_starts_at_a_user_message() -> None:
    messages = [*_turn(1), *_turn(2), *_turn(3), HumanMessage(content="now?")]
    sent = recent_history(messages, 300)

    assert sent[-1].text == "now?"
    assert isinstance(sent[0], HumanMessage)
    assert len(sent) < len(messages)
    assert "question 3" in sent[0].text  # the most recent turn survives


def test_the_current_turn_is_always_sent_even_over_budget() -> None:
    current = _turn(9, words=2_000)
    sent = recent_history([*_turn(1), *current], 100)
    assert sent == current
