"""Human-in-the-loop: data changes only after the user confirms.

Each runner() call builds a fresh graph and fresh tools, like a new HTTP request; only
the checkpointer (the saved conversation) is shared between them.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hrm.agents.graph import DECLINED_REPLY, build_agent
from hrm.agents.prompts import system_prompt
from hrm.agents.runner import (
    ACTION_SUMMARIES,
    AgentRunner,
    NothingToConfirmError,
    describe_action,
)
from hrm.agents.tools import is_write
from hrm.core.errors import ConfigurationError
from hrm.db.models import AuditLog, LeaveRequest, Role
from tests.agents.helpers import TODAY, tools_for
from tests.fakes import ScriptedChatModel, tool_call, wants_tools

THREAD = "alice:1"
PTO = {"leave_type": "pto", "start_date": "2026-11-10", "end_date": "2026-11-11"}


@pytest.fixture
def saver() -> InMemorySaver:
    return InMemorySaver()


def runner(db: Session, saver: InMemorySaver, llm: ScriptedChatModel) -> AgentRunner:
    tools = list(tools_for(db, "alice.park").values())
    prompt = system_prompt(name="Alice Park", role=Role.EMPLOYEE, today=TODAY)
    return AgentRunner(build_agent(llm, tools, prompt, checkpointer=saver))


def leave_requests(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(LeaveRequest)) or 0


def booking_script(*after_confirm: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(
        script=[
            wants_tools(tool_call("check_leave_request", **PTO)),
            wants_tools(tool_call("request_leave", **PTO, reason="Trip")),
            *after_confirm,
        ]
    )


def test_a_write_pauses_and_changes_nothing_until_confirmed(
    db: Session, saver: InMemorySaver
) -> None:
    before = leave_requests(db)
    reply = runner(db, saver, booking_script()).ask(THREAD, "Book PTO Nov 10-11 for a trip")

    assert reply.kind == "confirm"
    assert [a.tool for a in reply.actions] == ["request_leave"]
    assert reply.actions[0].summary == "Request pto leave from 2026-11-10 to 2026-11-11 (Trip)"
    assert leave_requests(db) == before


def test_approving_runs_the_tool_in_a_later_request(db: Session, saver: InMemorySaver) -> None:
    before = leave_requests(db)
    runner(db, saver, booking_script()).ask(THREAD, "Book PTO Nov 10-11")

    done = AIMessage(content="Filed: 2 working days, waiting for Bob.")
    reply = runner(db, saver, ScriptedChatModel(script=[done])).resume(THREAD, approved=True)

    assert reply.kind == "answer"
    assert reply.text == done.text
    assert leave_requests(db) == before + 1
    assert db.scalar(select(AuditLog).where(AuditLog.action == "leave.request")) is not None


def test_declining_changes_nothing_and_needs_no_model_call(
    db: Session, saver: InMemorySaver
) -> None:
    before = leave_requests(db)
    runner(db, saver, booking_script()).ask(THREAD, "Book PTO Nov 10-11")

    llm = ScriptedChatModel(script=[])
    reply = runner(db, saver, llm).resume(THREAD, approved=False)

    assert reply.kind == "answer"
    assert reply.text == DECLINED_REPLY
    assert llm.seen == []
    assert leave_requests(db) == before


def test_anything_but_an_explicit_yes_is_a_no(db: Session, saver: InMemorySaver) -> None:
    runner(db, saver, booking_script()).ask(THREAD, "Book PTO Nov 10-11")
    tools = list(tools_for(db, "alice.park").values())
    agent = build_agent(ScriptedChatModel(script=[]), tools, "prompt", checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": THREAD}}

    result = agent.invoke(Command(resume="yes"), config)  # not {"approved": True}

    assert result["messages"][-1].text == DECLINED_REPLY


def test_a_new_message_cancels_the_waiting_action(db: Session, saver: InMemorySaver) -> None:
    before = leave_requests(db)
    runner(db, saver, booking_script()).ask(THREAD, "Book PTO Nov 10-11")

    llm = ScriptedChatModel(script=[AIMessage(content="Sure, what dates instead?")])
    reply = runner(db, saver, llm).ask(THREAD, "Actually, different dates")

    assert reply.kind == "answer"
    assert leave_requests(db) == before
    history = llm.seen[0]
    assert any(isinstance(m, ToolMessage) and "declined" in m.text for m in history)


def test_read_only_tools_never_ask_for_confirmation(db: Session, saver: InMemorySaver) -> None:
    llm = ScriptedChatModel(
        script=[wants_tools(tool_call("get_my_profile")), AIMessage(content="You're Alice.")]
    )
    reply = runner(db, saver, llm).ask(THREAD, "Who am I?")

    assert reply.kind == "answer"
    assert reply.actions == []


def test_resume_without_a_waiting_action_is_an_error(db: Session, saver: InMemorySaver) -> None:
    with pytest.raises(NothingToConfirmError):
        runner(db, saver, ScriptedChatModel(script=[])).resume(THREAD, approved=True)


def test_write_tools_without_a_checkpointer_fail_at_build_time(db: Session) -> None:
    tools = list(tools_for(db, "alice.park").values())
    with pytest.raises(ConfigurationError):
        build_agent(ScriptedChatModel(script=[]), tools, "prompt")


def test_every_write_tool_has_a_confirmation_summary(db: Session) -> None:
    writes = {t.name for t in tools_for(db, "bob.chen").values() if is_write(t)}
    assert writes == set(ACTION_SUMMARIES)


def test_summaries_show_the_exact_arguments() -> None:
    assert describe_action("decide_leave_request", {"request_id": 7, "decision": "approve"}) == (
        "Approve leave request #7"
    )
    claim = {
        "description": "External Monitor",
        "amount": "249.99",
        "currency": "EUR",
        "purchase_date": "2026-10-20",
    }
    assert describe_action("submit_expense_claim", claim) == (
        "Submit an expense claim: External Monitor, 249.99 EUR, bought 2026-10-20"
    )


def test_summaries_fall_back_when_an_argument_is_missing() -> None:
    assert describe_action("request_leave", {"leave_type": "pto"}) == (
        "Run request_leave with {'leave_type': 'pto'}"
    )


def test_a_confirmed_action_is_kept_even_if_the_reply_fails(
    db: Session, saver: InMemorySaver
) -> None:
    before = leave_requests(db)
    runner(db, saver, booking_script()).ask(THREAD, "Book PTO Nov 10-11")

    broken = ScriptedChatModel(script=[])  # the model call after the tool fails
    with pytest.raises(AssertionError):
        runner(db, saver, broken).resume(THREAD, approved=True)
    db.rollback()  # what the API does with the failed request

    # the conversation says the tool ran, and the database agrees
    assert leave_requests(db) == before + 1
