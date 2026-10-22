"""The agent with the real Groq model and the real policy index (skipped by default).

The offline tests script the model's decisions, so they cannot notice when the provider or
the model starts behaving differently. These two scenarios can, in ~30 s and ~10k tokens:

    pytest -m network tests/agents/test_agent_live.py
"""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hrm.agents.factory import AgentFactory
from hrm.agents.graph import AgentState
from hrm.agents.runner import AgentRunner
from hrm.core.config import get_settings
from hrm.core.llm import get_fast_model
from hrm.db.models import LeaveRequest
from hrm.evaluation.agent import all_messages
from hrm.integrations.payroll import PayrollClient
from hrm.rag.service import RAGService
from hrm.rag.vectorstore import ChromaStore
from tests.agents.helpers import TODAY, user_ctx
from tests.fakes import StubHTTP

pytestmark = pytest.mark.network

Parts = tuple[BaseChatModel, RAGService]
Graph = CompiledStateGraph[AgentState]
THREAD: RunnableConfig = {"configurable": {"thread_id": "live"}}


@pytest.fixture(scope="module")
def factory_parts() -> Parts:
    settings = get_settings()
    if not settings.groq_api_key:
        pytest.skip("HRM_GROQ_API_KEY is not set")
    if ChromaStore.from_settings(settings).count() == 0:
        pytest.skip("the policy index is empty; run `hrm ingest`")
    return get_fast_model(settings), RAGService.from_settings(settings)


def alice(db: Session, factory_parts: Parts) -> tuple[AgentRunner, Graph]:
    llm, policies = factory_parts
    factory = AgentFactory(
        settings=get_settings(),
        llm=llm,
        policies=policies,
        http=StubHTTP(),
        payroll=PayrollClient(StubHTTP(), base_url="http://payroll.test"),
        checkpointer=InMemorySaver(),
    )
    graph = factory.graph(db, user_ctx(db, "alice.park"), TODAY)
    return AgentRunner(graph), graph


def requested_tools(graph: Graph) -> list[str]:
    snapshot = graph.get_state(THREAD, subgraphs=True)
    return [c["name"] for m in all_messages(snapshot) for c in getattr(m, "tool_calls", [])]


def test_a_policy_question_is_answered_from_the_policies(db: Session, factory_parts: Parts) -> None:
    runner, graph = alice(db, factory_parts)
    reply = runner.ask("live", "How many business days of notice do I give before PTO?")

    assert reply.kind == "answer"
    assert "search_hr_policies" in requested_tools(graph)
    assert "5" in reply.text


def test_booking_leave_waits_for_confirmation_then_files_it(
    db: Session, factory_parts: Parts
) -> None:
    runner, _ = alice(db, factory_parts)
    before = db.scalar(select(func.count()).select_from(LeaveRequest)) or 0

    asked = runner.ask("live", "Please book PTO from 2026-11-23 to 2026-11-24 for a family visit.")

    assert asked.kind == "confirm"
    action = asked.actions[0]
    assert action.tool == "request_leave"
    assert (action.args["start_date"], action.args["end_date"]) == ("2026-11-23", "2026-11-24")
    assert db.scalar(select(func.count()).select_from(LeaveRequest)) == before

    done = runner.resume("live", approved=True)

    assert done.kind == "answer"
    assert db.scalar(select(func.count()).select_from(LeaveRequest)) == before + 1
