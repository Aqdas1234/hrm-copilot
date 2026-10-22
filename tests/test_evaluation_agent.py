"""Agent evaluation: the scoring rules, and the dataset's consistency with the real tools."""

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.orm import Session

from hrm.agents.graph import build_agent
from hrm.agents.runner import AgentRunner
from hrm.core.config import PROJECT_ROOT
from hrm.db.seed import seed
from hrm.evaluation.agent import (
    AgentCase,
    CaseOutcome,
    ExpectedAction,
    agent_metrics,
    all_messages,
    load_cases,
    same_value,
    score_case,
)
from tests.agents.helpers import TODAY, tools_for
from tests.fakes import ScriptedChatModel, tool_call, wants_tools

DATASET = PROJECT_ROOT / "evals" / "datasets" / "agent_cases.jsonl"


def case(**fields: object) -> AgentCase:
    return AgentCase.model_validate(
        {"id": "x", "kind": "tools", "user": "alice.park", "message": "m", **fields}
    )


def outcome(**fields: object) -> CaseOutcome:
    return CaseOutcome.model_validate(
        {"requested_tools": [], "action": None, "output": "", "blocked": False, **fields}
    )


def test_expected_tools_must_all_be_requested() -> None:
    c = case(expect_tools=["a", "b"])
    assert score_case(c, outcome(requested_tools=["b", "a", "c"])).passed
    assert score_case(c, outcome(requested_tools=["a"])).reasons == ["missing tools: ['b']"]


def test_any_of_and_forbidden_tools() -> None:
    c = case(expect_any_tool=["a", "b"], forbid_tools=["w"])
    assert score_case(c, outcome(requested_tools=["b"])).passed
    assert not score_case(c, outcome(requested_tools=["c"])).passed
    assert not score_case(c, outcome(requested_tools=["a", "w"])).passed


def test_actions_check_tool_and_listed_arguments_only() -> None:
    c = case(kind="actions", expect_action={"tool": "pay", "args": {"amount": "89.50"}})
    right = ExpectedAction(tool="pay", args={"amount": "89.5", "note": "anything"})
    assert score_case(c, outcome(action=right)).passed
    assert not score_case(c, outcome(action=right.model_copy(update={"tool": "other"}))).passed
    assert score_case(c, outcome()).reasons == ["no confirmation asked (expected pay)"]


def test_reveals_state_and_errors_fail_a_case() -> None:
    c = case(kind="attacks")
    assert not score_case(c, outcome(revealed=["salary:bob.chen"])).passed
    assert not score_case(c, outcome(failed_state=["leave:2:pending"])).passed
    assert not score_case(c, outcome(error="RateLimitError")).passed


def test_values_compare_as_text_or_numbers() -> None:
    assert same_value("89.5", "89.50")
    assert same_value(3, "3")
    assert same_value("PTO ", "pto")
    assert not same_value("2026-11-23", "2026-11-24")


def test_metrics_per_kind() -> None:
    cases = [case(id="1"), case(id="2"), case(id="3", kind="attacks")]
    outcomes = [outcome(tokens=100), outcome(tokens=300), outcome(blocked=True)]
    scores = [score_case(c, o) for c, o in zip(cases, outcomes, strict=True)]
    scores[1] = scores[1].model_copy(update={"passed": False})

    m = agent_metrics(cases, outcomes, scores)

    assert m.pass_rate == {"tools": 0.5, "attacks": 1.0}
    assert m.blocked_by_guard == 1
    assert round(m.tokens_per_case) == 133


def test_the_dataset_names_only_real_tools_and_users(session: Session) -> None:
    seed(session, today=TODAY, demo_password="x")
    cases = load_cases(DATASET)
    all_tools = set(tools_for(session, "carol.reyes"))
    for c in cases:
        named = {*c.expect_tools, *c.expect_any_tool, *c.forbid_tools}
        if c.expect_action:
            named.add(c.expect_action.tool)
        assert named <= all_tools, (c.id, named - all_tools)
        assert tools_for(session, c.user), c.id
    assert {c.kind for c in cases} == {"tools", "actions", "attacks"}


def test_a_proposed_action_waiting_for_confirmation_is_visible(session: Session) -> None:
    seed(session, today=TODAY, demo_password="x")
    llm = ScriptedChatModel(
        script=[wants_tools(tool_call("cancel_my_leave_request", request_id=2))]
    )
    graph = build_agent(
        llm, list(tools_for(session, "alice.park").values()), "p", checkpointer=InMemorySaver()
    )
    assert AgentRunner(graph).ask("t", "cancel it").kind == "confirm"
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}

    def requested(messages: list[AnyMessage]) -> list[str]:
        return [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]

    # the paused subgraph holds the call; the parent graph does not have it yet
    assert requested(graph.get_state(config).values["messages"]) == []
    assert requested(all_messages(graph.get_state(config, subgraphs=True))) == [
        "cancel_my_leave_request"
    ]
