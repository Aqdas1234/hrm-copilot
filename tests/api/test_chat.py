"""Chat endpoints: the real agent stack on the seeded database, with a scripted model."""

from collections.abc import Iterator
from datetime import date, timedelta

import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from hrm.agents.factory import AgentFactory
from hrm.api.app import create_app
from hrm.api.hr_routes import AuthConfig
from hrm.api.rate_limit import RateLimiter
from hrm.core.config import Settings
from hrm.db.models import LeaveRequest
from hrm.integrations.payroll import PayrollClient
from tests.agents.helpers import FakePolicies
from tests.api.helpers import SECRET, login
from tests.api.test_app import FakeService
from tests.fakes import ScriptedChatModel, StubHTTP, tool_call, wants_tools


@pytest.fixture
def llm() -> ScriptedChatModel:
    return ScriptedChatModel(script=[])


def chat_app(
    factory: sessionmaker[Session], llm: ScriptedChatModel, per_minute: int = 100
) -> TestClient:
    agents = AgentFactory(
        settings=Settings(_env_file=None),
        llm=llm,
        policies=FakePolicies(),
        http=StubHTTP(),
        payroll=PayrollClient(StubHTTP(), base_url="http://payroll.test"),
        checkpointer=InMemorySaver(),
    )
    app = create_app(
        FakeService(),
        session_factory=factory,
        auth=AuthConfig(SECRET, ttl_minutes=60),
        agents=agents,
        chat_limiter=RateLimiter(per_minute),
    )
    return TestClient(app)


@pytest.fixture
def chat(factory: sessionmaker[Session], llm: ScriptedChatModel) -> Iterator[TestClient]:
    with chat_app(factory, llm) as client:
        yield client


def leave_count(factory: sessionmaker[Session]) -> int:
    with factory() as s:
        return s.scalar(select(func.count()).select_from(LeaveRequest)) or 0


def free_dates() -> dict[str, str]:
    """A Monday-Tuesday six weeks out: no overlap with the seeded requests."""
    start = date.today() + timedelta(days=42)
    start += timedelta(days=-start.weekday())
    return {
        "leave_type": "pto",
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=1)).isoformat(),
    }


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/chat/1/messages", {"message": "hi"}),
        ("POST", "/chat/1/confirm", {"approved": True}),
        ("GET", "/chat/1", None),
    ],
)
def test_chat_needs_a_login(chat: TestClient, method: str, path: str, body: object) -> None:
    assert chat.request(method, path, json=body).status_code == 401


def test_a_message_gets_an_answer_and_is_kept(chat: TestClient, llm: ScriptedChatModel) -> None:
    alice = login(chat, "alice.park")
    llm.script.append(AIMessage(content="Hello Alice!"))

    reply = chat.post("/chat/c1/messages", json={"message": "hi"}, headers=alice)
    history = chat.get("/chat/c1", headers=alice).json()

    assert reply.status_code == 200
    assert reply.json() == {"kind": "answer", "text": "Hello Alice!", "actions": []}
    assert history["messages"] == [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "Hello Alice!"},
    ]
    assert history["pending"] is None


def test_an_action_waits_for_confirmation_then_runs(
    chat: TestClient, llm: ScriptedChatModel, factory: sessionmaker[Session]
) -> None:
    alice, before = login(chat, "alice.park"), leave_count(factory)
    llm.script += [
        wants_tools(tool_call("request_leave", **free_dates())),
        AIMessage(content="Filed."),
    ]

    asked = chat.post("/chat/c1/messages", json={"message": "book it"}, headers=alice).json()
    waiting = chat.get("/chat/c1", headers=alice).json()["pending"]
    assert asked["kind"] == "confirm"
    assert waiting == asked
    assert leave_count(factory) == before

    done = chat.post("/chat/c1/confirm", json={"approved": True}, headers=alice)

    assert done.json()["text"] == "Filed."
    assert leave_count(factory) == before + 1
    assert chat.get("/chat/c1", headers=alice).json()["pending"] is None


def test_confirming_with_nothing_waiting_is_a_conflict(chat: TestClient) -> None:
    response = chat.post(
        "/chat/c1/confirm", json={"approved": True}, headers=login(chat, "alice.park")
    )
    assert response.status_code == 409


def test_users_cannot_read_each_others_conversations(
    chat: TestClient, llm: ScriptedChatModel
) -> None:
    llm.script.append(AIMessage(content="noted"))
    chat.post("/chat/c1/messages", json={"message": "secret"}, headers=login(chat, "alice.park"))

    bobs_view = chat.get("/chat/c1", headers=login(chat, "bob.chen")).json()

    assert bobs_view["messages"] == []


@pytest.mark.parametrize("bad_id", ["a.b", "x" * 65, "user-1:c1"])
def test_unsafe_conversation_ids_are_rejected(chat: TestClient, bad_id: str) -> None:
    assert chat.get(f"/chat/{bad_id}", headers=login(chat, "alice.park")).status_code == 422


def test_oversized_bodies_are_rejected(chat: TestClient) -> None:
    body = {"message": "x" * 8_001}
    assert (
        chat.post("/chat/c1/messages", json=body, headers=login(chat, "alice.park")).status_code
        == 422
    )


def test_too_many_messages_are_refused_with_retry_after(
    factory: sessionmaker[Session], llm: ScriptedChatModel
) -> None:
    llm.script.append(AIMessage(content="ok"))
    with chat_app(factory, llm, per_minute=1) as client:
        alice = login(client, "alice.park")
        assert client.post("/chat/c1/messages", json={"message": "a"}, headers=alice).is_success
        refused = client.post("/chat/c1/messages", json={"message": "b"}, headers=alice)

    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "60"


def test_a_model_outage_is_a_clean_502(chat: TestClient, llm: ScriptedChatModel) -> None:
    llm.script.append(groq.APIConnectionError(request=httpx.Request("POST", "https://groq.test")))

    response = chat.post(
        "/chat/c1/messages", json={"message": "hi"}, headers=login(chat, "alice.park")
    )

    assert response.status_code == 502
    assert "groq.test" not in response.text


def test_chat_is_503_when_the_assistant_is_not_configured(client: TestClient) -> None:
    response = client.post(
        "/chat/c1/messages", json={"message": "hi"}, headers=login(client, "alice.park")
    )
    assert response.status_code == 503
