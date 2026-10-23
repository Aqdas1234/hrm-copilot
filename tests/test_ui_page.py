"""The Streamlit app, run headless with Streamlit's AppTest; the API is a fake backend."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from hrm.agents.runner import AgentReply, ChatMessage, PendingAction
from hrm.api.chat_routes import Conversation, ConversationSummary
from hrm.api.schemas import AskResponse
from hrm.auth.access import EmployeeView, LeaveRequestView
from hrm.db.models import RequestStatus, Role, TicketCategory, TicketStatus
from hrm.ui.client import ApiClient, ApiError, Login
from hrm.workflows.tickets import TicketView
from tests.test_ui_client import ASK_OK

PAGE = Path(__file__).parents[1] / "src" / "hrm" / "ui" / "streamlit_app.py"
BOB = EmployeeView(
    id=2,
    employee_code="E0002",
    name="Bob Chen",
    email="bob.chen@example.com",
    job_title="Engineering Manager",
    department="Engineering",
    country="US",
    manager_name=None,
    start_date=date(2021, 1, 1),
)
ALICE = BOB.model_copy(update={"id": 3, "name": "Alice Park", "job_title": "Software Engineer"})
TICKET = TicketView(
    id=7,
    employee_id=3,
    category=TicketCategory.ACCESS,
    subject="Badge broken",
    body="It stopped working. ![x](https://evil.example/track.png)",
    status=TicketStatus.OPEN,
    resolution=None,
    created_at=datetime(2026, 10, 22, 9, 0),
    updated_at=None,
    closed_at=None,
)
PENDING = AgentReply(
    kind="confirm",
    text="Please confirm",
    actions=[
        PendingAction(
            tool="request_leave",
            args={},
            summary="Request pto leave from 2026-11-16 to 2026-11-17",
        )
    ],
)


class FakeBackend:
    """Stands in for the API; records what the page sent."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.pending: AgentReply | None = None
        self.sent: list[str] = []
        self.confirmed: list[bool] = []
        self.error: ApiError | None = None
        self.ask_reply: AskResponse | ApiError = AskResponse.model_validate(ASK_OK)
        self.saved: list[ConversationSummary] = []
        self.deleted: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = self

        def method(name: str) -> Any:
            def call(_: ApiClient, *args: Any) -> Any:
                if backend.error:
                    raise backend.error
                return getattr(backend, f"do_{name}")(*args)

            return call

        names = ("login", "me", "team", "approvals", "send", "confirm", "conversation", "ask")
        extra = ("my_tickets", "ticket_queue", "all_employees", "conversations")
        for name in (*names, *extra, "delete_conversation"):
            monkeypatch.setattr(ApiClient, name, method(name))

    def do_login(self, email: str, password: str) -> Login:
        if password != "right":
            raise ApiError("Invalid email or password.", status_code=401)
        return Login(token="token", role=Role.MANAGER)

    def do_my_tickets(self) -> list[TicketView]:
        return [TICKET.model_copy(update={"id": 1, "subject": "My own question"})]

    def do_ticket_queue(self) -> list[TicketView]:
        return [TICKET]

    def do_all_employees(self) -> list[EmployeeView]:
        return [ALICE, BOB]

    def do_me(self) -> EmployeeView:
        return BOB

    def do_team(self) -> list[EmployeeView]:
        return [ALICE]

    def do_approvals(self) -> list[LeaveRequestView]:
        return [
            LeaveRequestView(
                id=6,
                employee_id=3,
                leave_type="pto",
                start_date=date(2026, 11, 16),
                end_date=date(2026, 11, 17),
                working_days=2,
                status=RequestStatus.PENDING,
                reason="Family trip",
            )
        ]

    def do_send(self, conversation_id: str, message: str) -> AgentReply:
        self.sent.append(message)
        self.messages += [
            ChatMessage(role="user", text=message),
            ChatMessage(role="assistant", text="It costs $234 and $1,000."),
        ]
        return AgentReply(kind="answer", text="ok")

    def do_confirm(self, conversation_id: str, approved: bool) -> AgentReply:
        self.confirmed.append(approved)
        self.pending = None
        return AgentReply(kind="answer", text="ok")

    def do_conversations(self) -> list[ConversationSummary]:
        return self.saved

    def do_delete_conversation(self, conversation_id: str) -> None:
        self.deleted.append(conversation_id)
        self.saved = [c for c in self.saved if c.conversation_id != conversation_id]

    def do_conversation(self, conversation_id: str) -> Conversation:
        return Conversation(
            conversation_id=conversation_id, messages=self.messages, pending=self.pending
        )

    def do_ask(self, question: str, category: str | None = None) -> AskResponse:
        if isinstance(self.ask_reply, ApiError):
            raise self.ask_reply
        return self.ask_reply


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    fake = FakeBackend()
    fake.install(monkeypatch)
    return fake


def logged_in(page: str = "Assistant", role: Role = Role.MANAGER) -> AppTest:
    app = AppTest.from_file(str(PAGE))
    app.session_state["token"] = "token"
    app.session_state["role"] = role
    app.session_state["me"] = BOB
    app.session_state["page"] = page
    app.run()
    assert not app.exception, app.exception
    return app


def markdown(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown)


# --- login ---------------------------------------------------------------------------------


def test_the_assistant_asks_for_a_login_first(backend: FakeBackend) -> None:
    app = AppTest.from_file(str(PAGE)).run()
    assert [s.value for s in app.subheader] == ["Log in"]


def test_logging_in_shows_the_user(backend: FakeBackend) -> None:
    app = AppTest.from_file(str(PAGE)).run()
    app.text_input[0].input("bob.chen@example.com")
    app.text_input[1].input("right")
    app.button[0].click().run()

    assert not app.exception
    assert "Bob Chen" in " ".join(m.value for m in app.sidebar.markdown)
    assert app.session_state["role"] == Role.MANAGER


def test_a_wrong_password_shows_the_error(backend: FakeBackend) -> None:
    app = AppTest.from_file(str(PAGE)).run()
    app.text_input[0].input("bob.chen@example.com")
    app.text_input[1].input("wrong")
    app.button[0].click().run()

    assert [e.value for e in app.error] == ["Invalid email or password."]
    assert "token" not in app.session_state


def test_an_expired_login_logs_out(backend: FakeBackend) -> None:
    backend.error = ApiError("Token expired.", status_code=401)
    app = logged_in()

    assert "token" not in app.session_state
    assert "expired" in app.warning[0].value


# --- assistant -----------------------------------------------------------------------------


def test_a_message_is_sent_and_the_reply_shown(backend: FakeBackend) -> None:
    app = logged_in()
    app.chat_input[0].set_value("What does the monitor cost?").run()

    assert backend.sent == ["What does the monitor cost?"]
    assert r"It costs \$234 and \$1,000." in markdown(app)  # money, not LaTeX


def test_a_pending_action_can_be_confirmed(backend: FakeBackend) -> None:
    backend.pending = PENDING
    app = logged_in()
    assert "Request pto leave from 2026-11-16 to 2026-11-17" in markdown(app)

    next(b for b in app.button if b.label == "Confirm").click().run()

    assert backend.confirmed == [True]
    assert "Please confirm" not in markdown(app)


def test_a_pending_action_can_be_declined(backend: FakeBackend) -> None:
    backend.pending = PENDING
    app = logged_in()
    next(b for b in app.button if b.label == "Decline").click().run()
    assert backend.confirmed == [False]


SAVED = [
    ConversationSummary(conversation_id="c2", title="Payslip question"),
    ConversationSummary(conversation_id="c1", title="PTO in November"),
]


def test_a_new_conversation_can_be_started(backend: FakeBackend) -> None:
    backend.saved = list(SAVED)
    app = logged_in()
    next(b for b in app.sidebar.button if b.label == "New conversation").click().run()

    assert app.session_state["conversation"] not in {"c1", "c2"}
    assert app.sidebar.selectbox[0].options == ["New conversation", *[c.title for c in SAVED]]


def test_saved_conversations_come_back_after_logging_in_again(backend: FakeBackend) -> None:
    backend.saved = list(SAVED)
    app = logged_in()
    next(b for b in app.sidebar.button if b.label == "Log out").click().run()
    app.text_input[0].input("bob.chen@example.com")
    app.text_input[1].input("right")
    app.button[0].click().run()

    assert app.sidebar.selectbox[0].options == [c.title for c in SAVED]
    assert app.session_state["conversation"] == "c2"  # the most recent one


def test_a_conversation_can_be_deleted(backend: FakeBackend) -> None:
    backend.saved = list(SAVED)
    app = logged_in()
    next(b for b in app.sidebar.button if b.label == "Delete").click().run()

    assert backend.deleted == ["c2"]
    assert app.sidebar.selectbox[0].options == ["PTO in November"]


# --- approvals -----------------------------------------------------------------------------


def test_approvals_show_names_and_go_through_the_assistant(backend: FakeBackend) -> None:
    app = logged_in("Approvals")
    assert "Alice Park" in markdown(app)

    next(b for b in app.button if b.label == "Approve").click().run()

    assert backend.sent == ["Approve leave request #6."]
    assert app.session_state["page"] == "Assistant"


# --- tickets -------------------------------------------------------------------------------


def test_employees_see_their_tickets_but_not_the_queue(backend: FakeBackend) -> None:
    app = logged_in("Tickets")
    texts = [t.value for t in app.text]
    assert "My own question" in texts
    assert "Badge broken" not in texts
    assert [s.value for s in app.subheader] == ["My tickets"]


def test_hr_admins_also_see_the_queue_and_start_through_the_assistant(
    backend: FakeBackend,
) -> None:
    app = logged_in("Tickets", Role.HR_ADMIN)
    assert [s.value for s in app.subheader] == ["My tickets", "Open tickets (HR)"]

    next(b for b in app.button if b.label == "Start working on it").click().run()

    assert backend.sent == ["Set HR ticket #7 to in progress."]


def test_ticket_text_is_shown_as_plain_text_not_markdown(backend: FakeBackend) -> None:
    app = logged_in("Tickets", Role.HR_ADMIN)
    assert any("![x](https://evil.example/track.png)" in t.value for t in app.text)
    assert not any("evil.example" in m.value for m in app.markdown)


# --- policy Q&A ----------------------------------------------------------------------------


def ask_policy(backend: FakeBackend, reply: AskResponse | ApiError) -> AppTest:
    backend.ask_reply = reply
    app = AppTest.from_file(str(PAGE))
    app.session_state["page"] = "Policy Q&A"
    app.run()
    app.text_input[0].input("Within how many days do I file an expense report?")
    app.button[0].click()
    return app.run()


def test_policy_answers_need_no_login(backend: FakeBackend) -> None:
    app = ask_policy(backend, AskResponse.model_validate(ASK_OK))

    assert not app.exception
    assert "File within 60 days [1]." in markdown(app)
    assert "home-office-and-stipends" in markdown(app)
    assert any("within 60 days" in c.value for c in app.caption)


def test_not_covered_is_shown_as_info(backend: FakeBackend) -> None:
    reply = AskResponse.model_validate(
        {**ASK_OK, "answer": "Not covered.", "found": False, "citations": []}
    )
    app = ask_policy(backend, reply)
    assert [i.value for i in app.info] == ["Not covered."]


def test_api_errors_show_the_message_and_request_id(backend: FakeBackend) -> None:
    app = ask_policy(backend, ApiError("Service temporarily unavailable.", "req-42"))

    assert [e.value for e in app.error] == ["Service temporarily unavailable."]
    assert any("req-42" in c.value for c in app.caption)
