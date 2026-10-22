"""HTTP client the UI uses to call the API. Kept separate from Streamlit so it is testable.

The UI holds no data of its own: conversations, pending confirmations and approvals all
come from the API, so a browser refresh or a second tab shows the same state.
"""

from typing import Any

import httpx
from pydantic import BaseModel

from hrm.agents.runner import AgentReply
from hrm.api.chat_routes import Conversation
from hrm.api.schemas import AskResponse
from hrm.auth.access import EmployeeView, LeaveRequestView
from hrm.db.models import Role
from hrm.rag.models import Category
from hrm.workflows.tickets import TicketView

REQUEST_ID_HEADER = "X-Request-ID"


class ApiError(Exception):
    """A user-presentable API failure, with the request id to quote when reporting it."""

    def __init__(
        self, message: str, request_id: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.status_code = status_code

    @property
    def login_expired(self) -> bool:
        return self.status_code == 401


class Login(BaseModel):
    token: str
    role: Role  # what to show; the API enforces it on every request


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ):
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._token = token

    # --- public ---------------------------------------------------------------------------

    def ask(self, question: str, category: Category | None = None) -> AskResponse:
        body = self._call("POST", "/rag/ask", {"question": question, "category": category})
        return AskResponse.model_validate(body)

    def login(self, email: str, password: str) -> Login:
        body = self._call("POST", "/auth/login", {"email": email, "password": password})
        return Login(token=body["access_token"], role=body["role"])

    # --- logged in ------------------------------------------------------------------------

    def me(self) -> EmployeeView:
        return EmployeeView.model_validate(self._call("GET", "/me"))

    def team(self) -> list[EmployeeView]:
        return [EmployeeView.model_validate(e) for e in self._call("GET", "/team")]

    def my_tickets(self) -> list[TicketView]:
        return [TicketView.model_validate(t) for t in self._call("GET", "/me/tickets")]

    def ticket_queue(self) -> list[TicketView]:
        return [TicketView.model_validate(t) for t in self._call("GET", "/tickets/open")]

    def approvals(self) -> list[LeaveRequestView]:
        return [LeaveRequestView.model_validate(r) for r in self._call("GET", "/team/approvals")]

    def send(self, conversation_id: str, message: str) -> AgentReply:
        path = f"/chat/{conversation_id}/messages"
        return AgentReply.model_validate(self._call("POST", path, {"message": message}))

    def confirm(self, conversation_id: str, approved: bool) -> AgentReply:
        path = f"/chat/{conversation_id}/confirm"
        return AgentReply.model_validate(self._call("POST", path, {"approved": approved}))

    def conversation(self, conversation_id: str) -> Conversation:
        return Conversation.model_validate(self._call("GET", f"/chat/{conversation_id}"))

    # --- helpers --------------------------------------------------------------------------

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            response = self._client.request(method, path, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Cannot reach the API ({type(exc).__name__}). Is `hrm serve` running?"
            ) from exc
        if not response.is_success:
            raise ApiError(
                _detail(response), response.headers.get(REQUEST_ID_HEADER), response.status_code
            )
        return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail
    if response.status_code == 422:
        return "Please check your input (it may be empty or too long)."
    return f"The API returned an error (HTTP {response.status_code})."
