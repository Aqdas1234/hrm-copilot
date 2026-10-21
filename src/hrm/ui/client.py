"""HTTP client the UI uses to call the API. Kept separate from Streamlit so it is testable."""

import httpx

from hrm.api.schemas import AskResponse
from hrm.rag.models import Category

REQUEST_ID_HEADER = "X-Request-ID"


class ApiError(Exception):
    """A user-presentable API failure, with the request id to quote when reporting it."""

    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class ApiClient:
    def __init__(
        self, base_url: str, *, timeout_s: float = 60.0, client: httpx.Client | None = None
    ):
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def ask(self, question: str, category: Category | None = None) -> AskResponse:
        try:
            response = self._client.post(
                "/rag/ask", json={"question": question, "category": category}
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Cannot reach the API ({type(exc).__name__}). Is `hrm serve` running?"
            ) from exc
        request_id = response.headers.get(REQUEST_ID_HEADER)
        if response.status_code != 200:
            raise ApiError(_detail(response), request_id)
        return AskResponse.model_validate(response.json())


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail
    if response.status_code == 422:
        return "Please enter a question (up to 500 characters)."
    return f"The API returned an error (HTTP {response.status_code})."
