"""The UI's API client, with a mock HTTP transport (no server needed)."""

import httpx
import pytest

from hrm.ui.client import ApiClient, ApiError

ASK_OK = {
    "answer": "File within 60 days [1].",
    "found": True,
    "citations": [
        {
            "number": 1,
            "doc_id": "home-office-and-stipends",
            "section": "Spending company money",
            "chunk_id": "home-office-and-stipends:spending-company-money:0",
            "quote": "Please file an expense report within 60 days ...",
        }
    ],
    "timings": {"retrieve_ms": 900, "generate_ms": 1200, "total_ms": 2100},
}


def client_returning(
    response: httpx.Response, seen: list[httpx.Request] | None = None
) -> ApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return response

    return ApiClient(
        "http://api",
        client=httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)),
    )


def test_ask_posts_the_question_and_parses_the_answer() -> None:
    seen: list[httpx.Request] = []
    client = client_returning(httpx.Response(200, json=ASK_OK), seen)

    result = client.ask("expense deadline?", "expenses")

    assert result.found
    assert result.citations[0].quote.startswith("Please file")
    assert seen[0].url.path == "/rag/ask"
    assert seen[0].read() == b'{"question":"expense deadline?","category":"expenses"}'


def test_api_error_carries_the_message_and_request_id() -> None:
    response = httpx.Response(
        502,
        json={"detail": "The answer service is temporarily unavailable. Please retry."},
        headers={"X-Request-ID": "abc123"},
    )

    with pytest.raises(ApiError) as caught:
        client_returning(response).ask("q")

    assert "temporarily unavailable" in str(caught.value)
    assert caught.value.request_id == "abc123"


def test_validation_error_gets_a_friendly_message() -> None:
    with pytest.raises(ApiError, match="Please enter a question"):
        client_returning(httpx.Response(422, json={"detail": [{"msg": "too short"}]})).ask("")


def test_unreachable_api_suggests_starting_the_server() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = ApiClient(
        "http://api",
        client=httpx.Client(base_url="http://api", transport=httpx.MockTransport(refuse)),
    )

    with pytest.raises(ApiError, match="hrm serve"):
        client.ask("q")
