"""The HTTP API with a fake service: real FastAPI app, middleware and validation, no models."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from hrm import __version__
from hrm.api.app import REQUEST_ID_HEADER, create_app
from hrm.core.errors import ConfigurationError, LLMError
from hrm.core.logging import configure_logging
from hrm.rag.generation import not_covered
from hrm.rag.models import Category, Chunk, Citation, PolicyAnswer, RetrievedChunk
from hrm.rag.service import AnswerResult

CHUNK = Chunk(
    chunk_id="time-off-policy:pto-ground-rules:0",
    doc_id="time-off-policy",
    category="time_off",
    audience="all",
    section_path=("Paid time off (PTO) policy", "PTO ground rules"),
    text="Inform your Manager a minimum of 5 business days before your leave.",
    token_count=20,
)
ANSWER = PolicyAnswer(
    answer="Give at least 5 business days notice [1].",
    found=True,
    citations=[
        Citation(
            number=1,
            chunk_id=CHUNK.chunk_id,
            doc_id=CHUNK.doc_id,
            section_path=CHUNK.section_path,
        )
    ],
)


class FakeService:
    def __init__(self, answer: PolicyAnswer = ANSWER, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.search_calls: list[tuple[str, int | None, Category | None]] = []
        self.ask_calls: list[tuple[str, Category | None]] = []

    def search(
        self, query: str, k: int | None = None, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        self.search_calls.append((query, k, category))
        return [RetrievedChunk(chunk=CHUNK, score=3.5, rank=1)]

    def answer(self, question: str, *, category: Category | None = None) -> AnswerResult:
        self.ask_calls.append((question, category))
        if self._error:
            raise self._error
        return AnswerResult(answer=self._answer, context=[])


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def client(service: FakeService) -> Iterator[TestClient]:
    with TestClient(create_app(service)) as test_client:  # `with` runs the startup step
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_search_returns_ranked_hits(client: TestClient, service: FakeService) -> None:
    response = client.post("/rag/search", json={"query": "notice", "k": 3, "category": "time_off"})

    assert response.status_code == 200
    [hit] = response.json()["results"]
    assert hit["chunk_id"] == CHUNK.chunk_id
    assert hit["section"] == "Paid time off (PTO) policy > PTO ground rules"
    assert hit["rank"] == 1
    assert service.search_calls == [("notice", 3, "time_off")]


def test_ask_returns_answer_and_citations(client: TestClient, service: FakeService) -> None:
    response = client.post("/rag/ask", json={"question": "How much notice?"})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["answer"] == "Give at least 5 business days notice [1]."
    assert body["citations"] == [
        {
            "number": 1,
            "doc_id": "time-off-policy",
            "section": "Paid time off (PTO) policy > PTO ground rules",
            "chunk_id": CHUNK.chunk_id,
        }
    ]
    assert service.ask_calls == [("How much notice?", None)]
    assert set(body["timings"]) == {"retrieve_ms", "generate_ms", "total_ms"}


def test_not_covered_is_a_normal_200_answer() -> None:
    with TestClient(create_app(FakeService(answer=not_covered()))) as client:
        body = client.post("/rag/ask", json={"question": "Stock options?"}).json()

    assert body["found"] is False
    assert body["citations"] == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/rag/ask", {"question": ""}),
        ("/rag/ask", {"question": "x" * 501}),
        ("/rag/ask", {"question": "q", "category": "payroll"}),
        ("/rag/search", {"query": "q", "k": 0}),
        ("/rag/search", {"query": "q", "k": 21}),
        ("/rag/search", {}),
    ],
)
def test_invalid_requests_are_rejected(
    client: TestClient, service: FakeService, path: str, payload: dict[str, object]
) -> None:
    response = client.post(path, json=payload)

    assert response.status_code == 422
    assert service.search_calls == []  # never reached the service
    assert service.ask_calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [(LLMError("Groq 429: secret details"), 502), (ConfigurationError("no API key"), 503)],
)
def test_known_failures_map_to_http_errors_without_leaking_details(
    error: Exception, status: int
) -> None:
    with TestClient(create_app(FakeService(error=error))) as client:
        response = client.post("/rag/ask", json={"question": "q"})

    assert response.status_code == status
    assert "secret" not in response.text and "API key" not in response.text


def test_request_id_is_returned_and_generated_when_missing(client: TestClient) -> None:
    given = client.get("/health", headers={REQUEST_ID_HEADER: "trace-123"})
    generated = client.get("/health")
    unsafe = client.get("/health", headers={REQUEST_ID_HEADER: "bad id\nwith newline"})

    assert given.headers[REQUEST_ID_HEADER] == "trace-123"
    assert len(generated.headers[REQUEST_ID_HEADER]) == 32
    assert unsafe.headers[REQUEST_ID_HEADER] != "bad id\nwith newline"


def test_request_id_appears_in_the_request_log(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO", "json")

    client.get("/health", headers={REQUEST_ID_HEADER: "trace-456"})

    records = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]
    request_log = next(r for r in records if r["event"] == "http.request")
    assert request_log["request_id"] == "trace-456"
    assert request_log["status"] == 200
