import json

import pytest

from hrm.core.logging import configure_logging
from hrm.rag.generation import not_covered
from hrm.rag.models import Category, Chunk, Citation, PolicyAnswer, RetrievedChunk
from hrm.rag.service import RAGService


def retrieved(chunk_id: str, score: float, rank: int, tokens: int = 100) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        category="time_off",
        audience="all",
        section_path=("Policy", chunk_id),
        text=f"text of {chunk_id}",
        token_count=tokens,
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=rank)


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, Category | None]] = []

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, category))
        return self._results[:k]


class FakeGenerator:
    """Answers "ok [1]" citing the first chunk, or not covered when given nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> PolicyAnswer:
        self.calls.append((question, chunks))
        if not chunks:
            return not_covered()
        first = chunks[0].chunk
        citation = Citation(
            number=1, chunk_id=first.chunk_id, doc_id=first.doc_id, section_path=first.section_path
        )
        return PolicyAnswer(answer="ok [1]", found=True, citations=[citation])


RESULTS = [retrieved(f"doc{i}:s:0", score=-1.0 - i, rank=i + 1) for i in range(8)]


def make_service(
    top_k: int = 5, max_context_tokens: int = 2500, min_score: float | None = None
) -> tuple[RAGService, FakeRetriever, FakeGenerator]:
    retriever, generator = FakeRetriever(RESULTS), FakeGenerator()
    service = RAGService(
        retriever,
        generator,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        min_score=min_score,
    )
    return service, retriever, generator


def test_answer_retrieves_top_k_and_passes_them_to_the_generator() -> None:
    service, retriever, generator = make_service(top_k=5)

    result = service.answer("How much notice?", category="time_off")

    assert retriever.calls == [("How much notice?", 5, "time_off")]
    assert [r.chunk.chunk_id for r in generator.calls[0][1]] == [f"doc{i}:s:0" for i in range(5)]
    assert result.answer.found
    assert result.context == generator.calls[0][1]


def test_irrelevant_results_mean_no_context_and_a_not_covered_answer() -> None:
    service, _, generator = make_service(min_score=0.0)  # best score is -1.0

    result = service.answer("Stock options?")

    assert not result.answer.found
    assert result.context == []
    assert generator.calls == [("Stock options?", [])]  # the real generator skips the LLM here


def test_context_budget_is_applied() -> None:
    service, _, generator = make_service(top_k=5, max_context_tokens=250)  # 100 tokens each

    service.answer("q")

    assert len(generator.calls[0][1]) == 2


def test_search_uses_top_k_unless_k_is_given() -> None:
    service, retriever, _ = make_service(top_k=3)

    assert len(service.search("q")) == 3
    assert len(service.search("q", k=7, category="benefits")) == 7
    assert retriever.calls[-1] == ("q", 7, "benefits")


def test_answer_reports_step_timings_and_logs_one_summary_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json")
    service, _, _ = make_service(top_k=3)

    result = service.answer("How much notice?")

    t = result.timings
    assert t.total_ms >= t.retrieve_ms + t.generate_ms - 1  # rounding
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]
    [summary] = [r for r in records if r["event"] == "rag.answer"]
    assert summary["found"] is True
    assert summary["citations"] == 1
    assert summary["context_chunks"] == 3
    assert summary["top_score"] == -1.0
    assert {"retrieve_ms", "generate_ms", "total_ms"} <= summary.keys()
