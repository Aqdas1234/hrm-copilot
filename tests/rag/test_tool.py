from hrm.rag.models import Audience, Category, Chunk, RetrievedChunk
from hrm.rag.tool import NO_RESULTS, TOOL_NAME, make_policy_search_tool


def retrieved(chunk_id: str, text: str, audience: Audience = "all") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        category="benefits",
        audience=audience,
        section_path=("Health benefits at Sourcegraph", "Eligibility"),
        text=text,
        token_count=5,
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=1)


class FakeService:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int | None, Category | None]] = []

    def search(
        self, query: str, k: int | None = None, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, category))
        return self._results


def test_tool_has_a_name_description_and_arguments_for_the_agent() -> None:
    search = make_policy_search_tool(FakeService([]))

    assert search.name == TOOL_NAME
    assert "HR policies" in search.description
    assert set(search.args) == {"query", "category"}


def test_tool_returns_numbered_excerpts_with_sources_and_audience() -> None:
    service = FakeService(
        [retrieved("benefits-overview:eligibility:0", "Children up to age 26.", "us")]
    )
    search = make_policy_search_tool(service, k=3)

    output = search.invoke({"query": "children coverage age", "category": "benefits"})

    assert output == (
        "[1] benefits-overview > Eligibility (US-based teammates only)\nChildren up to age 26."
    )
    assert service.calls == [("children coverage age", 3, "benefits")]


def test_tool_says_when_nothing_is_found() -> None:
    assert make_policy_search_tool(FakeService([])).invoke({"query": "stock options"}) == NO_RESULTS
