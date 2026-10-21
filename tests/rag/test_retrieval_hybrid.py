import pytest

from hrm.rag.models import Category, Chunk, RetrievedChunk
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.bm25 import BM25Retriever
from hrm.rag.retrieval.dense import DenseRetriever
from hrm.rag.retrieval.factory import build_retriever
from hrm.rag.retrieval.hybrid import HybridRetriever
from tests.conftest import MakeSettings
from tests.fakes import FakeEmbedder, InMemoryVectorStore


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        category="conduct",
        audience="all",
        section_path=("Title", chunk_id),
        text=f"text {chunk_id}",
        token_count=2,
    )


class FixedRetriever:
    """Returns a fixed ranking and records the calls it receives."""

    def __init__(self, *chunk_ids: str) -> None:
        self._ids = chunk_ids
        self.calls: list[tuple[int, Category | None]] = []

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((k, category))
        return [
            RetrievedChunk(chunk=make_chunk(cid), score=1.0, rank=rank)
            for rank, cid in enumerate(self._ids[:k], start=1)
        ]


def ids(results: list[RetrievedChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


def test_chunk_ranked_well_by_both_beats_chunk_found_by_one() -> None:
    dense = FixedRetriever("a", "b", "c")
    bm25 = FixedRetriever("c", "d", "a")

    results = HybridRetriever([dense, bm25], rrf_k=60).retrieve("q", k=4)

    # a: 1/61 + 1/63, c: 1/63 + 1/61 (tie, broken by id), then b: 1/62, d: 1/62
    assert ids(results) == ["a", "c", "b", "d"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 63)
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_each_chunk_appears_once_and_k_limits_results() -> None:
    results = HybridRetriever([FixedRetriever("a", "b"), FixedRetriever("b", "a")]).retrieve(
        "q", k=1
    )

    assert len(results) == 1


def test_candidates_and_category_are_passed_to_every_retriever() -> None:
    dense, bm25 = FixedRetriever("a"), FixedRetriever("b")

    HybridRetriever([dense, bm25], candidates=20).retrieve("q", k=5, category="benefits")

    assert dense.calls == bm25.calls == [(20, "benefits")]


def test_a_retriever_that_finds_nothing_does_not_break_fusion() -> None:
    results = HybridRetriever([FixedRetriever("a", "b"), FixedRetriever()]).retrieve("q", k=5)

    assert ids(results) == ["a", "b"]


def test_needs_at_least_one_retriever() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_satisfies_the_retriever_protocol() -> None:
    as_protocol: Retriever = HybridRetriever([FixedRetriever("a")])

    assert ids(as_protocol.retrieve("q", k=1)) == ["a"]


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [("dense", DenseRetriever), ("bm25", BM25Retriever), ("hybrid", HybridRetriever)],
)
def test_factory_builds_the_configured_retriever(
    make_settings: MakeSettings, mode: str, expected_type: type
) -> None:
    store = InMemoryVectorStore()
    embedder = FakeEmbedder()
    store.upsert([make_chunk("a")], embedder.embed_documents(["text a"]))

    settings = make_settings(retrieval_mode=mode, rerank=False)
    retriever = build_retriever(settings, embedder, store)

    assert isinstance(retriever, expected_type)
