import pytest

from hrm.rag.models import Category, Chunk
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.dense import DenseRetriever
from tests.fakes import FakeEmbedder, InMemoryVectorStore


def make_chunk(chunk_id: str, text: str, category: Category = "conduct") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        category=category,
        audience="all",
        section_path=("Title", chunk_id),
        text=text,
        token_count=len(text.split()),
    )


CHUNKS = [
    make_chunk("leave:weeks", "parental leave lasts up to sixteen weeks", "parental_leave"),
    make_chunk("leave:pay", "parental leave is paid at full salary", "parental_leave"),
    make_chunk("gifts:vendor", "do not accept expensive gifts from a vendor", "conduct"),
    make_chunk("benefits:401k", "the company does not match the 401k", "benefits"),
]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def retriever(embedder: FakeEmbedder) -> DenseRetriever:
    store = InMemoryVectorStore()
    store.upsert(CHUNKS, embedder.embed_documents([c.embedding_text for c in CHUNKS]))
    return DenseRetriever(embedder, store)


def test_returns_ranked_results_best_first(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("how many weeks of parental leave", k=3)

    assert results[0].chunk.chunk_id == "leave:weeks"
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_question_is_embedded_as_a_query(retriever: DenseRetriever, embedder: FakeEmbedder) -> None:
    retriever.retrieve("  can I accept a gift?  ", k=1)

    assert embedder.query_calls[-1] == "can I accept a gift?"  # query path, whitespace trimmed


def test_category_filter_is_passed_to_the_store(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("parental leave", k=5, category="benefits")

    assert [r.chunk.chunk_id for r in results] == ["benefits:401k"]


def test_empty_query_is_rejected(retriever: DenseRetriever) -> None:
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   ", k=3)


def test_dense_retriever_satisfies_the_protocol(retriever: DenseRetriever) -> None:
    as_protocol: Retriever = retriever  # checked by mypy

    assert as_protocol.retrieve("401k", k=1)[0].chunk.chunk_id == "benefits:401k"
