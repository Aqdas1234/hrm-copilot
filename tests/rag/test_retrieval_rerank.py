import pytest

from hrm.rag.models import Category, Chunk, RetrievedChunk
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.factory import build_retriever
from hrm.rag.retrieval.hybrid import HybridRetriever
from hrm.rag.retrieval.rerank import FastEmbedReranker, RerankingRetriever
from tests.conftest import MakeSettings
from tests.fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        category="conduct",
        audience="all",
        section_path=("Title",),
        text=text,
        token_count=len(text.split()),
    )


class FixedRetriever:
    def __init__(self, *chunks: Chunk) -> None:
        self._chunks = chunks
        self.calls: list[tuple[int, Category | None]] = []

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((k, category))
        return [
            RetrievedChunk(chunk=c, score=1.0, rank=rank)
            for rank, c in enumerate(self._chunks[:k], start=1)
        ]


SIDE_LETTERS = make_chunk("side-letters", "Do not sign side letters with customers.")
OUTSIDE_WORK = make_chunk("outside-work", "Freelance or outside work for another company.")
GIFTS = make_chunk("gifts", "Gifts around 50 USD are fine.")


def test_reranker_reorders_candidates_and_renumbers_ranks() -> None:
    base = FixedRetriever(SIDE_LETTERS, GIFTS, OUTSIDE_WORK)  # right answer is 3rd
    reranking = RerankingRetriever(base, FakeReranker(), candidates=20)

    results = reranking.retrieve("freelance work for another company", k=2)

    assert [r.chunk.chunk_id for r in results] == ["outside-work", "side-letters"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score == 5.0  # 5 shared words; the reranker score replaces the old one


def test_fetches_candidates_and_passes_the_category_through() -> None:
    base = FixedRetriever(SIDE_LETTERS)

    RerankingRetriever(base, FakeReranker(), candidates=20).retrieve("q", k=5, category="conduct")

    assert base.calls == [(20, "conduct")]


def test_reranker_sees_section_path_and_text() -> None:
    reranker = FakeReranker()

    RerankingRetriever(FixedRetriever(GIFTS), reranker).retrieve("gifts", k=1)

    assert reranker.calls == [("gifts", [GIFTS.embedding_text])]


def test_no_candidates_means_no_reranker_call() -> None:
    reranker = FakeReranker()

    assert RerankingRetriever(FixedRetriever(), reranker).retrieve("q", k=3) == []
    assert reranker.calls == []


def test_satisfies_the_retriever_protocol() -> None:
    as_protocol: Retriever = RerankingRetriever(FixedRetriever(GIFTS), FakeReranker())

    assert as_protocol.retrieve("gifts", k=1)[0].chunk.chunk_id == "gifts"


def test_factory_wraps_the_base_retriever_only_when_rerank_is_on(
    make_settings: MakeSettings,
) -> None:
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    store.upsert([GIFTS], embedder.embed_documents([GIFTS.embedding_text]))

    plain = build_retriever(make_settings(retrieval_mode="hybrid"), embedder, store)
    reranked = build_retriever(
        make_settings(retrieval_mode="hybrid", rerank=True), embedder, store, FakeReranker()
    )

    assert isinstance(plain, HybridRetriever)
    assert isinstance(reranked, RerankingRetriever)


# --- the real cross-encoder (downloads ~80 MB once): pytest -m network ---------------------


@pytest.mark.network
def test_real_cross_encoder_prefers_the_relevant_text(make_settings: MakeSettings) -> None:
    reranker = FastEmbedReranker.from_settings(make_settings())

    side_letters, outside_work = reranker.score(
        "Can I do freelance work for another company on the side?",
        [
            "Do not enter into any side letter, agreement, or promise with a customer.",
            "Outside employment: you want to work for or provide services to a third party.",
        ],
    )

    assert outside_work > side_letters
