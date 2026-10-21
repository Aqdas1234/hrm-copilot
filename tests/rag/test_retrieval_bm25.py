import pytest

from hrm.rag.models import Category, Chunk
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.bm25 import BM25Retriever, tokenize


def make_chunk(chunk_id: str, text: str, category: Category = "benefits") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        category=category,
        audience="all",
        section_path=("Policy", chunk_id.split(":")[1]),
        text=text,
        token_count=len(text.split()),
    )


CHUNKS = [
    make_chunk("benefits:401k", "Human Interest handles the 401k for US based teammates."),
    make_chunk("benefits:eligibility", "Dependent children up until age 26 are eligible."),
    make_chunk("conduct:gifts", "Gifts around $50 USD or less are fine.", "conduct"),
    make_chunk("pto:rules", "Inform your manager 5 business days before time off.", "time_off"),
]


@pytest.fixture
def retriever() -> BM25Retriever:
    return BM25Retriever(CHUNKS)


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("How many weeks of PTO do I get?") == ["many", "weeks", "pto", "get"]
    assert tokenize("my 401k") == ["401k"]


def test_rare_exact_term_ranks_its_chunk_first(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("who handles my 401k?", k=3)

    assert results[0].chunk.chunk_id == "benefits:401k"
    assert results[0].rank == 1


def test_section_heading_words_are_searchable(retriever: BM25Retriever) -> None:
    # "eligibility" only appears in the section path, not in the chunk text.
    assert retriever.retrieve("eligibility", k=1)[0].chunk.chunk_id == "benefits:eligibility"


def test_chunks_without_any_query_word_are_not_returned(retriever: BM25Retriever) -> None:
    assert retriever.retrieve("quantum physics", k=5) == []


def test_category_filter(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("manager gifts 401k", k=5, category="conduct")

    assert [r.chunk.chunk_id for r in results] == ["conduct:gifts"]


@pytest.mark.parametrize(("query", "k"), [("  ", 3), ("401k", 0)])
def test_invalid_input_is_rejected(retriever: BM25Retriever, query: str, k: int) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve(query, k)


def test_satisfies_the_retriever_protocol(retriever: BM25Retriever) -> None:
    as_protocol: Retriever = retriever

    assert as_protocol.retrieve("age 26", k=1)[0].chunk.chunk_id == "benefits:eligibility"
