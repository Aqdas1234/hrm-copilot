"""Contract tests: every VectorStore implementation must pass all of them.

The `store` fixture runs each test once per implementation, which proves the in-memory
fake used by other tests behaves like the real ChromaStore. Chroma runs locally on a
temporary folder, so these tests stay offline.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from hrm.rag.models import Category, Chunk
from hrm.rag.vectorstore import ChromaStore, VectorStore
from tests.fakes import InMemoryVectorStore


@pytest.fixture(params=["memory", "chroma"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    if request.param == "memory":
        return InMemoryVectorStore()
    return ChromaStore(tmp_path / "chroma")


def make_chunk(chunk_id: str, doc_id: str = "doc-a", category: Category = "conduct") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        category=category,
        audience="all",
        section_path=("Title", "Section", chunk_id),
        text=f"text of {chunk_id}",
        token_count=12,
    )


# Unit vectors in 3D: X is closest to "x", then "xy" (45 degrees), then "y" (90 degrees).
X, Y, XY = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7071068, 0.7071068, 0.0]


def test_query_returns_closest_first_with_cosine_scores(store: VectorStore) -> None:
    store.upsert([make_chunk("x"), make_chunk("y"), make_chunk("xy")], [X, Y, XY])

    hits = store.query(X, k=3)

    assert [h.chunk.chunk_id for h in hits] == ["x", "xy", "y"]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.7071068, 0.0], abs=1e-4)


def test_query_returns_chunks_unchanged(store: VectorStore) -> None:
    original = make_chunk("x")
    store.upsert([original], [X])

    [hit] = store.query(X, k=1)

    assert hit.chunk == original  # every field survives the round trip, incl. section_path


def test_k_limits_results_and_may_exceed_the_store_size(store: VectorStore) -> None:
    store.upsert([make_chunk("x"), make_chunk("y")], [X, Y])

    assert len(store.query(X, k=1)) == 1
    assert len(store.query(X, k=10)) == 2


def test_upsert_with_existing_id_replaces_the_chunk(store: VectorStore) -> None:
    store.upsert([make_chunk("x")], [X])
    updated = make_chunk("x").model_copy(update={"text": "new text"})

    store.upsert([updated], [Y])

    assert store.count() == 1
    [hit] = store.query(Y, k=1)
    assert hit.chunk.text == "new text"
    assert hit.score == pytest.approx(1.0, abs=1e-4)  # the vector was replaced too


def test_delete_by_doc_removes_only_that_document(store: VectorStore) -> None:
    store.upsert(
        [
            make_chunk("a1", doc_id="doc-a"),
            make_chunk("a2", doc_id="doc-a"),
            make_chunk("b1", "doc-b"),
        ],
        [X, XY, Y],
    )

    store.delete_by_doc("doc-a")

    assert store.count() == 1
    assert [h.chunk.chunk_id for h in store.query(X, k=5)] == ["b1"]


def test_category_filter(store: VectorStore) -> None:
    store.upsert(
        [make_chunk("x", category="conduct"), make_chunk("y", category="benefits")], [X, Y]
    )

    hits = store.query(X, k=5, category="benefits")

    assert [h.chunk.chunk_id for h in hits] == ["y"]


def test_empty_store_returns_no_hits(store: VectorStore) -> None:
    assert store.query(X, k=3) == []
    assert store.count() == 0


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda s: s.upsert([make_chunk("x")], []),  # chunks and vectors must line up
        lambda s: s.query(X, k=0),
    ],
)
def test_invalid_input_is_rejected(
    store: VectorStore, bad_call: Callable[[VectorStore], object]
) -> None:
    with pytest.raises(ValueError):
        bad_call(store)


# --- Chroma only -------------------------------------------------------------------------


def test_chroma_keeps_data_on_disk_between_instances(tmp_path: Path) -> None:
    ChromaStore(tmp_path / "chroma").upsert([make_chunk("x")], [X])

    reopened = ChromaStore(tmp_path / "chroma")

    assert reopened.count() == 1
    assert reopened.query(X, k=1)[0].chunk.chunk_id == "x"
