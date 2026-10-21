import math
from collections.abc import Iterator

import numpy as np
import pytest

from hrm.core.config import PROJECT_ROOT, Settings
from hrm.rag.chunking import chunk_document, tiktoken_counter
from hrm.rag.cleaning import clean_document
from hrm.rag.embeddings import Embedder, FastEmbedEmbedder
from hrm.rag.loading import load_documents
from tests.fakes import FakeEmbedder


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# --- FastEmbedEmbedder logic, with fastembed's model replaced by a stub (offline) ----------


class StubTextEmbedding:
    """Records what it is asked to embed; returns fixed, unnormalized vectors."""

    def __init__(self, model_name: str, cache_dir: str | None) -> None:
        self.passages: list[str] = []
        self.queries: list[str] = []
        self.batch_sizes: list[int] = []

    def passage_embed(self, texts: list[str], batch_size: int) -> Iterator[np.ndarray]:
        self.passages += texts
        self.batch_sizes.append(batch_size)
        for _ in texts:
            yield np.array([3.0, 4.0])  # length 5

    def query_embed(self, query: str) -> Iterator[np.ndarray]:
        self.queries.append(query)
        yield np.array([0.0, 2.0])  # length 2


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> FastEmbedEmbedder:
    monkeypatch.setattr("hrm.rag.embeddings.TextEmbedding", StubTextEmbedding)
    return FastEmbedEmbedder("stub-model", query_prefix="Query: ", batch_size=7)


def test_query_gets_the_instruction_but_documents_do_not(stubbed: FastEmbedEmbedder) -> None:
    stubbed.embed_query("how many weeks of leave?")
    stubbed.embed_documents(["Up to 16 weeks."])

    stub = stubbed._model
    assert isinstance(stub, StubTextEmbedding)
    assert stub.queries[-1] == "Query: how many weeks of leave?"
    assert stub.passages == ["Up to 16 weeks."]
    assert stub.batch_sizes == [7]


def test_vectors_are_normalized_to_length_one(stubbed: FastEmbedEmbedder) -> None:
    [doc_vector] = stubbed.embed_documents(["text"])

    assert doc_vector == pytest.approx([0.6, 0.8])
    assert stubbed.embed_query("q") == pytest.approx([0.0, 1.0])
    assert stubbed.dimension == 2
    assert stubbed.name == "stub-model"


# --- FakeEmbedder: the double the other tests rely on --------------------------------------


def test_fake_embedder_satisfies_the_protocol_and_rewards_shared_words() -> None:
    embedder: Embedder = FakeEmbedder()  # mypy checks the Protocol here

    query = embedder.embed_query("parental leave weeks")
    near, far = embedder.embed_documents(["parental leave is 16 weeks", "gifts and bribery"])

    assert len(query) == embedder.dimension
    assert math.isclose(dot(near, near), 1.0)
    assert dot(query, near) > dot(query, far)


# --- the real model (downloads ~64 MB once): pytest -m network ----------------------------


@pytest.fixture(scope="module")
def real_embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder.from_settings(Settings(_env_file=None))


@pytest.fixture(scope="module")
def real_chunks() -> list[str]:
    settings = Settings(_env_file=None)
    return [
        chunk.embedding_text
        for doc in load_documents(PROJECT_ROOT / "data" / "docs")
        for chunk in chunk_document(
            clean_document(doc),
            max_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
            min_tokens=settings.chunk_min_size,
            count_tokens=tiktoken_counter(),
        )
    ]


@pytest.mark.network
def test_every_chunk_fits_bge_512_token_window(
    real_embedder: FastEmbedEmbedder, real_chunks: list[str]
) -> None:
    tokenizer = real_embedder._model.model.tokenizer  # type: ignore[attr-defined]

    too_long = [t[:60] for t in real_chunks if tokenizer.encode(t).overflowing]

    assert too_long == []  # nothing would be silently cut off by the embedder


@pytest.mark.network
def test_real_model_returns_normalized_384_dim_vectors(real_embedder: FastEmbedEmbedder) -> None:
    [vector] = real_embedder.embed_documents(["Paid time off policy"])

    assert real_embedder.dimension == 384
    assert len(vector) == 384
    assert math.isclose(dot(vector, vector), 1.0, rel_tol=1e-5)


@pytest.mark.network
def test_real_model_ranks_the_right_policy_first(
    real_embedder: FastEmbedEmbedder, real_chunks: list[str]
) -> None:
    vectors = real_embedder.embed_documents(real_chunks)
    query = real_embedder.embed_query("how many weeks of paid parental leave do I get?")

    best = max(range(len(real_chunks)), key=lambda i: dot(query, vectors[i]))

    assert real_chunks[best].startswith("Sourcegraph Paid Parental Leave Policy")
