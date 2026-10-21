"""Test doubles that stand in for real models: offline, instant and deterministic."""

import hashlib
import math
import re
from collections.abc import Sequence

from hrm.rag.embeddings import Vector
from hrm.rag.models import Category, Chunk, ScoredChunk

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words hashing: each word adds 1 to a fixed slot, then the vector is normalized.

    Texts that share words get similar vectors, so retrieval tests behave sensibly,
    without downloading a model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return f"fake-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.document_calls.append(list(texts))
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            slot = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            vector[slot % self._dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector


class InMemoryVectorStore:
    """A dict of chunk_id -> (chunk, vector); query scores every chunk by dot product.

    Fine for tests and tiny data; it must pass the same contract tests as ChromaStore.
    """

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, Vector]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.chunk_id] = (chunk, list(vector))

    def delete_by_doc(self, doc_id: str) -> None:
        self._rows = {k: row for k, row in self._rows.items() if row[0].doc_id != doc_id}

    def query(
        self, vector: Vector, k: int, *, category: Category | None = None
    ) -> list[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        scored = [
            ScoredChunk(chunk=chunk, score=sum(a * b for a, b in zip(vector, v, strict=True)))
            for chunk, v in self._rows.values()
            if category is None or chunk.category == category
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:k]

    def count(self) -> int:
        return len(self._rows)

    def all_chunks(self) -> list[Chunk]:
        return sorted((chunk for chunk, _ in self._rows.values()), key=lambda c: c.chunk_id)
