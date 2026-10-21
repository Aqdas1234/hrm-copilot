"""Store chunk vectors and find the ones closest to a query vector.

VectorStore is a Protocol, so the rest of the code does not depend on ChromaDB:
tests use an in-memory fake, and a server-backed store (e.g. pgvector) could replace it.
Both implementations must pass the same contract tests (tests/rag/test_vectorstore.py).

Vectors are expected to be normalized (length 1), so cosine similarity is the score.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from hrm.core.config import Settings
from hrm.rag.embeddings import Vector
from hrm.rag.models import Category, Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Insert chunks, or replace the ones whose chunk_id already exists."""
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove every chunk of one document (used when it changes or is deleted)."""
        ...

    def query(
        self, vector: Vector, k: int, *, category: Category | None = None
    ) -> list[ScoredChunk]:
        """The k closest chunks, best first, optionally only from one category."""
        ...

    def count(self) -> int: ...

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk, sorted by chunk_id (e.g. to build a keyword index)."""
        ...


class ChromaStore:
    """ChromaDB saved on disk. We pass our own vectors; Chroma never embeds anything."""

    def __init__(self, path: Path, collection: str = "hr_policies") -> None:
        client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),  # on by default; keep data local
        )
        self._collection = client.get_or_create_collection(
            collection,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChromaStore":
        return cls(settings.index_dir / "chroma")

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        _check_same_length(chunks, vectors)
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32),
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )

    def delete_by_doc(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def query(
        self, vector: Vector, k: int, *, category: Category | None = None
    ) -> list[ScoredChunk]:
        _check_k(k)
        result = self._collection.query(
            query_embeddings=np.asarray([vector], dtype=np.float32),
            n_results=k,
            where={"category": category} if category else None,
            include=["documents", "metadatas", "distances"],
        )
        # One query was sent, so each field holds one list of results: take index [0].
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            ScoredChunk(chunk=_from_metadata(i, d, m), score=1.0 - dist)  # cosine distance -> score
            for i, d, m, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]

    def count(self) -> int:
        return self._collection.count()

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        chunks = [
            _from_metadata(i, d, m)
            for i, d, m in zip(result["ids"], documents, metadatas, strict=True)
        ]
        return sorted(chunks, key=lambda c: c.chunk_id)


# Chroma metadata values must be str, int, float or bool, so the section path is stored as JSON.
def _to_metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "doc_id": chunk.doc_id,
        "category": chunk.category,
        "audience": chunk.audience,
        "section_path": json.dumps(chunk.section_path),
        "token_count": chunk.token_count,
    }


def _from_metadata(chunk_id: str, text: str | None, metadata: Any) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=metadata["doc_id"],
        category=metadata["category"],
        audience=metadata["audience"],
        section_path=tuple(json.loads(metadata["section_path"])),
        text=text or "",
        token_count=metadata["token_count"],
    )


def _check_same_length(chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
