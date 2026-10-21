"""Dense (semantic) retrieval: embed the question, return the nearest chunk vectors."""

import time

from hrm.core.logging import get_logger
from hrm.rag.embeddings import Embedder
from hrm.rag.models import Category, RetrievedChunk
from hrm.rag.vectorstore import VectorStore

log = get_logger(__name__)


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        start = time.perf_counter()
        hits = self._store.query(self._embedder.embed_query(query), k, category=category)
        results = [
            RetrievedChunk(chunk=hit.chunk, score=hit.score, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
        log.info(
            "retrieval.dense",
            k=k,
            category=category,
            hits=len(results),
            top_score=round(results[0].score, 3) if results else None,
            ms=round((time.perf_counter() - start) * 1000),
        )
        return results
