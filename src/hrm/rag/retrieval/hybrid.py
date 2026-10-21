"""Hybrid retrieval: fuse several retrievers' rankings with Reciprocal Rank Fusion (RRF).

Each chunk gets sum(1 / (rrf_k + rank)) over the lists it appears in. Only ranks are used,
so retrievers with incomparable scores (cosine 0..1 vs. BM25 0..30) combine fairly, and a
chunk ranked well by several retrievers rises to the top. rrf_k=60 is the value from the
original RRF paper (Cormack et al., 2009); it keeps one list's top pick from dominating.
"""

import time
from collections.abc import Sequence

from hrm.core.logging import get_logger
from hrm.rag.models import Category, Chunk, RetrievedChunk
from hrm.rag.retrieval import Retriever

log = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self, retrievers: Sequence[Retriever], *, candidates: int = 20, rrf_k: int = 60
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = list(retrievers)
        self._candidates = candidates  # how many results to take from each retriever
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        start = time.perf_counter()
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            depth = max(k, self._candidates)
            for result in retriever.retrieve(query, depth, category=category):
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + result.rank)
                chunks[chunk_id] = result.chunk

        # Highest fused score first; chunk_id breaks ties so results are deterministic.
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
        results = [
            RetrievedChunk(chunk=chunks[cid], score=scores[cid], rank=rank)
            for rank, cid in enumerate(ranked, start=1)
        ]
        log.info(
            "retrieval.hybrid",
            k=k,
            category=category,
            hits=len(results),
            ms=round((time.perf_counter() - start) * 1000),
        )
        return results
