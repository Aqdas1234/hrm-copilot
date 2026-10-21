"""Reranking: re-score retrieved candidates with a cross-encoder.

A cross-encoder reads the question and one chunk together and outputs a relevance score.
That is more accurate than comparing two separately computed vectors, but too slow to run
on every chunk, so it only re-sorts the candidates another retriever already found.
"""

import time
from collections.abc import Sequence
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from hrm.core.config import Settings
from hrm.core.logging import get_logger
from hrm.rag.models import Category, RetrievedChunk
from hrm.rag.retrieval import Retriever

log = get_logger(__name__)


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the query; higher is more relevant."""
        ...


class FastEmbedReranker:
    """Local ONNX cross-encoder (default: ms-marco-MiniLM-L-6-v2, ~80 MB)."""

    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        log.info("reranker.loading", model=model_name)
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    @classmethod
    def from_settings(cls, settings: Settings) -> "FastEmbedReranker":
        return cls(settings.reranker_model, cache_dir=str(settings.model_cache_dir))

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        # Scores are raw logits (unbounded); only their order matters here.
        return [float(s) for s in self._model.rerank(query, list(texts))]


class RerankingRetriever:
    """Wraps a retriever: take its top `candidates`, re-sort them with a reranker, keep k."""

    def __init__(self, retriever: Retriever, reranker: Reranker, *, candidates: int = 20) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        candidates = self._retriever.retrieve(query, max(k, self._candidates), category=category)
        if not candidates:
            return []
        start = time.perf_counter()
        scores = self._reranker.score(query, [c.chunk.embedding_text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]
        results = [
            RetrievedChunk(chunk=candidates[i].chunk, score=scores[i], rank=rank)
            for rank, i in enumerate(order, start=1)
        ]
        log.info(
            "retrieval.rerank",
            candidates=len(candidates),
            k=k,
            moved_to_top=candidates[order[0]].rank,  # where the new #1 was before reranking
            ms=round((time.perf_counter() - start) * 1000),
        )
        return results
