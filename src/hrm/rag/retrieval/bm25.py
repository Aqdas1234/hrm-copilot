"""Keyword retrieval with BM25 (Okapi).

BM25 scores a chunk by the query words it contains, weighting rare words (e.g. "401k",
"Convercent") more than common ones, and dampening very long chunks. It knows nothing
about meaning, but it is precise for exact terms, which is where dense search is weakest.

The index is built in memory from chunks (e.g. VectorStore.all_chunks()); for a few
hundred chunks this takes milliseconds, so there is no separate index file to keep in sync.
"""

import re
import time

from rank_bm25 import BM25Okapi

from hrm.core.logging import get_logger
from hrm.rag.models import Category, Chunk, RetrievedChunk

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that appear in almost every question or chunk and carry no meaning for search.
_STOPWORDS_TEXT = """
a an and are as at be but by can could do does for from had has have how i if in is it its
me my of on or our should so that the their them then there these they this to us was we
what when where which who why will with would you your
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    """'How many weeks of PTO?' -> ['many', 'weeks', 'pto']"""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 needs at least one chunk")
        self._chunks = chunks
        # Index the section path too: headings are short, precise descriptions of the text.
        self._bm25 = BM25Okapi([tokenize(c.embedding_text) for c in chunks])

    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        start = time.perf_counter()
        scores = self._bm25.get_scores(tokenize(query))
        candidates = [
            (float(score), chunk)
            for score, chunk in zip(scores, self._chunks, strict=True)
            if score > 0 and (category is None or chunk.category == category)
        ]
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        results = [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(candidates[:k], start=1)
        ]
        log.info(
            "retrieval.bm25",
            k=k,
            category=category,
            hits=len(results),
            ms=round((time.perf_counter() - start) * 1000),
        )
        return results
