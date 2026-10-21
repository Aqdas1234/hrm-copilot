"""Retrievers: find the chunks most relevant to a question.

Every retriever (dense now; BM25, hybrid and reranked later) has the same interface,
so the answer step and the evaluations can swap one for another.
"""

from typing import Protocol

from hrm.rag.models import Category, RetrievedChunk


class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        """The k most relevant chunks for the query, best first (rank 1 = best)."""
        ...
