"""RAGService: the one entry point to policy search and answers.

The CLI, the API and the agent tool all use this class, so the pipeline
(retrieve -> select context -> generate) is assembled in exactly one place. Its parts are
passed in, so tests use fakes; from_settings() builds the real ones once.
"""

import time
from dataclasses import dataclass, field
from typing import Protocol

from hrm.core.config import Settings
from hrm.core.logging import get_logger
from hrm.rag.context import select_context
from hrm.rag.embeddings import FastEmbedEmbedder
from hrm.rag.generation import AnswerGenerator
from hrm.rag.models import Category, PolicyAnswer, RetrievedChunk
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.factory import build_retriever
from hrm.rag.vectorstore import ChromaStore

log = get_logger(__name__)


class Generator(Protocol):
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> PolicyAnswer: ...


@dataclass(frozen=True)
class StepTimings:
    """Milliseconds per pipeline step (retrieve includes fusion and reranking)."""

    retrieve_ms: int
    context_ms: int
    generate_ms: int
    total_ms: int


@dataclass(frozen=True)
class AnswerResult:
    answer: PolicyAnswer
    context: list[RetrievedChunk]  # what the answer model was given (empty: not called)
    timings: StepTimings = field(default_factory=lambda: StepTimings(0, 0, 0, 0))


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        top_k: int = 5,
        max_context_tokens: int = 2500,
        min_score: float | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._max_context_tokens = max_context_tokens
        self._min_score = min_score

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGService":
        retriever = build_retriever(
            settings,
            FastEmbedEmbedder.from_settings(settings),
            ChromaStore.from_settings(settings),
        )
        return cls(
            retriever,
            AnswerGenerator.from_settings(settings),
            top_k=settings.top_k,
            max_context_tokens=settings.max_context_tokens,
            # The threshold is calibrated on reranker scores; other scores use other scales.
            min_score=settings.min_rerank_score if settings.rerank else None,
        )

    def search(
        self, query: str, k: int | None = None, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        return self._retriever.retrieve(query, k or self._top_k, category=category)

    def answer(self, question: str, *, category: Category | None = None) -> AnswerResult:
        start = time.perf_counter()
        retrieved = self.search(question, category=category)
        after_retrieve = time.perf_counter()
        context = select_context(
            retrieved, max_tokens=self._max_context_tokens, min_score=self._min_score
        )
        after_context = time.perf_counter()
        answer = self._generator.generate(question, context)
        end = time.perf_counter()

        timings = StepTimings(
            retrieve_ms=_ms(start, after_retrieve),
            context_ms=_ms(after_retrieve, after_context),
            generate_ms=_ms(after_context, end),
            total_ms=_ms(start, end),
        )
        # One summary line per question: easy to filter, chart and alert on.
        log.info(
            "rag.answer",
            found=answer.found,
            citations=len(answer.citations),
            context_chunks=len(context),
            top_score=round(retrieved[0].score, 3) if retrieved else None,
            retrieve_ms=timings.retrieve_ms,
            generate_ms=timings.generate_ms,
            total_ms=timings.total_ms,
        )
        return AnswerResult(answer=answer, context=context, timings=timings)


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)
