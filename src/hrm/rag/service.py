"""RAGService: the one entry point to policy search and answers.

The CLI, the API and the agent tool all use this class, so the pipeline
(retrieve -> select context -> generate) is assembled in exactly one place. Its parts are
passed in, so tests use fakes; from_settings() builds the real ones once.
"""

from dataclasses import dataclass
from typing import Protocol

from hrm.core.config import Settings
from hrm.rag.context import select_context
from hrm.rag.embeddings import FastEmbedEmbedder
from hrm.rag.generation import AnswerGenerator
from hrm.rag.models import Category, PolicyAnswer, RetrievedChunk
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.factory import build_retriever
from hrm.rag.vectorstore import ChromaStore


class Generator(Protocol):
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> PolicyAnswer: ...


@dataclass(frozen=True)
class AnswerResult:
    answer: PolicyAnswer
    context: list[RetrievedChunk]  # what the answer model was given (empty: not called)


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
        context = select_context(
            self.search(question, category=category),
            max_tokens=self._max_context_tokens,
            min_score=self._min_score,
        )
        return AnswerResult(answer=self._generator.generate(question, context), context=context)
