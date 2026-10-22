"""Test doubles that stand in for real models: offline, instant and deterministic."""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from itertools import count
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from hrm.db.models import HolidaySource, RateSource
from hrm.integrations.currency import Conversion
from hrm.integrations.holidays import Holiday
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


class FakeReranker:
    """Scores a text by how many distinct query words it contains (stand-in for a
    cross-encoder). Records each call so tests can check what was reranked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        words = set(_WORD.findall(query.lower()))
        return [float(len(words & set(_WORD.findall(t.lower())))) for t in texts]


IQBAL_DAY = date(2026, 11, 9)  # a public holiday in Pakistan
FAKE_RATES = {"USD": Decimal(1), "EUR": Decimal("1.1225"), "PKR": Decimal("0.0036")}


class FakeHolidays:
    """Pakistan has Iqbal Day; every other country has no holidays."""

    def holidays(self, country: str, year: int) -> list[Holiday]:
        return [Holiday(IQBAL_DAY, "Iqbal Day", HolidaySource.MANUAL)] if country == "PK" else []


class FakeRates:
    """Fixed exchange rates; records each lookup."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, date]] = []

    def to_usd(self, amount_cents: int, currency: str, on: date) -> Conversion:
        self.calls.append((amount_cents, currency, on))
        rate = FAKE_RATES[currency]
        source = None if currency == "USD" else RateSource.API
        return Conversion(amount_cents, currency, int(amount_cents * rate), rate, on, source)


class ScriptedChatModel(BaseChatModel):
    """Replies with a fixed script of AI messages, one per call, and records what it saw.

    Lets agent-graph tests run the real graph and real tools without an LLM:
    the script plays the model's decisions ("call this tool", then "answer").
    """

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        if not self.script:
            raise AssertionError("The agent called the model more often than scripted.")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **_: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = [t.name for t in tools if isinstance(t, BaseTool)]
        return self


_call_ids = count(1)


def tool_call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, args=args, id=f"call_{next(_call_ids)}", type="tool_call")


def wants_tools(*calls: ToolCall) -> AIMessage:
    """A model turn that asks for tools instead of answering."""
    return AIMessage(content="", tool_calls=list(calls))
