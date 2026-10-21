"""Retrieval metrics over evals/datasets/rag_retrieval.jsonl.

A retrieved chunk is relevant to an expected item when it comes from the expected document
and contains the item's evidence phrase. Being from the right document is not enough: the
chunk must hold the fact the answer needs.

Metrics (answerable questions only):
  hit@1, hit@5  share of questions with a relevant chunk at rank 1 / within the top 5
  MRR@10        mean of 1/rank of the first relevant chunk (0 if none in the top 10)
  recall@5      share of a question's expected items found in the top 5, averaged
Unanswerable questions have nothing to find; their top-1 score is reported instead, to
compare with answerable ones when choosing a relevance threshold.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel

from hrm.rag.models import Chunk, RetrievedChunk

QuestionType = Literal["fact", "paraphrase", "multi", "unanswerable"]
MRR_DEPTH = 10


class ExpectedItem(BaseModel):
    doc_id: str
    section: str
    evidence: str


class EvalQuestion(BaseModel):
    id: str
    type: QuestionType
    question: str
    expected: list[ExpectedItem]

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


def load_dataset(path: Path) -> list[EvalQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion.model_validate(json.loads(line)) for line in lines if line.strip()]


def is_relevant(item: ExpectedItem, chunk: Chunk) -> bool:
    # Compare with whitespace collapsed: chunking may join lines differently than the source.
    return chunk.doc_id == item.doc_id and _squash(item.evidence) in _squash(chunk.text)


@dataclass(frozen=True)
class QuestionResult:
    question: EvalQuestion
    results: list[RetrievedChunk]
    ms: float

    @property
    def first_relevant_rank(self) -> int | None:
        for r in self.results[:MRR_DEPTH]:
            if any(is_relevant(item, r.chunk) for item in self.question.expected):
                return r.rank
        return None

    def hit_at(self, k: int) -> bool:
        rank = self.first_relevant_rank
        return rank is not None and rank <= k

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank
        return 1.0 / rank if rank else 0.0

    @property
    def recall_at_5(self) -> float:
        top = [r.chunk for r in self.results[:5]]
        found = sum(any(is_relevant(item, c) for c in top) for item in self.question.expected)
        return found / len(self.question.expected)

    @property
    def top_score(self) -> float | None:
        return self.results[0].score if self.results else None


@dataclass(frozen=True)
class Metrics:
    n: int
    hit_at_1: float
    hit_at_5: float
    mrr: float
    recall_at_5: float


def metrics(results: list[QuestionResult]) -> Metrics:
    answerable = [r for r in results if r.question.answerable]
    if not answerable:
        return Metrics(n=0, hit_at_1=0.0, hit_at_5=0.0, mrr=0.0, recall_at_5=0.0)
    return Metrics(
        n=len(answerable),
        hit_at_1=mean(r.hit_at(1) for r in answerable),
        hit_at_5=mean(r.hit_at(5) for r in answerable),
        mrr=mean(r.reciprocal_rank for r in answerable),
        recall_at_5=mean(r.recall_at_5 for r in answerable),
    )


def breakdown(results: list[QuestionResult], by: Literal["type", "doc"]) -> dict[str, Metrics]:
    """Metrics per question type, or per expected document (multi questions count for each)."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        if not r.question.answerable:
            continue
        keys = [r.question.type] if by == "type" else {e.doc_id for e in r.question.expected}
        for key in keys:
            groups[key].append(r)
    return {key: metrics(group) for key, group in sorted(groups.items())}


def _squash(text: str) -> str:
    return " ".join(text.split())
