"""Answer evaluation with an LLM judge, over evals/datasets/rag_answers.jsonl.

For each answered question, the judge (the fast model, temperature 0, structured output)
sees what the answer model saw (the numbered documents), the question, the answer, the
key facts a correct answer must contain and the claims it must not make. It reports:
  - which key facts are present           -> correctness
  - which forbidden claims are asserted   -> part of "fully correct"
  - claims supported by no document       -> faithfulness
  - claims whose [n] points at a document that does not support them -> citation accuracy
Refusals need no judge: unanswerable questions must get found=False, answerable ones must not.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from hrm.core.errors import LLMError
from hrm.rag.models import PolicyAnswer, RetrievedChunk

AnswerCaseType = Literal["fact", "paraphrase", "multi", "unanswerable"]


class AnswerCase(BaseModel):
    id: str
    type: AnswerCaseType
    question: str
    key_facts: list[str]
    must_not: list[str]
    expected_docs: list[str]

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_cases(path: Path) -> list[AnswerCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AnswerCase.model_validate(json.loads(line)) for line in lines if line.strip()]


# --- the judge ---------------------------------------------------------------------------


class StatementCheck(BaseModel):
    statement: str = Field(description="The statement exactly as given in the list.")
    holds: bool = Field(description="True if the answer states this (by meaning).")


class JudgeVerdict(BaseModel):
    key_facts: list[StatementCheck] = Field(
        description="One entry per KEY FACT, same order: does the answer state it?"
    )
    forbidden_claims: list[StatementCheck] = Field(
        description="One entry per FORBIDDEN CLAIM, same order: does the answer assert it?"
    )
    unsupported_claims: list[str] = Field(
        description="Claims in the answer that no provided document supports. Empty if none."
    )
    miscited_claims: list[str] = Field(
        description="Claims supported by some document, but whose [n] citation points to a "
        "document that does not support them. Empty if none."
    )


JUDGE_SYSTEM_PROMPT = """\
You are a strict, careful evaluator of an HR policy assistant. You receive the policy \
documents the assistant was given (numbered), a question, the assistant's answer with [n] \
citations, a list of KEY FACTS and a list of FORBIDDEN CLAIMS.

Judge by meaning, not exact wording.
- KEY FACTS: for each one, decide whether the answer states it. A fact that is only partly \
stated, or stated with a different number, amount, or condition, is NOT present.
- FORBIDDEN CLAIMS: for each one, decide whether the answer asserts it, directly or by \
clear implication. For example, answering "Yes" to "Do I need approval?" asserts that \
approval is needed.
- UNSUPPORTED CLAIMS: list every factual claim in the answer that none of the documents \
supports, including subtly changed ones ("at least" where a document says "up to").
- MISCITED CLAIMS: list claims that are supported by some document but whose citation [n] \
points to a document that does not support them.
Ignore style, tone, and suggestions to contact the People Team. Return the lists in order."""


class StructuredJudgeModel(Protocol):
    def invoke(self, input: list[BaseMessage]) -> Any: ...


class Judge:
    def __init__(self, model: StructuredJudgeModel) -> None:
        self._model = model

    def judge(
        self, case: AnswerCase, answer: PolicyAnswer, context: list[RetrievedChunk]
    ) -> JudgeVerdict:
        try:
            raw = self._model.invoke(build_judge_messages(case, answer, context))
            verdict = raw if isinstance(raw, JudgeVerdict) else JudgeVerdict.model_validate(raw)
        except ValidationError as exc:
            raise LLMError(f"The judge returned the wrong format: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"The judge model failed: {type(exc).__name__}: {exc}") from exc
        if len(verdict.key_facts) != len(case.key_facts) or len(verdict.forbidden_claims) != len(
            case.must_not
        ):
            raise LLMError(f"The judge skipped or added statements for {case.id}")
        return verdict


def build_judge_messages(
    case: AnswerCase, answer: PolicyAnswer, context: list[RetrievedChunk]
) -> list[BaseMessage]:
    documents = "\n\n".join(
        f'<document id="{n}" section="{" > ".join(r.chunk.section_path)}">\n{r.chunk.text}\n'
        "</document>"
        for n, r in enumerate(context, start=1)
    )
    facts = "\n".join(f"- {f}" for f in case.key_facts)
    forbidden = "\n".join(f"- {f}" for f in case.must_not) or "(none)"
    user = (
        f"<documents>\n{documents}\n</documents>\n\n"
        f"QUESTION: {case.question}\n\nANSWER:\n{answer.answer}\n\n"
        f"KEY FACTS:\n{facts}\n\nFORBIDDEN CLAIMS:\n{forbidden}"
    )
    return [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=user)]


# --- scoring -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    case: AnswerCase
    answer: PolicyAnswer
    verdict: JudgeVerdict | None  # None when nothing was judged (a refusal) or the judge failed
    seconds: float
    tokens: int
    error: str | None = None

    @property
    def refused(self) -> bool:
        return not self.answer.found

    @property
    def correctness(self) -> float:
        """Share of key facts present (0 for a refusal of an answerable question)."""
        if self.verdict is None or not self.case.key_facts:
            return 0.0
        return mean(check.holds for check in self.verdict.key_facts)

    @property
    def fully_correct(self) -> bool:
        return (
            self.verdict is not None
            and all(c.holds for c in self.verdict.key_facts)
            and not any(c.holds for c in self.verdict.forbidden_claims)
        )

    @property
    def faithful(self) -> bool:
        return self.verdict is not None and not self.verdict.unsupported_claims

    @property
    def citations_accurate(self) -> bool:
        return self.verdict is not None and not self.verdict.miscited_claims

    @property
    def forbidden_asserted(self) -> list[str]:
        if self.verdict is None:
            return []
        return [c.statement for c in self.verdict.forbidden_claims if c.holds]


@dataclass(frozen=True)
class AnswerMetrics:
    answerable: int
    correctness: float  # mean share of key facts, over all answerable questions
    fully_correct: float  # all key facts and no forbidden claim, over all answerable
    faithfulness: float  # over judged answers
    citation_accuracy: float  # over judged answers
    false_refusals: float  # answerable questions answered "not covered"
    refusal_accuracy: float  # unanswerable questions correctly answered "not covered"
    seconds: float
    tokens: float


def answer_metrics(results: list[CaseResult]) -> AnswerMetrics:
    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]
    judged = [r for r in answerable if r.verdict is not None]

    def share(values: list[bool] | list[float]) -> float:
        return float(mean(values)) if values else 0.0

    return AnswerMetrics(
        answerable=len(answerable),
        correctness=share([r.correctness for r in answerable]),
        fully_correct=share([r.fully_correct for r in answerable]),
        faithfulness=share([r.faithful for r in judged]),
        citation_accuracy=share([r.citations_accurate for r in judged]),
        false_refusals=share([r.refused for r in answerable]),
        refusal_accuracy=share([r.refused for r in unanswerable]),
        seconds=share([r.seconds for r in results]),
        tokens=share([float(r.tokens) for r in results]),
    )
