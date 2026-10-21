"""Answer-evaluation scoring and the judge wrapper, with a fake judge model (no LLM calls)."""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from hrm.core.errors import LLMError
from hrm.evaluation.answers import (
    AnswerCase,
    CaseResult,
    Judge,
    JudgeVerdict,
    StatementCheck,
    answer_metrics,
    build_judge_messages,
)
from hrm.rag.generation import not_covered
from hrm.rag.models import Chunk, Citation, PolicyAnswer, RetrievedChunk

CASE = AnswerCase(
    id="a03",
    type="paraphrase",
    question="Do I need approval before vacation?",
    key_facts=["No formal approval is needed.", "Inform your manager 5 business days ahead."],
    must_not=["You must get approval."],
    expected_docs=["time-off-policy"],
)
UNANSWERABLE = AnswerCase(
    id="u01",
    type="unanswerable",
    question="Stock options?",
    key_facts=[],
    must_not=[],
    expected_docs=[],
)
CONTEXT = [
    RetrievedChunk(
        chunk=Chunk(
            chunk_id="time-off-policy:pto-ground-rules:0",
            doc_id="time-off-policy",
            category="time_off",
            audience="all",
            section_path=("PTO policy", "PTO ground rules"),
            text="You don't need formal PTO approval. Inform your Manager 5 business days ahead.",
            token_count=20,
        ),
        score=1.0,
        rank=1,
    )
]
ANSWER = PolicyAnswer(
    answer="Yes, you need approval [1].",
    found=True,
    citations=[
        Citation(
            number=1,
            chunk_id="time-off-policy:pto-ground-rules:0",
            doc_id="time-off-policy",
            section_path=("PTO policy", "PTO ground rules"),
        )
    ],
)


def verdict(
    facts: list[bool], forbidden: list[bool], unsupported: int = 0, miscited: int = 0
) -> JudgeVerdict:
    return JudgeVerdict(
        key_facts=[StatementCheck(statement=f"f{i}", holds=h) for i, h in enumerate(facts)],
        forbidden_claims=[
            StatementCheck(statement=f"n{i}", holds=h) for i, h in enumerate(forbidden)
        ],
        unsupported_claims=[f"claim {i}" for i in range(unsupported)],
        miscited_claims=[f"miscited {i}" for i in range(miscited)],
    )


def result(case: AnswerCase, answer: PolicyAnswer, v: JudgeVerdict | None) -> CaseResult:
    return CaseResult(case=case, answer=answer, verdict=v, seconds=2.0, tokens=1000)


# --- scoring -----------------------------------------------------------------------------


def test_correct_faithful_answer_scores_fully() -> None:
    r = result(CASE, ANSWER, verdict([True, True], [False]))

    assert r.correctness == 1.0
    assert r.fully_correct and r.faithful and r.citations_accurate


def test_forbidden_claim_breaks_fully_correct_even_with_all_facts() -> None:
    r = result(CASE, ANSWER, verdict([True, True], [True]))

    assert r.correctness == 1.0
    assert not r.fully_correct
    assert r.forbidden_asserted == ["n0"]


def test_partial_facts_unsupported_and_miscited_claims() -> None:
    r = result(CASE, ANSWER, verdict([True, False], [False], unsupported=1, miscited=1))

    assert r.correctness == 0.5
    assert not r.faithful
    assert not r.citations_accurate


def test_metrics_count_refusals_separately_for_answerable_and_unanswerable() -> None:
    results = [
        result(CASE, ANSWER, verdict([True, True], [False])),  # perfect
        result(CASE, not_covered(), None),  # false refusal: correctness 0
        result(UNANSWERABLE, not_covered(), None),  # correct refusal
        result(UNANSWERABLE, ANSWER, None),  # answered something it should have refused
    ]

    m = answer_metrics(results)

    assert m.answerable == 2
    assert m.correctness == 0.5
    assert m.fully_correct == 0.5
    assert m.faithfulness == 1.0  # only the judged answer counts
    assert m.false_refusals == 0.5
    assert m.refusal_accuracy == 0.5
    assert m.seconds == 2.0


# --- the judge wrapper ---------------------------------------------------------------------


class FakeJudgeModel:
    def __init__(self, reply: Any) -> None:
        self._reply = reply
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, input: list[BaseMessage]) -> Any:
        self.calls.append(input)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def test_judge_sees_documents_answer_facts_and_forbidden_claims() -> None:
    model = FakeJudgeModel(verdict([False, True], [True]))

    Judge(model).judge(CASE, ANSWER, CONTEXT)

    user = str(model.calls[0][1].content)
    assert '<document id="1" section="PTO policy > PTO ground rules">' in user
    assert "ANSWER:\nYes, you need approval [1]." in user
    assert "- Inform your manager 5 business days ahead." in user
    assert "FORBIDDEN CLAIMS:\n- You must get approval." in user


def test_judge_messages_handle_cases_without_forbidden_claims() -> None:
    case = CASE.model_copy(update={"must_not": []})

    user = str(build_judge_messages(case, ANSWER, CONTEXT)[1].content)

    assert user.endswith("FORBIDDEN CLAIMS:\n(none)")


@pytest.mark.parametrize(
    "reply",
    [
        verdict([True], [False]),  # one key fact missing from the verdict
        {"key_facts": "not a list"},
        TimeoutError("judge timed out"),
    ],
)
def test_bad_judge_replies_become_llm_errors(reply: Any) -> None:
    with pytest.raises(LLMError):
        Judge(FakeJudgeModel(reply)).judge(CASE, ANSWER, CONTEXT)
