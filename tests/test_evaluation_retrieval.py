import pytest

from hrm.evaluation.retrieval import (
    EvalQuestion,
    ExpectedItem,
    QuestionResult,
    breakdown,
    is_relevant,
    metrics,
)
from hrm.rag.models import Chunk, RetrievedChunk


def chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:x:{abs(hash(text)) % 1000}",
        doc_id=doc_id,
        category="conduct",
        audience="all",
        section_path=("Title",),
        text=text,
        token_count=1,
    )


def ranked(*chunks: Chunk) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=c, score=1.0 - i / 10, rank=i + 1) for i, c in enumerate(chunks)]


def question(qid: str, *items: tuple[str, str], qtype: str = "fact") -> EvalQuestion:
    return EvalQuestion.model_validate(
        {
            "id": qid,
            "type": qtype,
            "question": "?",
            "expected": [{"doc_id": d, "section": "S", "evidence": e} for d, e in items],
        }
    )


RELEVANT = chunk("pto", "We encourage a minimum of 30 days total paid time off.")
WRONG_DOC = chunk("other", "a minimum of 30 days total paid time off")  # same words, wrong doc
SAME_DOC_NO_FACT = chunk("pto", "Record your PTO in Deel.")


def test_relevance_needs_right_document_and_the_evidence() -> None:
    item = ExpectedItem(doc_id="pto", section="S", evidence="minimum of 30 days")

    assert is_relevant(item, RELEVANT)
    assert not is_relevant(item, WRONG_DOC)
    assert not is_relevant(item, SAME_DOC_NO_FACT)


def test_evidence_matches_across_line_breaks() -> None:
    item = ExpectedItem(doc_id="pto", section="S", evidence="minimum of 30 days")

    assert is_relevant(item, chunk("pto", "a minimum of\n30 days"))


def test_rank_hits_and_reciprocal_rank() -> None:
    q = question("q1", ("pto", "minimum of 30 days"))
    result = QuestionResult(q, ranked(SAME_DOC_NO_FACT, WRONG_DOC, RELEVANT), ms=1.0)

    assert result.first_relevant_rank == 3
    assert not result.hit_at(1)
    assert result.hit_at(5)
    assert result.reciprocal_rank == pytest.approx(1 / 3)


def test_no_relevant_result_scores_zero() -> None:
    q = question("q1", ("pto", "minimum of 30 days"))
    result = QuestionResult(q, ranked(WRONG_DOC), ms=1.0)

    assert result.first_relevant_rank is None
    assert result.reciprocal_rank == 0.0
    assert result.recall_at_5 == 0.0


def test_recall_counts_each_expected_item() -> None:
    q = question("q1", ("pto", "minimum of 30 days"), ("pto", "in Deel"), qtype="multi")

    only_one = QuestionResult(q, ranked(RELEVANT), ms=1.0)
    both = QuestionResult(q, ranked(RELEVANT, SAME_DOC_NO_FACT), ms=1.0)

    assert only_one.recall_at_5 == 0.5
    assert both.recall_at_5 == 1.0


def test_metrics_skip_unanswerable_questions_and_average_the_rest() -> None:
    hit = QuestionResult(question("a", ("pto", "minimum of 30 days")), ranked(RELEVANT), 1.0)
    miss = QuestionResult(question("b", ("pto", "minimum of 30 days")), ranked(WRONG_DOC), 1.0)
    unanswerable = QuestionResult(question("u", qtype="unanswerable"), ranked(WRONG_DOC), 1.0)

    m = metrics([hit, miss, unanswerable])

    assert m.n == 2
    assert m.hit_at_1 == 0.5
    assert m.mrr == 0.5


def test_breakdown_by_type_and_by_document() -> None:
    fact = QuestionResult(question("a", ("pto", "minimum of 30 days")), ranked(RELEVANT), 1.0)
    para = QuestionResult(question("b", ("gifts", "x"), qtype="paraphrase"), ranked(WRONG_DOC), 1.0)

    assert breakdown([fact, para], by="type")["fact"].hit_at_1 == 1.0
    assert breakdown([fact, para], by="type")["paraphrase"].hit_at_1 == 0.0
    assert set(breakdown([fact, para], by="doc")) == {"pto", "gifts"}
