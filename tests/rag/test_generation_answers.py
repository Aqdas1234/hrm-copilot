"""Structured answers and citation checks, with a fake model (no LLM calls)."""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from hrm.core.errors import LLMError
from hrm.rag.generation import NOT_COVERED_MESSAGE, AnswerGenerator, LLMAnswer
from hrm.rag.models import Chunk, RetrievedChunk


class FakeModel:
    """Returns a fixed reply (or raises) and records the messages it was given."""

    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, input: list[BaseMessage]) -> Any:
        self.calls.append(input)
        if self._error:
            raise self._error
        return self._reply


def retrieved(chunk_id: str, section: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        category="time_off",
        audience="all",
        section_path=("Policy", section),
        text=f"text of {section}",
        token_count=3,
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=1)


CHUNKS = [
    retrieved("time-off-policy:pto-ground-rules:0", "PTO ground rules"),
    retrieved("time-off-policy:submitting-pto:0", "Submitting PTO"),
    retrieved("benefits-overview:eligibility:0", "Eligibility"),
]


def generate(reply: Any, chunks: list[RetrievedChunk] = CHUNKS) -> Any:
    return AnswerGenerator(FakeModel(reply)).generate("question?", chunks)


def test_valid_citations_are_mapped_to_their_sources_in_order() -> None:
    answer = generate(
        LLMAnswer(answer="Give 5 days notice [1] and use Deel [2][1].", found=True, sources=[])
    )

    assert answer.found
    assert answer.answer == "Give 5 days notice [1] and use Deel [2][1]."
    assert [c.number for c in answer.citations] == [1, 2]  # unique, first-appearance order
    assert answer.citations[0].chunk_id == "time-off-policy:pto-ground-rules:0"
    assert answer.citations[1].section_path == ("Policy", "Submitting PTO")
    assert answer.removed_citations == []


def test_invented_citation_is_removed_from_the_text_and_reported() -> None:
    answer = generate(
        LLMAnswer(answer="Notice is 5 days [1]. Approval is needed [7].", found=True, sources=[])
    )

    assert answer.answer == "Notice is 5 days [1]. Approval is needed."
    assert [c.number for c in answer.citations] == [1]
    assert answer.removed_citations == [7]


def test_citation_zero_is_invalid_too() -> None:
    answer = generate(LLMAnswer(answer="Something [0] and more [3].", found=True, sources=[]))

    assert [c.number for c in answer.citations] == [3]
    assert answer.removed_citations == [0]


def test_not_found_returns_the_standard_reply() -> None:
    answer = generate(LLMAnswer(answer="Sorry, not in the documents.", found=False, sources=[]))

    assert not answer.found
    assert answer.answer == NOT_COVERED_MESSAGE
    assert answer.citations == []


def test_answer_without_any_valid_citation_is_treated_as_not_covered() -> None:
    answer = generate(
        LLMAnswer(answer="You get unlimited stock options [9].", found=True, sources=[])
    )

    assert not answer.found
    assert answer.answer == NOT_COVERED_MESSAGE
    assert answer.removed_citations == [9]


def test_uncited_answer_is_treated_as_not_covered() -> None:
    answer = generate(LLMAnswer(answer="Yes, that is allowed.", found=True, sources=[]))

    assert not answer.found


def test_no_chunks_means_no_model_call() -> None:
    model = FakeModel(LLMAnswer(answer="x [1]", found=True, sources=[]))

    answer = AnswerGenerator(model).generate("question?", [])

    assert not answer.found
    assert model.calls == []


def test_the_model_receives_the_prompt_with_the_documents() -> None:
    model = FakeModel(LLMAnswer(answer="ok [1]", found=True, sources=[]))

    AnswerGenerator(model).generate("How much notice?", CHUNKS)

    [messages] = model.calls
    assert 'id="3" doc="benefits-overview"' in str(messages[1].content)
    assert str(messages[1].content).endswith("Question: How much notice?")


def test_a_plain_dict_reply_is_accepted() -> None:
    answer = generate({"answer": "Use Deel [2].", "found": True, "sources": [2]})

    assert [c.number for c in answer.citations] == [2]


@pytest.mark.parametrize(
    "model",
    [
        FakeModel(error=TimeoutError("Groq timed out")),
        FakeModel(reply={"text": "missing fields"}),
    ],
)
def test_model_failures_become_llm_errors(model: FakeModel) -> None:
    with pytest.raises(LLMError):
        AnswerGenerator(model).generate("question?", CHUNKS)


def test_answer_without_inline_markers_is_kept_when_sources_are_valid() -> None:
    answer = generate(LLMAnswer(answer="Record PTO in Deel.", found=True, sources=[2]))

    assert answer.found
    assert answer.answer == "Record PTO in Deel."
    assert [c.number for c in answer.citations] == [2]


def test_inline_markers_come_first_then_extra_sources_without_repeats() -> None:
    answer = generate(LLMAnswer(answer="Notice [2] and Deel [1].", found=True, sources=[3, 2]))

    assert [c.number for c in answer.citations] == [2, 1, 3]


def test_invalid_sources_are_reported_and_ignored() -> None:
    answer = generate(LLMAnswer(answer="Notice is 5 days [1].", found=True, sources=[1, 9]))

    assert [c.number for c in answer.citations] == [1]
    assert answer.removed_citations == [9]


def test_no_valid_marker_or_source_is_still_not_covered() -> None:
    answer = generate(LLMAnswer(answer="Unlimited stock options.", found=True, sources=[8]))

    assert not answer.found
    assert answer.removed_citations == [8]
