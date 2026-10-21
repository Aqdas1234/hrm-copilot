"""The evaluation datasets must agree with the documents, or eval failures would be fake.

For every expected item: the document exists, a section whose path contains the given heading
exists, and that section's text contains the evidence phrase (copied from the cleaned text).
"""

import json
from collections import Counter

import pytest

from hrm.core.config import PROJECT_ROOT
from hrm.rag.chunking import Section, split_sections
from hrm.rag.cleaning import clean_document
from hrm.rag.loading import load_documents

DATASET = PROJECT_ROOT / "evals" / "datasets" / "rag_retrieval.jsonl"
ROWS = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
SECTIONS: dict[str, list[Section]] = {
    doc.doc_id: split_sections(clean_document(doc))
    for doc in load_documents(PROJECT_ROOT / "data" / "docs")
}


def test_ids_are_unique_and_types_known() -> None:
    assert len({row["id"] for row in ROWS}) == len(ROWS)
    assert {row["type"] for row in ROWS} <= {"fact", "paraphrase", "multi", "unanswerable"}


def test_mix_of_question_types() -> None:
    types = Counter(row["type"] for row in ROWS)

    assert types["unanswerable"] == 10
    assert types["paraphrase"] >= 8
    assert types["multi"] >= 3
    assert len(ROWS) >= 40


def test_unanswerable_questions_expect_nothing_and_others_expect_something() -> None:
    for row in ROWS:
        assert bool(row["expected"]) == (row["type"] != "unanswerable"), row["id"]


@pytest.mark.parametrize(
    ("row_id", "item"),
    [(row["id"], item) for row in ROWS for item in row["expected"]],
)
def test_expected_section_exists_and_contains_the_evidence(
    row_id: str, item: dict[str, str]
) -> None:
    assert item["doc_id"] in SECTIONS, f"{row_id}: unknown doc {item['doc_id']}"
    matching = [s for s in SECTIONS[item["doc_id"]] if item["section"] in s.path]

    assert matching, f"{row_id}: no section {item['section']!r}"
    assert any(item["evidence"] in s.text for s in matching), (
        f"{row_id}: evidence {item['evidence']!r} not in section {item['section']!r}"
    )
