import re

import pytest

from hrm.core.config import PROJECT_ROOT
from hrm.rag.chunking import Section, split_sections
from hrm.rag.cleaning import clean_document
from hrm.rag.loading import load_documents
from hrm.rag.models import SourceDocument


def make_doc(text: str) -> SourceDocument:
    return SourceDocument(
        doc_id="doc",
        title="Title",
        category="conduct",
        audience="all",
        source_path="doc.md",
        content_hash="0" * 64,
        last_updated=None,
        text=text,
    )


def paths(sections: list[Section]) -> list[tuple[str, ...]]:
    return [s.path for s in sections]


# --- section splitting on small examples --------------------------------------------------


def test_nested_headings_build_the_section_path() -> None:
    doc = make_doc("# Title\nintro\n## A\na text\n### A1\na1 text\n## B\nb text\n")

    sections = split_sections(doc)

    assert paths(sections) == [("Title",), ("Title", "A"), ("Title", "A", "A1"), ("Title", "B")]
    assert [s.text for s in sections] == ["intro", "a text", "a1 text", "b text"]


def test_skipped_heading_level_still_nests_correctly() -> None:
    doc = make_doc("# Title\n## A\n#### Deep\ndeep text\n### Mid\nmid text\n")

    assert paths(split_sections(doc)) == [("Title", "A", "Deep"), ("Title", "A", "Mid")]


def test_heading_without_text_is_only_part_of_the_path() -> None:
    doc = make_doc("# Title\n## FAQ\n### Q. One?\nAnswer one.\n")

    [section] = split_sections(doc)

    assert section.path == ("Title", "FAQ", "Q. One?")


def test_questions_are_siblings_even_with_inconsistent_levels() -> None:
    doc = make_doc("# Title\n## FAQ\n#### Q: First?\nA: one\n##### Q: Second?\nA: two\n")

    sections = split_sections(doc)

    assert paths(sections) == [("Title", "FAQ", "Q: First?"), ("Title", "FAQ", "Q: Second?")]
    assert all(s.is_faq for s in sections)


def test_text_before_any_heading_uses_the_document_title() -> None:
    [section] = split_sections(make_doc("no heading here\n"))

    assert section.path == ("Title",)


# --- the real documents --------------------------------------------------------------------

DOCS = [clean_document(d) for d in load_documents(PROJECT_ROOT / "data" / "docs")]
IS_HEADING = re.compile(r"^#{1,6} ")


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_no_text_is_lost_or_duplicated(doc: SourceDocument) -> None:
    def content_lines(text: str) -> list[str]:
        return [ln for ln in text.splitlines() if ln.strip() and not IS_HEADING.match(ln)]

    from_sections = [ln for s in split_sections(doc) for ln in content_lines(s.text)]

    assert from_sections == content_lines(doc.text)


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_no_section_crosses_a_heading(doc: SourceDocument) -> None:
    for section in split_sections(doc):
        assert not any(IS_HEADING.match(ln) for ln in section.text.splitlines()), section.path
        assert section.path[0] == doc.title


def test_every_faq_question_is_one_section_with_its_answer() -> None:
    question_headings = sum(len(re.findall(r"(?m)^#{1,6} Q[.:] ", doc.text)) for doc in DOCS)
    faq_sections = [s for doc in DOCS for s in split_sections(doc) if s.is_faq]

    assert len(faq_sections) == question_headings == 40
    for section in faq_sections:
        assert section.text  # the answer is in the same section as its question
        assert sum(bool(re.match(r"Q[.:] ", p)) for p in section.path) == 1  # never nested
