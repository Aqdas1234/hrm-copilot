import re
from collections import Counter
from itertools import pairwise

import pytest

from hrm.core.config import PROJECT_ROOT
from hrm.rag.chunking import Section, chunk_document, split_sections
from hrm.rag.cleaning import clean_document
from hrm.rag.loading import load_documents
from hrm.rag.models import Chunk, SourceDocument


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


# --- size limits (R3.2) -------------------------------------------------------------------


def count_words(text: str) -> int:
    """Fake token counter for tests: one token per word. Offline and easy to reason about."""
    return len(text.split())


def chunk(
    doc: SourceDocument, max_tokens: int = 40, overlap: int = 8, min_tokens: int = 5
) -> list[Chunk]:
    return chunk_document(
        doc,
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        min_tokens=min_tokens,
        count_tokens=count_words,
    )


def words(n: int, word: str = "w") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def test_small_section_becomes_one_chunk_with_path_prefix() -> None:
    [c] = chunk(make_doc("# Title\n## A\n" + words(10) + "\n"))

    assert c.section_path == ("Title", "A")
    assert c.text == words(10)
    assert c.embedding_text == "Title > A\n\n" + words(10)
    assert c.token_count == count_words(c.embedding_text)
    assert (c.doc_id, c.category, c.audience) == ("doc", "conduct", "all")


def test_big_section_splits_at_paragraphs_with_overlap() -> None:
    paragraphs = [words(6, f"p{i}_") for i in range(10)]  # 10 paragraphs x 6 words
    chunks = chunk(make_doc("# Title\n## A\n" + "\n\n".join(paragraphs) + "\n"))

    assert len(chunks) > 1
    assert all(c.token_count <= 40 for c in chunks)
    for before, after in pairwise(chunks):
        last_paragraph = before.text.split("\n\n")[-1]
        assert after.text.startswith(last_paragraph)  # overlap: shared boundary paragraph


def test_one_huge_paragraph_falls_back_to_sentences_then_words() -> None:
    sentence = words(15) + "."
    no_punctuation = words(100)
    text = "# Title\n## A\n" + " ".join([sentence] * 4) + "\n\n" + no_punctuation + "\n"

    chunks = chunk(make_doc(text))

    assert all(c.token_count <= 40 for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_tiny_section_merges_with_next_sibling_and_keeps_its_heading() -> None:
    doc = make_doc("# Title\n## A\ntiny\n## B\n" + words(10) + "\n")

    [c] = chunk(doc)

    assert c.section_path == ("Title",)  # common parent of A and B
    assert c.text == "A\ntiny\n\nB\n" + words(10)


def test_tiny_section_does_not_merge_across_parents() -> None:
    doc = make_doc("# Title\n## A\n### A1\ntiny\n## B\n" + words(10) + "\n")

    assert [c.section_path for c in chunk(doc)] == [("Title", "A", "A1"), ("Title", "B")]


def test_faq_sections_are_never_merged() -> None:
    doc = make_doc("# Title\n## FAQ\n### Q. Short?\nYes.\n### Q. Other?\nNo.\n")

    assert [c.section_path[-1] for c in chunk(doc)] == ["Q. Short?", "Q. Other?"]


def test_question_paragraph_stays_with_its_answer_when_splitting() -> None:
    filler = "\n\n".join(words(9, f"f{i}_") for i in range(3))  # 27 words: fills most of a chunk
    text = f"# Title\n## Gifts\n{filler}\n\nQ: May I accept a gift?\n\n{words(8, 'answer')}\n"

    chunks = chunk(make_doc(text))

    holding_question = [c for c in chunks if "Q: May I accept a gift?" in c.text]
    assert holding_question
    assert all("answer0" in c.text for c in holding_question)


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_real_documents_chunks_fit_and_lose_no_words(doc: SourceDocument) -> None:
    chunks = chunk(doc, max_tokens=400, overlap=60, min_tokens=50)

    assert all(0 < c.token_count <= 400 for c in chunks)
    assert all(c.text.strip() for c in chunks)
    content = [ln for ln in doc.text.splitlines() if not IS_HEADING.match(ln)]
    from_doc = Counter(" ".join(content).split())
    from_chunks = Counter(" ".join(c.text for c in chunks).split())
    assert not from_doc - from_chunks  # every word of the document is in some chunk


# --- chunk IDs (R3.3) ---------------------------------------------------------------------


def test_chunk_ids_number_chunks_that_share_a_heading() -> None:
    paragraphs = "\n\n".join(words(6, f"p{i}_") for i in range(10))
    doc = make_doc(f"# Title\n## Big part\n{paragraphs}\n## Q. Done?\n{words(6)}\n")

    ids = [c.chunk_id for c in chunk(doc)]

    assert ids[0] == "doc:big-part:0"
    assert ids[1] == "doc:big-part:1"
    assert ids[-1] == "doc:q-done:0"


def test_same_heading_in_two_places_gets_distinct_ids() -> None:
    doc = make_doc(f"# Title\n## A\n### Notes\n{words(10)}\n## B\n### Notes\n{words(10)}\n")

    assert [c.chunk_id for c in chunk(doc)] == ["doc:notes:0", "doc:notes:1"]


def test_real_chunk_ids_are_unique_and_stable_across_runs() -> None:
    def all_ids() -> list[str]:
        return [
            c.chunk_id for d in DOCS for c in chunk(d, max_tokens=400, overlap=60, min_tokens=50)
        ]

    first = all_ids()

    assert len(first) == len(set(first))
    assert first == all_ids()
    assert all(i.startswith(f"{i.split(':')[0]}:") and i.count(":") == 2 for i in first)
