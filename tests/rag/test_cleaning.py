import re

import pytest

from hrm.core.config import PROJECT_ROOT
from hrm.rag.cleaning import (
    clean_document,
    clean_headings,
    clean_markdown,
    links_to_text,
    normalize_bullets,
    normalize_typography,
    normalize_whitespace,
    strip_emphasis,
    strip_html,
    unescape_markdown,
)
from hrm.rag.loading import load_documents
from hrm.rag.models import SourceDocument

# --- one test per rule (examples taken from the real documents) -------------------------


def test_normalize_typography() -> None:
    curly = "Sourcegraph\u2019s \u201cqualified life event\u201d \u2013 yes"

    assert normalize_typography(curly) == 'Sourcegraph\'s "qualified life event" - yes'
    assert normalize_typography("16\u00a0weeks") == "16 weeks"  # non-breaking space


def test_strip_html_keeps_inner_text_and_decodes_entities() -> None:
    text = 'from <span style="text-decoration:underline;">2023-11-20</span>, &lt;30 days'

    assert strip_html(text) == "from 2023-11-20, <30 days"


def test_strip_html_leaves_autolinks_alone() -> None:
    assert strip_html("see <https://example.com>") == "see <https://example.com>"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("in your [Rippling](https://app.rippling.com/) account", "in your Rippling account"),
        ("email [people-ops@x.com](mailto:people-ops@x.com)", "email people-ops@x.com"),
        ("see [Sequoia](#benefits-support)", "see Sequoia"),
        ("the [policy](leave-of-absence.md)", "the policy"),
        ("by [Rippling](<https://a.com/x?y=%5D(https://a.com/z)>).", "by Rippling."),
    ],
)
def test_links_to_text(text: str, expected: str) -> None:
    assert links_to_text(text) == expected


def test_unescape_markdown() -> None:
    assert unescape_markdown(r"\*Exceptions\* \#1") == "*Exceptions* #1"


def test_strip_emphasis() -> None:
    assert strip_emphasis("**What:** Open _Enrollment_ **and** more") == (
        "What: Open Enrollment and more"
    )


def test_strip_emphasis_keeps_snake_case() -> None:
    assert strip_emphasis("set max_retries and log_level") == "set max_retries and log_level"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("## **2024 Open enrollment:**", "## 2024 Open enrollment"),
        (
            "#### **Q: How do I see my benefits selections?**",
            "#### Q: How do I see my benefits selections?",
        ),
        ("### Eligibility  ", "### Eligibility"),
    ],
)
def test_clean_headings(line: str, expected: str) -> None:
    assert clean_headings(line) == expected


def test_normalize_bullets_keeps_indentation() -> None:
    assert normalize_bullets("* top\n  * nested\n- already") == "- top\n  - nested\n- already"


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("\n\nA  \n\n\n\nB\t\n\n") == "A\n\nB\n"


# --- the whole pipeline on the real documents ---------------------------------------------

DOCS = load_documents(PROJECT_ROOT / "data" / "docs")


def headings(text: str) -> list[str]:
    return [line.split(" ", 1)[0] for line in text.splitlines() if re.match(r"#{1,6} ", line)]


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_real_documents_lose_noise_but_keep_structure(doc: SourceDocument) -> None:
    cleaned = clean_document(doc)

    assert headings(cleaned.text) == headings(doc.text)  # same heading levels, same order
    assert "](" not in cleaned.text  # no links left
    assert "**" not in cleaned.text  # no bold markers left
    assert "<span" not in cleaned.text
    assert "\u2019" not in cleaned.text  # curly apostrophe
    assert cleaned.content_hash == doc.content_hash  # hash stays the raw file's
    assert clean_markdown(cleaned.text) == cleaned.text  # cleaning twice changes nothing


def test_key_facts_survive_cleaning() -> None:
    text = {d.doc_id: clean_document(d).text for d in DOCS}

    assert "within 60 days of an eligible purchase" in text["home-office-and-stipends"]
    assert "more than $1,000 on any given physical item" in text["home-office-and-stipends"]
    assert "up to 16 weeks" in text["parental-leave"]
