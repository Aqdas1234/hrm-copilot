"""Normalize the policy markdown before chunking.

Each rule is a small pure function (str -> str) with its own test. They are applied in
order by clean_markdown(). Headings are kept (chunking splits on them); only noise that
hurts retrieval is removed: URLs, HTML, emphasis markers, typographic characters.
Rules are deterministic, so the same input always gives the same chunks.
"""

import html
import re
import unicodedata
from collections.abc import Callable

from hrm.rag.models import SourceDocument

# Curly quotes and en/em dashes, written as escapes so the lookalike characters are visible.
_TYPOGRAPHY = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)

_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(\s[^<>]*)?/?>")
# [text](<url>) first: the angle-bracket form may contain ")" inside the URL.
_LINK_ANGLE = re.compile(r"\[([^\]]*)\]\(<[^>]*>\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ESCAPE = re.compile(r"\\([-*_#\[\]()`])")
_BOLD = re.compile(r"\*\*(.*?)\*\*")
_ITALIC = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")  # not snake_case words
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
_STAR_BULLET = re.compile(r"^(\s*)\* ", re.MULTILINE)
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def normalize_typography(text: str) -> str:
    """NFKC (e.g. non-breaking space -> space), then curly quotes and dashes -> ASCII."""
    return unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)


def strip_html(text: str) -> str:
    """<span style=...>2023-11-20</span> -> 2023-11-20; &lt; -> <. Autolinks are not tags."""
    return html.unescape(_HTML_TAG.sub("", text))


def links_to_text(text: str) -> str:
    """[Rippling](https://...) -> Rippling. URLs are noise for retrieval and may be internal."""
    return _LINK.sub(r"\1", _LINK_ANGLE.sub(r"\1", text))


def unescape_markdown(text: str) -> str:
    r"""\*\* -> ** so the emphasis rule can remove it."""
    return _ESCAPE.sub(r"\1", text)


def strip_emphasis(text: str) -> str:
    """**What:** -> What:  and  _Note_ -> Note."""
    return _ITALIC.sub(r"\1", _BOLD.sub(r"\1", text))


def clean_headings(text: str) -> str:
    """## **2024 Open enrollment:** -> ## 2024 Open enrollment (clean section names)."""

    def fix(match: re.Match[str]) -> str:
        title = match.group(2).replace("**", "").strip().rstrip(":").strip()
        return f"{match.group(1)} {title}"

    return _HEADING.sub(fix, text)


def normalize_bullets(text: str) -> str:
    """'* item' -> '- item', keeping indentation (nesting)."""
    return _STAR_BULLET.sub(r"\1- ", text)


def normalize_whitespace(text: str) -> str:
    """No trailing spaces, at most one blank line in a row, exactly one final newline."""
    text = _TRAILING_SPACE.sub("", text)
    return _BLANK_RUNS.sub("\n\n", text).strip() + "\n"


# Order matters: unescape before emphasis; headings after emphasis; whitespace last.
RULES: tuple[Callable[[str], str], ...] = (
    normalize_typography,
    strip_html,
    links_to_text,
    unescape_markdown,
    strip_emphasis,
    clean_headings,
    normalize_bullets,
    normalize_whitespace,
)


def clean_markdown(text: str) -> str:
    """Apply every rule in order."""
    for rule in RULES:
        text = rule(text)
    return text


def clean_document(doc: SourceDocument) -> SourceDocument:
    """Return a copy with cleaned text. content_hash stays the hash of the raw file."""
    return doc.model_copy(update={"text": clean_markdown(doc.text)})
