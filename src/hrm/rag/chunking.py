"""Split cleaned documents into chunks that follow the documents' own structure.

Step 1 (this module, so far): split on markdown headings into sections. Each section knows
its section path, e.g. ("Parental Leave Policy", "FAQ", "Q: Can I extend my leave?"),
which is used later for citations and as context in front of every chunk.
"""

import re
from dataclasses import dataclass

from hrm.rag.models import SourceDocument

_HEADING_LINE = re.compile(r"^(#{1,6}) (.+)$")
_QUESTION = re.compile(r"^Q[.:]\s")


@dataclass(frozen=True)
class Section:
    """The text under one heading, up to the next heading of any level."""

    doc_id: str
    path: tuple[str, ...]  # document title first, this section's heading last
    text: str

    @property
    def is_faq(self) -> bool:
        return bool(_QUESTION.match(self.path[-1]))


def split_sections(doc: SourceDocument) -> list[Section]:
    """Split a cleaned document on headings (# to ######). Headings without text are skipped
    as sections but still appear in the path of the sections below them."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) of the headings above the current line
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            path = tuple(title for _, title in stack) or (doc.title,)
            sections.append(Section(doc_id=doc.doc_id, path=path, text=text))
        body.clear()

    for line in doc.text.splitlines():
        match = _HEADING_LINE.match(line)
        if match is None:
            body.append(line)
            continue
        flush()
        level, title = len(match.group(1)), match.group(2).strip()
        # Pop headings at the same or a deeper level. Also pop a question: FAQ entries are
        # siblings, even where the source gives them inconsistent levels (#### Q then ##### Q).
        while stack and (stack[-1][0] >= level or _QUESTION.match(stack[-1][1])):
            stack.pop()
        stack.append((level, title))
    flush()
    return sections
