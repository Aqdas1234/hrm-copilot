"""Split cleaned documents into chunks that follow the documents' own structure.

1. split_sections: split on markdown headings. Each section knows its section path, e.g.
   ("Parental Leave Policy", "FAQ", "Q: Can I extend my leave?"), used for citations and
   as context in front of every chunk.
2. chunk_document: bring sections within token limits. Tiny sections merge into a sibling,
   big ones split at paragraph > line > sentence > word boundaries with some overlap.
   FAQ questions are never merged, and a "Q:" paragraph is never separated from its answer.

Sizes are counted by a TokenCounter passed in, so tests can use a fake one offline.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from hrm.rag.models import Chunk, SourceDocument

TokenCounter = Callable[[str], int]

_HEADING_LINE = re.compile(r"^(#{1,6}) (.+)$")
_QUESTION = re.compile(r"^Q[.:]\s")
_QUESTION_PARAGRAPH = re.compile(r"^(- )?Q[.:]\s")  # a question written as body text
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def tiktoken_counter() -> TokenCounter:
    """Count tokens with tiktoken (cl100k_base). Downloads the vocabulary once, then cached."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text))


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


def chunk_document(
    doc: SourceDocument,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    """Turn a cleaned document into chunks whose embedding_text is at most max_tokens."""
    sections = _merge_tiny(split_sections(doc), max_tokens, min_tokens, count_tokens)
    chunks: list[Chunk] = []
    for section in sections:
        budget = max_tokens - count_tokens(_prefix(section.path))
        if budget <= overlap_tokens:
            raise ValueError(f"Section path too long for max_tokens={max_tokens}: {section.path}")
        for text in _split_text(section.text, budget, overlap_tokens, count_tokens):
            chunk = Chunk(
                doc_id=doc.doc_id,
                category=doc.category,
                audience=doc.audience,
                section_path=section.path,
                text=text,
                token_count=0,
            )
            chunks.append(
                chunk.model_copy(update={"token_count": count_tokens(chunk.embedding_text)})
            )
    return chunks


def _prefix(path: tuple[str, ...]) -> str:
    return " > ".join(path) + "\n\n"  # must match Chunk.embedding_text


# --- merging tiny sections -----------------------------------------------------------------


def _merge_tiny(
    sections: list[Section], max_tokens: int, min_tokens: int, count: TokenCounter
) -> list[Section]:
    """Merge a section smaller than min_tokens into the next section, or else the previous one,
    if they share a parent and the result still fits. FAQ sections are never merged."""

    def is_tiny(section: Section) -> bool:
        return count(section.text) < min_tokens and not section.is_faq

    def try_merge(first: Section, second: Section, tiny: Section) -> Section | None:
        if first.is_faq or second.is_faq:
            return None
        common = _common_prefix(first.path, second.path)
        if len(common) < len(tiny.path) - 1:  # only siblings, or a parent with its child
            return None
        merged = Section(
            doc_id=first.doc_id,
            path=common,
            text=f"{_labelled(first, common)}\n\n{_labelled(second, common)}",
        )
        return merged if count(_prefix(merged.path) + merged.text) <= max_tokens else None

    result: list[Section] = []
    i = 0
    while i < len(sections):
        current = sections[i]
        while is_tiny(current) and i + 1 < len(sections):
            merged = try_merge(current, sections[i + 1], tiny=current)
            if merged is None:
                break
            current, i = merged, i + 1
        if is_tiny(current) and result:
            merged = try_merge(result[-1], current, tiny=current)
            if merged is not None:
                result[-1] = merged
                i += 1
                continue
        result.append(current)
        i += 1
    return result


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _labelled(section: Section, common: tuple[str, ...]) -> str:
    """Keep the headings that the merged path no longer shows, as a first line of the text."""
    rest = section.path[len(common) :]
    return f"{' > '.join(rest)}\n{section.text}" if rest else section.text


# --- splitting big sections ----------------------------------------------------------------

# A piece of text plus the separator that joins it to the piece before it.
_Atom = tuple[str, str]


def _split_text(text: str, budget: int, overlap: int, count: TokenCounter) -> list[str]:
    """Pack text into pieces of at most `budget` tokens; consecutive pieces share up to
    `overlap` tokens so a fact on the boundary is not lost."""
    if count(text) <= budget:
        return [text]
    atoms = _keep_questions_with_answers(_atoms(text, budget, count), budget, count)

    pieces: list[list[_Atom]] = []
    current: list[_Atom] = []
    for atom in atoms:
        if current and count(_join([*current, atom])) > budget:
            pieces.append(current)
            current = _tail(current, overlap, count)
            if count(_join([*current, atom])) > budget:
                current = []
        current.append(atom)
    if current:
        pieces.append(current)
    return [_join(piece) for piece in pieces]


def _atoms(text: str, budget: int, count: TokenCounter) -> list[_Atom]:
    """Break text into the largest units that fit: paragraphs, else lines, else sentences,
    else runs of words."""
    atoms: list[_Atom] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        if count(paragraph) <= budget:
            atoms.append(("\n\n", paragraph))
            continue
        for line in paragraph.splitlines():
            if count(line) <= budget:
                atoms.append(("\n", line))
                continue
            for sentence in _SENTENCE_END.split(line):
                if count(sentence) <= budget:
                    atoms.append((" ", sentence))
                else:
                    atoms.extend((" ", words) for words in _word_runs(sentence, budget, count))
    return atoms


def _word_runs(text: str, budget: int, count: TokenCounter) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and count(" ".join([*current, word])) > budget:
            runs.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        runs.append(" ".join(current))
    return runs


def _keep_questions_with_answers(
    atoms: list[_Atom], budget: int, count: TokenCounter
) -> list[_Atom]:
    """Glue a 'Q: ...' paragraph to the paragraph after it (its answer), if they fit together."""
    result: list[_Atom] = []
    previous_is_question = False
    for atom in atoms:
        if previous_is_question:
            glued = (result[-1][0], result[-1][1] + atom[0] + atom[1])
            if count(glued[1]) <= budget:
                result[-1] = glued
                previous_is_question = False  # glue once: question + its answer
                continue
        result.append(atom)
        previous_is_question = bool(_QUESTION_PARAGRAPH.match(atom[1]))
    return result


def _tail(atoms: list[_Atom], overlap: int, count: TokenCounter) -> list[_Atom]:
    """The last atoms of a piece that together fit in `overlap` tokens."""
    tail: list[_Atom] = []
    for atom in reversed(atoms):
        if count(_join([atom, *tail])) > overlap:
            break
        tail.insert(0, atom)
    return tail


def _join(atoms: list[_Atom]) -> str:
    return atoms[0][1] + "".join(sep + text for sep, text in atoms[1:])
