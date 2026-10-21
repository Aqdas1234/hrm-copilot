"""Choose what goes into the answer prompt.

Three guards, applied in order:
1. Relevance: if even the best chunk scores below `min_score`, nothing is relevant enough;
   return no context, so the caller answers "not covered" without calling the LLM.
2. Overlap: neighbouring pieces of one split section repeat ~60 tokens (chunk overlap);
   the repeated text is trimmed from the later piece so the model doesn't read it twice.
3. Budget: chunks are added in rank order until `max_tokens` would be exceeded
   (the best chunk is always kept).
"""

from hrm.core.logging import get_logger
from hrm.rag.models import RetrievedChunk

log = get_logger(__name__)


def select_context(
    chunks: list[RetrievedChunk], *, max_tokens: int, min_score: float | None = None
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    if min_score is not None and chunks[0].score < min_score:
        log.info("context.below_threshold", top_score=round(chunks[0].score, 2), min=min_score)
        return []
    return _within_budget(_trim_overlaps(chunks), max_tokens)


def _trim_overlaps(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """For pieces n and n+1 of the same section, remove from n+1 the text it repeats from n."""
    by_id = {r.chunk.chunk_id: r for r in chunks}
    trimmed: dict[str, RetrievedChunk] = {}
    for r in chunks:
        previous = by_id.get(_previous_piece_id(r.chunk.chunk_id))
        if previous is None or previous.chunk.section_path != r.chunk.section_path:
            continue
        overlap = _overlap(previous.chunk.text, r.chunk.text)
        if overlap:
            text = r.chunk.text[overlap:].lstrip()
            trimmed[r.chunk.chunk_id] = r.model_copy(
                update={"chunk": r.chunk.model_copy(update={"text": text})}
            )
    return [trimmed.get(r.chunk.chunk_id, r) for r in chunks]


def _previous_piece_id(chunk_id: str) -> str:
    """'doc:section:3' -> 'doc:section:2' (no previous piece for :0)."""
    base, _, number = chunk_id.rpartition(":")
    return f"{base}:{int(number) - 1}" if number.isdigit() and int(number) > 0 else ""


def _overlap(earlier: str, later: str) -> int:
    """Length of the longest end of `earlier` that is also the start of `later`."""
    for size in range(min(len(earlier), len(later)), 0, -1):
        if earlier.endswith(later[:size]):
            return size
    return 0


def _within_budget(chunks: list[RetrievedChunk], max_tokens: int) -> list[RetrievedChunk]:
    kept = [chunks[0]]
    used = chunks[0].chunk.token_count
    for r in chunks[1:]:
        if used + r.chunk.token_count > max_tokens:
            log.info("context.budget_reached", kept=len(kept), dropped=len(chunks) - len(kept))
            break
        kept.append(r)
        used += r.chunk.token_count
    return kept
