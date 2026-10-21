"""Data types that flow through the RAG pipeline. More (Chunk, RetrievedChunk, ...) join later."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["benefits", "conduct", "expenses", "parental_leave", "time_off"]
Audience = Literal["all", "us"]


class CatalogEntry(BaseModel):
    """One document as described in data/docs/catalog.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    doc_id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")  # stable slug used in citations
    category: Category
    audience: Audience


class SourceDocument(BaseModel):
    """A policy document as read from disk, before cleaning and chunking."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    category: Category
    audience: Audience
    source_path: str  # relative to the docs folder, so the index is portable between machines
    content_hash: str  # sha256 of the file bytes; tells ingestion whether a doc changed
    last_updated: date | None
    text: str


class Chunk(BaseModel):
    """A piece of a document small enough to embed, with the metadata needed to cite it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str  # "doc_id:section-slug:n", the same on every run for the same file
    doc_id: str
    category: Category
    audience: Audience
    section_path: tuple[str, ...]  # document title first, e.g. ("PTO policy", "Ground rules")
    text: str
    token_count: int  # of embedding_text, which is what the embedder sees

    @property
    def embedding_text(self) -> str:
        """The text that gets embedded: section path as context, then the chunk text."""
        return f"{' > '.join(self.section_path)}\n\n{self.text}"


class ScoredChunk(BaseModel):
    """A chunk returned by a search, with its similarity to the query (higher is closer)."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class RetrievedChunk(BaseModel):
    """A search result as the rest of the app sees it: chunk, score and 1-based rank."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float  # meaning depends on the retriever (cosine for dense, RRF for hybrid)
    rank: int
