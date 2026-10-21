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
