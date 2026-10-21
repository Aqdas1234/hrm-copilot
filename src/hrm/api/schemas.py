"""Request and response bodies of the HTTP API.

These are separate from the internal models on purpose: the API is a public contract,
so internal fields can change without breaking clients.
"""

from pydantic import BaseModel, Field

from hrm.rag.models import Category


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    category: Category | None = None


class SearchHit(BaseModel):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    section: str
    text: str


class SearchResponse(BaseModel):
    results: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    category: Category | None = None


class CitationOut(BaseModel):
    number: int
    doc_id: str
    section: str
    chunk_id: str


class Timings(BaseModel):
    retrieve_ms: int
    generate_ms: int
    total_ms: int


class AskResponse(BaseModel):
    answer: str
    found: bool  # False: the policies do not cover the question
    citations: list[CitationOut]
    timings: Timings


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
