"""Application settings, read from environment variables (prefix HRM_) and an optional .env file.

This is the only place that knows about paths, model names and tuning numbers.
Other modules receive a Settings object instead of reading the environment themselves,
which keeps them easy to test with different values.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["console", "json"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HRM_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    # Optional here so commands that don't need the LLM (ingest, search) work without a key.
    # hrm.core.llm raises ConfigurationError if it is missing when a model is requested.
    groq_api_key: SecretStr | None = None
    answer_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # --- Paths ---
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    model_cache_dir: Path = PROJECT_ROOT / ".cache" / "models"  # downloaded local models

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge's instruction for search queries (not for passages). "" turns it off.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=64, gt=0)

    # --- Chunking (sizes in tokens) ---
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=60, ge=0)
    chunk_min_size: int = Field(default=50, ge=0)  # smaller sections merge into a neighbor

    # --- Retrieval ---
    # Defaults chosen by measurement: see "Retrieval design" in README.md and evals/RESULTS.md.
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rrf_k: int = Field(default=60, gt=0)  # Reciprocal Rank Fusion constant (hybrid mode)
    retrieve_k: int = Field(default=20, gt=0)  # candidates fetched before fusion/reranking
    rerank: bool = True  # re-score the retrieve_k candidates with a cross-encoder
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=5, gt=0)  # chunks passed to the answer model

    # --- Answer context ---
    max_context_tokens: int = Field(default=2500, gt=0)  # budget for documents in the prompt
    # Reranker score below which the best chunk is "clearly unrelated": answer "not covered"
    # without calling the LLM. Chosen conservatively from evals (lowest answerable: -6.7).
    # Only applied when rerank is on (other retrievers' scores are on different scales).
    min_rerank_score: float = -8.0

    # --- Logging ---
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"

    # --- HR database (Part B) ---
    # SQLAlchemy URL; a secret because it contains the password. Tests use SQLite in memory.
    database_url: SecretStr = SecretStr("mysql+pymysql://hrm:hrm@127.0.0.1:3306/hrm")
    # Signs login tokens (JWT). No default on purpose: without it, login is disabled (503).
    # Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"
    jwt_secret: SecretStr | None = None
    jwt_ttl_minutes: int = Field(default=60, gt=0)
    # Password of the scripted demo users created by `hrm db seed` (demo data only).
    demo_password: SecretStr = SecretStr("hrm-demo-123")

    # --- External services (Part B) ---
    holidays_api_url: str = "https://date.nager.at/api/v3"
    http_timeout_s: float = Field(default=10.0, gt=0)
    http_retries: int = Field(default=3, ge=1)  # attempts, including the first

    # --- UI ---
    api_url: str = "http://127.0.0.1:8000"  # where the Streamlit page finds the API

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chunk_min_size >= self.chunk_size:
            raise ValueError("chunk_min_size must be smaller than chunk_size")
        if self.top_k > self.retrieve_k:
            raise ValueError("top_k must not be larger than retrieve_k")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once."""
    return Settings()
