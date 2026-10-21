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

    # --- Chunking (sizes in tokens) ---
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=60, ge=0)
    chunk_min_size: int = Field(default=50, ge=0)  # smaller sections merge into a neighbor

    # --- Retrieval ---
    retrieve_k: int = Field(default=20, gt=0)  # candidates fetched before reranking
    top_k: int = Field(default=5, gt=0)  # chunks passed to the answer model

    # --- Logging ---
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"

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
