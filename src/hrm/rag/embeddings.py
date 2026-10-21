"""Turn text into vectors for semantic search.

Embedder is a Protocol (an interface): the rest of the code depends only on these methods,
so the real model can be swapped (bge-base, an API model) and tests can use a fake.

Documents and queries are embedded by different methods on purpose: retrieval models such
as bge expect an instruction in front of queries but not in front of passages.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

from hrm.core.config import Settings
from hrm.core.logging import get_logger

log = get_logger(__name__)

Vector = list[float]


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Identifies the model; stored with the index so a model change forces a rebuild."""
        ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class FastEmbedEmbedder:
    """Local ONNX embedding model via fastembed (default: BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        batch_size: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        log.info("embedder.loading", model=model_name)
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._dimension = len(self.embed_query("dimension probe"))

    @classmethod
    def from_settings(cls, settings: Settings) -> "FastEmbedEmbedder":
        return cls(
            settings.embedding_model,
            query_prefix=settings.embedding_query_prefix,
            batch_size=settings.embedding_batch_size,
            cache_dir=str(settings.model_cache_dir),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        vectors = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return [_normalize(v) for v in vectors]

    def embed_query(self, text: str) -> Vector:
        [vector] = self._model.query_embed(self._query_prefix + text)
        return _normalize(vector)


def _normalize(vector: np.typing.ArrayLike) -> Vector:
    """Scale to length 1, so cosine similarity is a plain dot product."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    result: Vector = (array / norm).tolist() if norm else array.tolist()  # tolist() is untyped
    return result
