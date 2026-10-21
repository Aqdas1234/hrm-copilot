"""Test doubles that stand in for real models: offline, instant and deterministic."""

import hashlib
import math
import re
from collections.abc import Sequence

from hrm.rag.embeddings import Vector

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words hashing: each word adds 1 to a fixed slot, then the vector is normalized.

    Texts that share words get similar vectors, so retrieval tests behave sensibly,
    without downloading a model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return f"fake-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.document_calls.append(list(texts))
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            slot = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            vector[slot % self._dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector
