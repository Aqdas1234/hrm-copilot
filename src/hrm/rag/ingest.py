"""Idempotent ingestion: load -> clean -> chunk -> embed -> store, only for what changed.

A manifest (JSON next to the index) remembers, per document, the content hash and the chunk
IDs that were stored, plus the settings the index was built with. Each run compares the
current documents with it:
  - same hash            -> skipped (nothing embedded)
  - new or changed hash  -> re-chunked, re-embedded, old chunks replaced
  - no longer on disk    -> its chunks deleted
If the index settings changed (embedding model, chunk sizes, PIPELINE_VERSION), every document
is rebuilt, because vectors or chunks from different settings must not be mixed.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from hrm.core.logging import get_logger
from hrm.rag.chunking import TokenCounter, chunk_document
from hrm.rag.cleaning import clean_document
from hrm.rag.embeddings import Embedder
from hrm.rag.models import SourceDocument
from hrm.rag.vectorstore import VectorStore

log = get_logger(__name__)

# Bump when cleaning or chunking code changes in a way that changes chunks: forces a rebuild.
PIPELINE_VERSION = 1


class IndexSettings(BaseModel):
    """Everything that, if changed, makes the existing index invalid."""

    pipeline_version: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    chunk_min_size: int


class DocumentEntry(BaseModel):
    content_hash: str
    chunk_ids: list[str]


class Manifest(BaseModel):
    settings: IndexSettings
    documents: dict[str, DocumentEntry] = {}


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_embedded: int = 0
    total_chunks: int = 0
    rebuilt: bool = False
    seconds: float = 0.0


@dataclass(frozen=True)
class Ingestor:
    embedder: Embedder
    store: VectorStore
    manifest_path: Path
    count_tokens: TokenCounter
    chunk_size: int
    chunk_overlap: int
    chunk_min_size: int

    def run(self, documents: list[SourceDocument], *, rebuild: bool = False) -> IngestReport:
        start = time.perf_counter()
        report = IngestReport()
        settings = self._index_settings()
        previous = self._load_manifest()

        if rebuild or previous is None or previous.settings != settings:
            report.rebuilt = True
            log.info("ingest.rebuild", reason=_rebuild_reason(rebuild, previous, settings))
            known = previous.documents if previous else {}
            for doc_id in set(known) | {d.doc_id for d in documents}:
                self.store.delete_by_doc(doc_id)
            previous = None

        old_docs = previous.documents if previous else {}
        manifest = Manifest(settings=settings)
        for doc in documents:
            old = old_docs.get(doc.doc_id)
            if old is not None and old.content_hash == doc.content_hash:
                manifest.documents[doc.doc_id] = old
                report.unchanged.append(doc.doc_id)
                continue
            chunk_ids = self._index_document(doc)
            manifest.documents[doc.doc_id] = DocumentEntry(
                content_hash=doc.content_hash, chunk_ids=chunk_ids
            )
            report.chunks_embedded += len(chunk_ids)
            (report.added if old is None else report.updated).append(doc.doc_id)

        current = {d.doc_id for d in documents}
        for doc_id in sorted(set(old_docs) - current):
            self.store.delete_by_doc(doc_id)
            report.removed.append(doc_id)

        self._save_manifest(manifest)
        report.total_chunks = sum(len(e.chunk_ids) for e in manifest.documents.values())
        report.seconds = time.perf_counter() - start
        log.info(
            "ingest.done",
            added=len(report.added),
            updated=len(report.updated),
            unchanged=len(report.unchanged),
            removed=len(report.removed),
            chunks_embedded=report.chunks_embedded,
            seconds=round(report.seconds, 2),
        )
        return report

    def _index_document(self, doc: SourceDocument) -> list[str]:
        chunks = chunk_document(
            clean_document(doc),
            max_tokens=self.chunk_size,
            overlap_tokens=self.chunk_overlap,
            min_tokens=self.chunk_min_size,
            count_tokens=self.count_tokens,
        )
        # Embed first: if it fails, the document's old chunks are still in the store.
        vectors = self.embedder.embed_documents([c.embedding_text for c in chunks])
        self.store.delete_by_doc(doc.doc_id)  # drops chunks whose IDs no longer exist
        self.store.upsert(chunks, vectors)
        log.info("ingest.document", doc_id=doc.doc_id, chunks=len(chunks))
        return [c.chunk_id for c in chunks]

    def _index_settings(self) -> IndexSettings:
        return IndexSettings(
            pipeline_version=PIPELINE_VERSION,
            embedding_model=self.embedder.name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_min_size=self.chunk_min_size,
        )

    def _load_manifest(self) -> Manifest | None:
        if not self.manifest_path.exists():
            return None
        try:
            return Manifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        except ValidationError:
            log.warning("ingest.manifest_unreadable", path=str(self.manifest_path))
            return None

    def _save_manifest(self, manifest: Manifest) -> None:
        # Write to a temp file, then rename: a crash never leaves a half-written manifest.
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)


def _rebuild_reason(rebuild: bool, previous: Manifest | None, settings: IndexSettings) -> str:
    if rebuild:
        return "requested"
    if previous is None:
        return "no usable manifest"
    changed = [
        name
        for name, value in settings.model_dump().items()
        if getattr(previous.settings, name) != value
    ]
    return "settings changed: " + ", ".join(changed)
