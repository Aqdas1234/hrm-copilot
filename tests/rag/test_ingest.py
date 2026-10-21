"""Ingestion with fakes: FakeEmbedder, InMemoryVectorStore and a word-count tokenizer."""

import hashlib
from pathlib import Path

import pytest

from hrm.rag.embeddings import Vector
from hrm.rag.ingest import Ingestor, Manifest
from hrm.rag.models import SourceDocument
from tests.fakes import FakeEmbedder, InMemoryVectorStore


def make_doc(doc_id: str, body: str) -> SourceDocument:
    text = f"# {doc_id} title\n\n## Section\n\n{body}\n"
    return SourceDocument(
        doc_id=doc_id,
        title=f"{doc_id} title",
        category="conduct",
        audience="all",
        source_path=f"{doc_id}.md",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        last_updated=None,
        text=text,
    )


def make_ingestor(
    tmp_path: Path, embedder: FakeEmbedder, store: InMemoryVectorStore, chunk_size: int = 400
) -> Ingestor:
    return Ingestor(
        embedder=embedder,
        store=store,
        manifest_path=tmp_path / "manifest.json",
        count_tokens=lambda text: len(text.split()),
        chunk_size=chunk_size,
        chunk_overlap=10,
        chunk_min_size=5,
    )


def embedded_texts(embedder: FakeEmbedder) -> int:
    return sum(len(call) for call in embedder.document_calls)


DOCS = [make_doc("doc-a", "alpha " * 30), make_doc("doc-b", "beta " * 30)]


def test_first_run_indexes_everything_and_writes_the_manifest(tmp_path: Path) -> None:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()

    report = make_ingestor(tmp_path, embedder, store).run(DOCS)

    assert report.added == ["doc-a", "doc-b"]
    assert report.chunks_embedded == store.count() == report.total_chunks > 0
    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert set(manifest.documents) == {"doc-a", "doc-b"}
    assert manifest.settings.embedding_model == embedder.name


def test_second_run_embeds_nothing(tmp_path: Path) -> None:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    make_ingestor(tmp_path, embedder, store).run(DOCS)
    calls_after_first_run = embedded_texts(embedder)

    report = make_ingestor(tmp_path, embedder, store).run(DOCS)

    assert report.chunks_embedded == 0
    assert report.unchanged == ["doc-a", "doc-b"]
    assert embedded_texts(embedder) == calls_after_first_run
    assert not report.rebuilt


def test_editing_one_document_reembeds_only_that_document(tmp_path: Path) -> None:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    ingestor = make_ingestor(tmp_path, embedder, store, chunk_size=20)  # several chunks per doc
    ingestor.run(DOCS)
    chunks_before = store.count()
    embedder.document_calls.clear()

    shorter_a = make_doc("doc-a", "alpha " * 5)  # fewer chunks than before
    report = ingestor.run([shorter_a, DOCS[1]])

    assert report.updated == ["doc-a"]
    assert report.unchanged == ["doc-b"]
    assert all("alpha" in t for call in embedder.document_calls for t in call)
    assert store.count() < chunks_before  # old doc-a chunks are gone, not left behind
    assert store.count() == report.total_chunks


def test_document_removed_from_disk_is_removed_from_the_index(tmp_path: Path) -> None:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    ingestor = make_ingestor(tmp_path, embedder, store)
    ingestor.run(DOCS)

    report = ingestor.run([DOCS[0]])

    assert report.removed == ["doc-b"]
    assert {hit.chunk.doc_id for hit in store.query([1.0] + [0.0] * 63, k=100)} == {"doc-a"}


def test_rebuild_flag_reembeds_everything(tmp_path: Path) -> None:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    ingestor = make_ingestor(tmp_path, embedder, store)
    ingestor.run(DOCS)

    report = ingestor.run(DOCS, rebuild=True)

    assert report.rebuilt
    assert report.added == ["doc-a", "doc-b"]
    assert report.chunks_embedded == store.count()


def test_changing_the_embedding_model_forces_a_rebuild(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    make_ingestor(tmp_path, FakeEmbedder(dimension=64), store).run(DOCS)

    new_model = FakeEmbedder(dimension=32)  # different name: fake-32
    report = make_ingestor(tmp_path, new_model, store).run(DOCS)

    assert report.rebuilt
    assert report.chunks_embedded == store.count()
    # Every stored vector is from the new model, so a 32-dim query works on all of them.
    assert len(store.query(new_model.embed_query("alpha"), k=100)) == store.count()


def test_unreadable_manifest_triggers_a_rebuild_instead_of_crashing(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")

    report = make_ingestor(tmp_path, FakeEmbedder(), InMemoryVectorStore()).run(DOCS)

    assert report.rebuilt
    assert report.added == ["doc-a", "doc-b"]


class FailingEmbedder(FakeEmbedder):
    def embed_documents(self, texts: list[str]) -> list[Vector]:  # type: ignore[override]
        raise RuntimeError("model crashed")


def test_failed_embedding_keeps_the_old_chunks_searchable(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    make_ingestor(tmp_path, FakeEmbedder(), store).run(DOCS)
    before = store.count()

    with pytest.raises(RuntimeError):
        make_ingestor(tmp_path, FailingEmbedder(), store).run(
            [make_doc("doc-a", "changed " * 30), DOCS[1]]
        )

    assert store.count() == before  # nothing was deleted before embedding succeeded
