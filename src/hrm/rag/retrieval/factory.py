"""Build the retriever chosen in settings (HRM_RETRIEVAL_MODE). One place, so the CLI,
evaluations and later the API always assemble retrieval the same way."""

from hrm.core.config import Settings
from hrm.rag.embeddings import Embedder
from hrm.rag.retrieval import Retriever
from hrm.rag.retrieval.bm25 import BM25Retriever
from hrm.rag.retrieval.dense import DenseRetriever
from hrm.rag.retrieval.hybrid import HybridRetriever
from hrm.rag.vectorstore import VectorStore


def build_retriever(settings: Settings, embedder: Embedder, store: VectorStore) -> Retriever:
    mode = settings.retrieval_mode
    if mode == "dense":
        return DenseRetriever(embedder, store)
    bm25 = BM25Retriever(store.all_chunks())
    if mode == "bm25":
        return bm25
    return HybridRetriever(
        [DenseRetriever(embedder, store), bm25],
        candidates=settings.retrieve_k,
        rrf_k=settings.rrf_k,
    )
