"""Command-line interface. Each RAG step gets its own subcommand as we build it."""

import time

import typer

from hrm import __version__
from hrm.core.config import get_settings
from hrm.core.errors import HRMError
from hrm.core.llm import get_answer_model, get_fast_model
from hrm.core.logging import configure_logging, get_logger
from hrm.rag.chunking import chunk_document, tiktoken_counter
from hrm.rag.cleaning import clean_document
from hrm.rag.embeddings import FastEmbedEmbedder
from hrm.rag.ingest import Ingestor
from hrm.rag.loading import load_documents
from hrm.rag.vectorstore import ChromaStore

log = get_logger(__name__)

app = typer.Typer(
    name="hrm",
    help="HRM Copilot: ingest HR policies, search them and ask questions.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """HRM Copilot command-line tools."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command("check-llm")
def check_llm(
    fast: bool = typer.Option(False, "--fast", help="Check the fast model instead."),
) -> None:
    """Send one tiny prompt to the LLM to confirm the key and model work (uses the network)."""
    try:
        model = get_fast_model() if fast else get_answer_model()
    except HRMError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    start = time.perf_counter()
    try:
        reply = model.invoke("Reply with exactly one word: OK")
    except Exception as exc:
        typer.echo(f"LLM call failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    log.info("llm.check", model=getattr(model, "model_name", "?"), ms=elapsed_ms)
    typer.echo(reply.text.strip())


@app.command("chunk-stats")
def chunk_stats() -> None:
    """Load, clean and chunk every document, then print chunk counts and token sizes."""
    settings = get_settings()
    try:
        documents = load_documents(settings.docs_dir)
    except HRMError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    count_tokens = tiktoken_counter()
    sizes: list[int] = []
    typer.echo(f"{'document':<26} {'chunks':>6} {'min':>5} {'median':>6} {'max':>5}")
    for doc in documents:
        chunks = chunk_document(
            clean_document(doc),
            max_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
            min_tokens=settings.chunk_min_size,
            count_tokens=count_tokens,
        )
        tokens = sorted(c.token_count for c in chunks)
        sizes += tokens
        typer.echo(
            f"{doc.doc_id:<26} {len(chunks):>6} {tokens[0]:>5} "
            f"{tokens[len(tokens) // 2]:>6} {tokens[-1]:>5}"
        )

    step = 100
    typer.echo(f"\n{len(sizes)} chunks, limit {settings.chunk_size} tokens. Size histogram:")
    for low in range(0, settings.chunk_size + 1, step):  # +1: a chunk may be exactly the limit
        n = sum(low <= size < low + step for size in sizes)
        typer.echo(f"  {low:>3}-{low + step - 1:<3} {'#' * n} {n}")


@app.command()
def ingest(
    rebuild: bool = typer.Option(False, "--rebuild", help="Re-embed every document."),
) -> None:
    """Index the policy documents. Only new or changed documents are embedded."""
    settings = get_settings()
    try:
        documents = load_documents(settings.docs_dir)
    except HRMError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    ingestor = Ingestor(
        embedder=FastEmbedEmbedder.from_settings(settings),
        store=ChromaStore.from_settings(settings),
        manifest_path=settings.index_dir / "manifest.json",
        count_tokens=tiktoken_counter(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_min_size=settings.chunk_min_size,
    )
    report = ingestor.run(documents, rebuild=rebuild)

    typer.echo(
        f"Ingest finished in {report.seconds:.1f}s" + (" (full rebuild)" if report.rebuilt else "")
    )
    for label, doc_ids in [
        ("added", report.added),
        ("updated", report.updated),
        ("unchanged", report.unchanged),
        ("removed", report.removed),
    ]:
        typer.echo(f"  {label:<10} {len(doc_ids):>2}  {', '.join(doc_ids)}")
    typer.echo(f"  chunks embedded: {report.chunks_embedded}, in index: {report.total_chunks}")
