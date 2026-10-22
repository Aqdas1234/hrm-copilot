"""Command-line interface. Each RAG step gets its own subcommand as we build it."""

import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import get_args

import typer
import uvicorn
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from hrm import __version__
from hrm.core.config import get_settings
from hrm.core.errors import HRMError
from hrm.core.llm import get_answer_model, get_fast_model
from hrm.core.logging import configure_logging, get_logger
from hrm.db.models import Base, Compensation, Employee
from hrm.db.seed import DatabaseNotEmptyError, seed
from hrm.db.session import create_schema, make_engine, session_scope
from hrm.integrations.mock_payroll import create_mock_payroll_app
from hrm.rag.chunking import chunk_document, tiktoken_counter
from hrm.rag.cleaning import clean_document
from hrm.rag.embeddings import FastEmbedEmbedder
from hrm.rag.ingest import Ingestor
from hrm.rag.loading import load_documents
from hrm.rag.models import Category
from hrm.rag.retrieval.factory import build_retriever
from hrm.rag.service import RAGService
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


@app.command()
def search(
    query: str = typer.Argument(..., help="The question to search for."),
    k: int = typer.Option(5, "-k", min=1, help="Number of results."),
    category: str | None = typer.Option(
        None, "--category", "-c", help=f"Only this category: {', '.join(get_args(Category))}."
    ),
) -> None:
    """Show the chunks most relevant to a question (HRM_RETRIEVAL_MODE, no LLM)."""
    if category is not None and category not in get_args(Category):
        typer.echo(f"Error: unknown category {category!r}.", err=True)
        raise typer.Exit(code=1)
    settings = get_settings()
    store = ChromaStore.from_settings(settings)
    if store.count() == 0:
        typer.echo("The index is empty. Run `hrm ingest` first.", err=True)
        raise typer.Exit(code=1)

    retriever = build_retriever(settings, FastEmbedEmbedder.from_settings(settings), store)
    results = retriever.retrieve(query, k, category=category)  # type: ignore[arg-type]

    for r in results:
        section = " > ".join(r.chunk.section_path[1:]) or "(introduction)"
        snippet = " ".join(r.chunk.text.split())[:160]
        typer.echo(f"{r.rank:>2}. {r.score:.3f}  {r.chunk.doc_id}  |  {section}")
        typer.echo(f"     {snippet}...\n")


@app.command()
def ask(question: str = typer.Argument(..., help="Your HR policy question.")) -> None:
    """Answer a policy question with citations (retrieval + Groq answer model)."""
    settings = get_settings()
    if ChromaStore.from_settings(settings).count() == 0:
        typer.echo("The index is empty. Run `hrm ingest` first.", err=True)
        raise typer.Exit(code=1)
    try:
        result = RAGService.from_settings(settings).answer(question)
    except HRMError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.answer.answer)
    if result.answer.citations:
        typer.echo("\nSources:")
        for c in result.answer.citations:
            section = " > ".join(c.section_path[1:]) or "(introduction)"
            typer.echo(f"  [{c.number}] {c.doc_id} > {section}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8000, help="Port to listen on."),
) -> None:
    """Run the HTTP API (docs at http://HOST:PORT/docs)."""
    uvicorn.run("hrm.api.app:create_app", factory=True, host=host, port=port, log_config=None)


@app.command()
def ui(port: int = typer.Option(8501, help="Port for the web page.")) -> None:
    """Open the Streamlit question page (needs `hrm serve` running for the API)."""
    page = Path(__file__).parent / "ui" / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(page), "--server.port", str(port)]
    raise typer.Exit(subprocess.run(command, check=False).returncode)


db_app = typer.Typer(help="HR database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init() -> None:
    """Create the HR database tables (safe to re-run: existing tables are kept)."""
    engine = make_engine(get_settings().database_url.get_secret_value())
    try:
        create_schema(engine)
    except Exception as exc:  # unreachable server, wrong credentials, missing database
        typer.echo(f"Error: could not set up the database: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Database ready: {engine.url.render_as_string(hide_password=True)}")


@db_app.command("seed")
def db_seed(
    reset: bool = typer.Option(False, "--reset", help="Drop ALL tables and data first."),
) -> None:
    """Fill the HR database with deterministic demo data."""
    settings = get_settings()
    engine = make_engine(settings.database_url.get_secret_value())
    try:
        if reset:
            Base.metadata.drop_all(engine)
        create_schema(engine)
        with session_scope(sessionmaker(engine)) as session:
            summary = seed(
                session,
                today=date.today(),
                demo_password=settings.demo_password.get_secret_value(),
            )
    except DatabaseNotEmptyError as exc:
        typer.echo(f"Error: {exc} (use --reset)", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # unreachable server, wrong credentials, ...
        typer.echo(f"Error: could not seed the database: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Seeded {summary.employees} employees, {summary.leave_requests} leave requests, "
        f"{summary.expense_claims} expense claims, {summary.holidays} public holidays."
    )
    typer.echo("Demo logins (password: HRM_DEMO_PASSWORD):")
    for email in summary.demo_users:
        typer.echo(f"  {email}")


@app.command("mock-payroll")
def mock_payroll(port: int = typer.Option(8090, help="Port for the mock payroll API.")) -> None:
    """Run the mock payroll provider (fails at HRM_PAYROLL_FAILURE_RATE, default 20%)."""
    settings = get_settings()
    engine = make_engine(settings.database_url.get_secret_value())
    with sessionmaker(engine)() as session:
        salaries = {
            code: (cents, currency)
            for code, cents, currency in session.execute(
                select(
                    Employee.employee_code, Compensation.base_salary_cents, Compensation.currency
                )
                .join(Compensation, Compensation.employee_id == Employee.id)
                .order_by(Compensation.effective_from)  # latest row per employee wins
            )
        }
    typer.echo(
        f"Mock payroll: {len(salaries)} employees, failure rate {settings.payroll_failure_rate:.0%}"
    )
    payroll_app = create_mock_payroll_app(salaries, failure_rate=settings.payroll_failure_rate)
    uvicorn.run(payroll_app, host="127.0.0.1", port=port, log_config=None)
