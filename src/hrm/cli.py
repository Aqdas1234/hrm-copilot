"""Command-line interface. Each RAG step gets its own subcommand as we build it."""

import time

import typer

from hrm import __version__
from hrm.core.config import get_settings
from hrm.core.errors import HRMError
from hrm.core.llm import get_answer_model, get_fast_model
from hrm.core.logging import configure_logging, get_logger

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
