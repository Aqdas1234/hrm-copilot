"""Command-line interface. Each RAG step gets its own subcommand as we build it."""

import typer

from hrm import __version__

app = typer.Typer(
    name="hrm",
    help="HRM Copilot: ingest HR policies, search them and ask questions.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """HRM Copilot command-line tools."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
