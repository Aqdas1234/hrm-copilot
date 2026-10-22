import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hrm import __version__
from hrm.cli import app
from hrm.core.config import get_settings
from hrm.core.errors import LLMError
from hrm.db.migrate import head_revision
from hrm.rag.models import Citation, PolicyAnswer
from hrm.rag.service import AnswerResult
from tests.fakes import FakeEmbedder, FakeReranker

runner = CliRunner()


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_chunk_stats_prints_a_row_per_document(monkeypatch: pytest.MonkeyPatch) -> None:
    # Count words instead of tiktoken tokens so the test needs no download.
    monkeypatch.setattr("hrm.cli.tiktoken_counter", lambda: lambda text: len(text.split()))

    result = runner.invoke(app, ["chunk-stats"])

    assert result.exit_code == 0, result.output
    for doc_id in ["benefits-overview", "code-of-conduct", "parental-leave", "time-off-policy"]:
        assert doc_id in result.stdout
    assert "Size histogram" in result.stdout


def test_ingest_twice_embeds_nothing_the_second_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Real documents and Chroma (on a temp folder); fake embedder and word counter, so offline.
    monkeypatch.setenv("HRM_INDEX_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr("hrm.cli.FastEmbedEmbedder.from_settings", lambda _s: FakeEmbedder())
    monkeypatch.setattr("hrm.cli.tiktoken_counter", lambda: lambda text: len(text.split()))

    first = runner.invoke(app, ["ingest"])
    second = runner.invoke(app, ["ingest"])
    get_settings.cache_clear()

    assert first.exit_code == 0, first.output
    assert "(full rebuild)" in first.stdout
    assert "added       5" in first.stdout
    assert second.exit_code == 0, second.output
    assert "unchanged   5" in second.stdout
    assert "chunks embedded: 0" in second.stdout
    assert (tmp_path / "manifest.json").exists()


@pytest.fixture
def offline_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """A real Chroma index in a temp folder, built with the fake embedder and fake reranker."""
    monkeypatch.setenv("HRM_INDEX_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr("hrm.cli.FastEmbedEmbedder.from_settings", lambda _s: FakeEmbedder())
    monkeypatch.setattr(
        "hrm.rag.retrieval.factory.FastEmbedReranker.from_settings", lambda _s: FakeReranker()
    )
    monkeypatch.setattr("hrm.cli.tiktoken_counter", lambda: lambda text: len(text.split()))
    yield
    get_settings.cache_clear()


def test_search_prints_ranked_results(offline_index: None) -> None:
    assert runner.invoke(app, ["ingest"]).exit_code == 0

    result = runner.invoke(app, ["search", "parental leave weeks", "-k", "3"])

    assert result.exit_code == 0, result.output
    assert " 1. " in result.stdout and " 3. " in result.stdout and " 4. " not in result.stdout


def test_search_with_category_only_returns_that_category(offline_index: None) -> None:
    runner.invoke(app, ["ingest"])

    result = runner.invoke(app, ["search", "leave", "-k", "5", "-c", "benefits"])

    assert result.exit_code == 0, result.output
    result_lines = [ln for ln in result.stdout.splitlines() if re.match(r"\s*\d+\. ", ln)]
    assert result_lines
    assert all("  benefits-overview  " in ln for ln in result_lines)


def test_search_on_empty_index_explains_what_to_do(offline_index: None) -> None:
    result = runner.invoke(app, ["search", "anything"])

    assert result.exit_code == 1
    assert "hrm ingest" in result.output


def test_search_rejects_unknown_category(offline_index: None) -> None:
    result = runner.invoke(app, ["search", "anything", "-c", "payroll"])

    assert result.exit_code == 1
    assert "unknown category" in result.output


def test_ask_prints_the_answer_and_its_sources(
    offline_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["ingest"])
    citation = Citation(
        number=1,
        chunk_id="time-off-policy:pto-ground-rules:0",
        doc_id="time-off-policy",
        section_path=("Paid time off (PTO) policy", "PTO ground rules"),
    )
    answer = PolicyAnswer(
        answer="Give 5 business days notice [1].", found=True, citations=[citation]
    )

    class FakeService:
        def answer(self, question: str) -> AnswerResult:
            return AnswerResult(answer=answer, context=[])

    monkeypatch.setattr("hrm.cli.RAGService.from_settings", lambda _s: FakeService())

    result = runner.invoke(app, ["ask", "How much notice?"])

    assert result.exit_code == 0, result.output
    assert "Give 5 business days notice [1]." in result.stdout
    assert "[1] time-off-policy > PTO ground rules" in result.stdout


def test_ask_reports_llm_errors_cleanly(
    offline_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["ingest"])

    class FailingService:
        def answer(self, question: str) -> AnswerResult:
            raise LLMError("Groq is down")

    monkeypatch.setattr("hrm.cli.RAGService.from_settings", lambda _s: FailingService())

    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "Groq is down" in result.output


def test_ask_on_empty_index_explains_what_to_do(offline_index: None) -> None:
    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "hrm ingest" in result.output


def test_serve_starts_uvicorn_with_the_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("hrm.cli.uvicorn.run", lambda app, **kw: calls.append((app, kw)))

    result = runner.invoke(app, ["serve", "--port", "9001"])

    assert result.exit_code == 0, result.output
    [(target, options)] = calls
    assert target == "hrm.api.app:create_app"
    assert options["factory"] is True
    assert options["port"] == 9001


def test_ui_runs_streamlit_on_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    class Done:
        returncode = 0

    def fake_run(cmd: list[str], check: bool) -> Done:
        commands.append(cmd)
        return Done()

    monkeypatch.setattr("hrm.cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["ui", "--port", "8600"])

    assert result.exit_code == 0, result.output
    [command] = commands
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[4].endswith("ui/streamlit_app.py")
    assert command[-2:] == ["--server.port", "8600"]


def test_db_upgrade_creates_the_tables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HRM_DATABASE_URL", f"sqlite:///{tmp_path / 'hr.db'}")
    get_settings.cache_clear()

    result = runner.invoke(app, ["db", "upgrade"])
    again = runner.invoke(app, ["db", "upgrade"])  # nothing left to do
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert f"Database ready at schema version {head_revision()}" in result.stdout
    assert again.exit_code == 0, again.output


def test_db_upgrade_reports_an_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HRM_DATABASE_URL", "mysql+pymysql://u:secret@127.0.0.1:1/none")
    get_settings.cache_clear()

    result = runner.invoke(app, ["db", "upgrade"])
    get_settings.cache_clear()

    assert result.exit_code == 1
    assert "could not set up the database" in result.output
    assert "secret" not in result.output  # the password never appears in error output


def test_db_seed_then_refuse_then_reset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HRM_DATABASE_URL", f"sqlite:///{tmp_path / 'hr.db'}")
    get_settings.cache_clear()

    first = runner.invoke(app, ["db", "seed"])
    again = runner.invoke(app, ["db", "seed"])
    reset = runner.invoke(app, ["db", "seed", "--reset"])
    get_settings.cache_clear()

    assert first.exit_code == 0, first.output
    assert "Seeded 30 employees" in first.stdout
    assert "alice.park@example.com" in first.stdout
    assert again.exit_code == 1 and "--reset" in again.output
    assert runner.invoke(app, ["db", "seed", "--if-empty"]).exit_code == 0
    assert reset.exit_code == 0, reset.output
