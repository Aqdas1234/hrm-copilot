import pytest
from typer.testing import CliRunner

from hrm import __version__
from hrm.cli import app

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
