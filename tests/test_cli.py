from pathlib import Path

import pytest
from typer.testing import CliRunner

from hrm import __version__
from hrm.cli import app
from hrm.core.config import get_settings
from tests.fakes import FakeEmbedder

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
