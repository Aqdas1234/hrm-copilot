import json

import pytest

from hrm.core.logging import configure_logging, get_logger


def test_json_format_writes_one_json_object_per_line_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json")

    get_logger("tests.logging").info("retrieval.done", hits=5)

    captured = capsys.readouterr()
    assert captured.out == ""
    record = json.loads(captured.err.strip())
    assert record["event"] == "retrieval.done"
    assert record["hits"] == 5
    assert record["level"] == "info"
    assert record["module"] == "tests.logging"


def test_logger_created_before_configuration_uses_later_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "console")
    log = get_logger("tests.logging")  # like a module-level logger, made at import time

    configure_logging("INFO", "json")
    log.info("late.config")

    assert json.loads(capsys.readouterr().err.strip())["event"] == "late.config"


def test_messages_below_the_level_are_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", "json")

    get_logger("tests.logging").info("noise")

    assert capsys.readouterr().err == ""
