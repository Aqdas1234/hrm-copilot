import pytest
from pydantic import ValidationError

from tests.conftest import MakeSettings


def test_defaults_are_valid(make_settings: MakeSettings) -> None:
    settings = make_settings()

    assert settings.answer_model == "openai/gpt-oss-120b"
    assert settings.fast_model == "openai/gpt-oss-20b"
    assert settings.top_k <= settings.retrieve_k


def test_environment_variables_override_defaults(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HRM_TOP_K", "3")
    monkeypatch.setenv("HRM_LOG_FORMAT", "json")

    settings = make_settings()

    assert settings.top_k == 3
    assert settings.log_format == "json"


def test_api_key_is_hidden_in_repr(make_settings: MakeSettings) -> None:
    settings = make_settings(groq_api_key="gsk-secret")

    assert "gsk-secret" not in repr(settings)
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "gsk-secret"


def test_overlap_must_be_smaller_than_chunk_size(make_settings: MakeSettings) -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_top_k_cannot_exceed_retrieve_k(make_settings: MakeSettings) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        make_settings(top_k=10, retrieve_k=5)


def test_invalid_log_level_is_rejected(make_settings: MakeSettings) -> None:
    with pytest.raises(ValidationError):
        make_settings(log_level="VERBOSE")
