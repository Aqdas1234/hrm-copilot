"""Building a model object makes no network call, so these tests stay offline."""

import pytest
from langchain_groq import ChatGroq

from hrm.core.errors import ConfigurationError
from hrm.core.llm import get_answer_model, get_fast_model
from tests.conftest import MakeSettings


def test_missing_api_key_raises_configuration_error(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HRM_GROQ_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="HRM_GROQ_API_KEY"):
        get_answer_model(make_settings())


def test_answer_and_fast_models_use_configured_names_and_limits(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(groq_api_key="gsk-test", llm_timeout_s=12.5, llm_max_retries=4)

    answer = get_answer_model(settings)
    fast = get_fast_model(settings)

    assert isinstance(answer, ChatGroq)
    assert isinstance(fast, ChatGroq)
    assert answer.model_name == "openai/gpt-oss-120b"
    assert fast.model_name == "openai/gpt-oss-20b"
    assert answer.request_timeout == 12.5
    assert answer.max_retries == 4
