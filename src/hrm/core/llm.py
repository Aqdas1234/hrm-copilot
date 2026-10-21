"""Chat model factory.

Two roles, two models:
- answer model (large): writes the final cited answer for the user.
- fast model (small): cheap internal jobs such as judging answers or rewriting queries.

Callers ask for a role, not a model name, so the models can be changed in settings
without touching code. Timeouts and retries are set here once for every call site.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from hrm.core.config import Settings, get_settings
from hrm.core.errors import ConfigurationError


def get_answer_model(settings: Settings | None = None) -> BaseChatModel:
    """The large model used to write user-facing answers."""
    settings = settings or get_settings()
    return _build(settings.answer_model, settings)


def get_fast_model(settings: Settings | None = None) -> BaseChatModel:
    """The small model used for internal tasks (judging, query rewriting)."""
    settings = settings or get_settings()
    return _build(settings.fast_model, settings)


def _build(model: str, settings: Settings) -> BaseChatModel:
    # An empty value ("HRM_GROQ_API_KEY=" copied from .env.example) counts as missing.
    api_key = settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else ""
    if not api_key:
        raise ConfigurationError("HRM_GROQ_API_KEY is not set. Add it to .env (see .env.example).")
    return init_chat_model(
        model,
        model_provider="groq",
        api_key=api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
