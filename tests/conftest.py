from collections.abc import Callable, Iterator
from typing import Any

import pytest
import structlog
from sqlalchemy.orm import Session, sessionmaker

from hrm.core.config import Settings
from hrm.db.session import create_schema, make_engine

MakeSettings = Callable[..., Settings]


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Undo configure_logging() after each test; it may point at a capture stream that closes."""
    yield
    structlog.reset_defaults()


@pytest.fixture
def make_settings() -> MakeSettings:
    """Build Settings from defaults + overrides, ignoring the developer's real .env file."""

    def _make(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def session() -> Iterator[Session]:
    """A fresh in-memory SQLite HR database per test (foreign keys enforced)."""
    engine = make_engine("sqlite://")
    create_schema(engine)
    with sessionmaker(engine)() as s:
        yield s
    engine.dispose()
