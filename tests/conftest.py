from collections.abc import Callable
from typing import Any

import pytest

from hrm.core.config import Settings

MakeSettings = Callable[..., Settings]


@pytest.fixture
def make_settings() -> MakeSettings:
    """Build Settings from defaults + overrides, ignoring the developer's real .env file."""

    def _make(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make
