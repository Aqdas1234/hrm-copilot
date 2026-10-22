from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker

from hrm.db.session import create_schema, make_engine


@pytest.fixture
def session() -> Iterator[Session]:
    """A fresh in-memory SQLite database per test (foreign keys enforced)."""
    engine = make_engine("sqlite://")
    create_schema(engine)
    with sessionmaker(engine)() as s:
        yield s
    engine.dispose()
