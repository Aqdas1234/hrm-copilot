from collections.abc import Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from hrm.api.app import create_app
from hrm.api.hr_routes import AuthConfig
from hrm.db.seed import seed
from hrm.db.session import create_schema, make_engine
from tests.api.helpers import PASSWORD, SECRET
from tests.api.test_app import FakeService


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    """A seeded in-memory HR database shared by the API's worker threads."""
    engine = make_engine("sqlite://")
    create_schema(engine)
    factory = sessionmaker(engine)
    with factory() as s:
        seed(s, today=date.today(), demo_password=PASSWORD)
        s.commit()
    yield factory
    engine.dispose()


@pytest.fixture
def client(factory: sessionmaker[Session]) -> Iterator[TestClient]:
    app = create_app(
        FakeService(), session_factory=factory, auth=AuthConfig(SECRET, ttl_minutes=60)
    )
    with TestClient(app) as c:
        yield c
