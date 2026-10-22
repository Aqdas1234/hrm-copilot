import pytest
from sqlalchemy.orm import Session

from hrm.db.seed import seed
from tests.agents.helpers import TODAY


@pytest.fixture
def db(session: Session) -> Session:
    seed(session, today=TODAY, demo_password="x")
    return session
