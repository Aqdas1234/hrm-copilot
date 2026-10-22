"""The analytics views: what they expose, and what they must never expose."""

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session

from hrm.analytics.schema import VIEWS
from hrm.db.migrate import upgrade
from hrm.db.seed import seed
from hrm.db.session import make_engine

FORBIDDEN_COLUMNS = {"email", "password_hash", "reason", "description", "base_salary_cents"}


@pytest.fixture
def engine() -> Iterator[Engine]:
    engine = make_engine("sqlite://")
    upgrade(engine)
    with Session(engine) as session:
        seed(session, today=date.today(), demo_password="x")
        session.commit()
    yield engine
    engine.dispose()


def test_the_description_matches_the_views(engine: Engine) -> None:
    assert set(inspect(engine).get_view_names()) == {v.name for v in VIEWS}
    for view in VIEWS:
        columns = [c["name"] for c in inspect(engine).get_columns(view.name)]
        assert columns == list(view.columns), view.name


def test_no_view_exposes_free_text_contact_data_or_single_salaries(engine: Engine) -> None:
    for view in VIEWS:
        assert not set(view.columns) & FORBIDDEN_COLUMNS, view.name


def test_views_return_data(engine: Engine) -> None:
    with engine.connect() as c:
        assert c.scalar(text("SELECT COUNT(*) FROM v_employees")) == 30
        assert c.scalar(text("SELECT COUNT(*) FROM v_leave_requests")) == 4
        assert c.scalar(text("SELECT COUNT(*) FROM v_expense_claims")) == 3


def test_salary_bands_never_describe_fewer_than_three_people(engine: Engine) -> None:
    with engine.connect() as c:
        groups = c.execute(text("SELECT employees FROM v_salary_bands")).scalars().all()
        everyone = c.scalar(text("SELECT COUNT(*) FROM compensation"))
    assert groups, "expected at least one group of 3+"
    assert min(groups) >= 3
    assert sum(groups) < (everyone or 0)  # small groups were left out
