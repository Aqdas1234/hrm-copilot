"""Running checked SQL: read-only, row-limited, values ready for JSON."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from hrm.analytics.runner import AnalyticsRunner, make_analytics_engine, with_row_limit
from hrm.analytics.sql_guard import check_query
from hrm.db.migrate import upgrade
from hrm.db.seed import seed
from hrm.db.session import make_engine


@pytest.fixture
def analytics(tmp_path: Path) -> Iterator[Engine]:
    url = f"sqlite:///{tmp_path / 'hr.db'}"
    owner = make_engine(url)
    upgrade(owner)
    with Session(owner) as session:
        seed(session, today=date.today(), demo_password="x")
        session.commit()
    owner.dispose()
    engine = make_analytics_engine(url)
    yield engine
    engine.dispose()


def test_a_checked_query_runs(analytics: Engine) -> None:
    result = AnalyticsRunner(analytics).run(
        check_query("SELECT department, COUNT(*) AS people FROM v_employees GROUP BY department")
    )
    assert result.columns == ["department", "people"]
    assert sum(row[1] for row in result.rows) == 30
    assert not result.truncated


def test_the_connection_cannot_write(analytics: Engine) -> None:
    # Bypasses the checker and the runner on purpose: the connection itself must refuse.
    with pytest.raises(OperationalError, match="readonly"), analytics.connect() as connection:
        connection.execute(text("DELETE FROM employees"))


def test_rows_are_limited_and_marked_truncated(analytics: Engine) -> None:
    result = AnalyticsRunner(analytics, max_rows=10).run(check_query("SELECT * FROM v_employees"))
    assert len(result.rows) == 10
    assert result.truncated


def test_a_smaller_limit_in_the_query_is_kept() -> None:
    assert with_row_limit("SELECT a FROM t ORDER BY a LIMIT 3", 201).endswith("LIMIT 3")
    assert with_row_limit("SELECT a FROM t ORDER BY a LIMIT 900", 201).endswith("LIMIT 201")
    assert with_row_limit("SELECT a FROM t UNION SELECT b FROM u", 11).endswith("LIMIT 11")


def test_values_are_plain_json(analytics: Engine) -> None:
    result = AnalyticsRunner(analytics).run(
        check_query("SELECT start_date, working_days FROM v_leave_requests LIMIT 1")
    )
    start, days = result.rows[0]
    assert isinstance(start, str) and isinstance(days, int | float)
