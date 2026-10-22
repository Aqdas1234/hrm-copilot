"""Run checked analytics SQL on a read-only connection, with a time and row limit.

Layers here (the SQL was already checked by sql_guard):
- a separate engine for a separate database account; in production that account may
  only read the analytics views (docker/mysql-init creates it), so even a query that
  slipped past the checker cannot read raw tables;
- every connection is read-only and time-limited as soon as it opens (MySQL: read-only
  session, max_execution_time; SQLite: query_only);
- at most `max_rows` rows come back. The limit is set in the query itself, not by
  wrapping it: MySQL may ignore ORDER BY inside a subquery, which would break "top 5".
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlglot
from pydantic import BaseModel
from sqlalchemy import Engine, event, text
from sqlglot import exp

from hrm.db.session import make_engine

MAX_ROWS = 200
TIMEOUT_MS = 2_000


class AnalyticsResult(BaseModel):
    sql: str  # exactly what ran
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool  # more rows existed than were returned


def make_analytics_engine(url: str, *, timeout_ms: int = TIMEOUT_MS) -> Engine:
    engine = make_engine(url)

    @event.listens_for(engine, "connect")
    def _read_only(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        if engine.dialect.name == "sqlite":
            cursor.execute("PRAGMA query_only = ON")
        else:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            cursor.execute(f"SET SESSION max_execution_time = {int(timeout_ms)}")
        cursor.close()

    return engine


class AnalyticsRunner:
    def __init__(self, engine: Engine, *, max_rows: int = MAX_ROWS) -> None:
        self._engine = engine
        self._max_rows = max_rows

    def run(self, checked_sql: str) -> AnalyticsResult:
        """`checked_sql` must come from sql_guard.check_query."""
        sql = with_row_limit(checked_sql, self._max_rows + 1)  # one extra: was it cut?
        with self._engine.connect() as connection:
            result = connection.execute(text(sql))
            columns = list(result.keys())
            rows = [[_plain(v) for v in row] for row in result.fetchall()]
        return AnalyticsResult(
            sql=sql,
            columns=columns,
            rows=rows[: self._max_rows],
            truncated=len(rows) > self._max_rows,
        )


def with_row_limit(sql: str, limit: int) -> str:
    """Keep the query's own LIMIT if it is smaller ("top 3" stays 3)."""
    tree = sqlglot.parse_one(sql, dialect="mysql")
    assert isinstance(tree, exp.Query)  # check_query accepts only SELECT / UNION
    existing = tree.args.get("limit")
    if existing is not None and existing.expression.is_int:
        limit = min(limit, int(existing.expression.name))
    return tree.limit(limit).sql(dialect="mysql")


def _plain(value: Any) -> Any:
    """JSON-friendly values for the model and the API."""
    if isinstance(value, Decimal):
        return float(value) if value % 1 else int(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
