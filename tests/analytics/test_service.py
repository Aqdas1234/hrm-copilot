"""The analytics service and tool: write, check, retry once, run, audit. Writer is faked."""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.agents.tools import ToolDeps, build_tools
from hrm.analytics.runner import AnalyticsRunner, make_analytics_engine
from hrm.analytics.service import AnalyticsQueryError, AnalyticsService, SQLDraft
from hrm.analytics.sql_guard import UnsafeQueryError
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.migrate import upgrade
from hrm.db.models import AuditLog
from hrm.db.seed import seed
from hrm.db.session import make_engine
from hrm.integrations.payroll import PayrollClient
from tests.agents.helpers import FakePolicies
from tests.fakes import FakeHolidays, FakeRates, StubHTTP

TODAY = date(2026, 10, 22)
PAYROLL = PayrollClient(StubHTTP(), base_url="http://payroll.test")  # not called here
COUNT_BY_COUNTRY = "SELECT country, COUNT(*) AS people FROM v_employees GROUP BY country"


class ScriptedWriter:
    def __init__(self, *sqls: str) -> None:
        self.sqls = list(sqls)
        self.seen: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self.seen.append(list(messages))
        return SQLDraft(sql=self.sqls.pop(0), explanation="People per country.")


@pytest.fixture
def parts(tmp_path: Path) -> Iterator[tuple[Session, AnalyticsRunner]]:
    url = f"sqlite:///{tmp_path / 'hr.db'}"
    owner = make_engine(url)
    upgrade(owner)
    analytics = make_analytics_engine(url)
    with Session(owner) as session:
        seed(session, today=TODAY, demo_password="x")
        session.commit()
        yield session, AnalyticsRunner(analytics, max_rows=200)
    analytics.dispose()
    owner.dispose()


def user(session: Session, name: str) -> UserContext:
    found = repo.get_user_by_email(session, f"{name}@example.com")
    assert found is not None
    return UserContext.from_user(found)


def service(parts: tuple[Session, AnalyticsRunner], writer: ScriptedWriter) -> AnalyticsService:
    session, runner = parts
    return AnalyticsService(session, user(session, "carol.reyes"), writer, runner, TODAY)


def audits(session: Session) -> list[str]:
    rows = session.scalars(select(AuditLog).where(AuditLog.entity == "analytics"))
    return [a.action for a in rows]


def test_a_valid_query_runs_and_is_audited(parts: tuple[Session, AnalyticsRunner]) -> None:
    answer = service(parts, ScriptedWriter(COUNT_BY_COUNTRY)).answer("people per country?")

    assert answer.columns == ["country", "people"]
    assert sum(r[1] for r in answer.rows) == 30
    assert audits(parts[0]) == ["analytics.query"]


def test_a_refused_query_gets_one_retry_with_the_reason(
    parts: tuple[Session, AnalyticsRunner],
) -> None:
    writer = ScriptedWriter("SELECT country FROM employees", COUNT_BY_COUNTRY)
    answer = service(parts, writer).answer("people per country?")

    assert answer.rows
    assert "Use only these views" in writer.seen[1][-1].text
    assert audits(parts[0]) == ["analytics.refused", "analytics.query"]


def test_two_refusals_are_final_and_nothing_runs(parts: tuple[Session, AnalyticsRunner]) -> None:
    writer = ScriptedWriter("SELECT * FROM compensation", "SELECT * FROM users")
    with pytest.raises(UnsafeQueryError):
        service(parts, writer).answer("everyone's salary")
    assert audits(parts[0]) == ["analytics.refused", "analytics.refused"]


def test_database_errors_become_a_clean_message(tmp_path: Path) -> None:
    empty = make_analytics_engine(f"sqlite:///{tmp_path / 'empty.db'}")  # no views at all
    owner = make_engine(f"sqlite:///{tmp_path / 'hr.db'}")
    upgrade(owner)
    with Session(owner) as session:
        seed(session, today=TODAY, demo_password="x")
        svc = AnalyticsService(
            session,
            user(session, "carol.reyes"),
            ScriptedWriter(COUNT_BY_COUNTRY),
            AnalyticsRunner(empty),
            TODAY,
        )
        with pytest.raises(AnalyticsQueryError, match="could not run"):
            svc.answer("people per country?")
        assert audits(session) == ["analytics.failed"]


def test_the_prompt_describes_the_views_and_today(parts: tuple[Session, AnalyticsRunner]) -> None:
    writer = ScriptedWriter(COUNT_BY_COUNTRY)
    service(parts, writer).answer("people per country?")
    prompt = writer.seen[0][0].text
    assert "v_salary_bands" in prompt and "2026-10-22" in prompt


# --- the agent tool ------------------------------------------------------------------------


def tool_names(session: Session, name: str, analytics: AnalyticsService | None) -> set[str]:
    deps = ToolDeps(
        session,
        user(session, name),
        FakePolicies(),
        FakeHolidays(),
        FakeRates(),
        payroll=PAYROLL,
        today=TODAY,
        analytics=analytics,
    )
    return {t.name for t in build_tools(deps)}


def test_only_hr_admins_get_the_analytics_tool(parts: tuple[Session, AnalyticsRunner]) -> None:
    svc = service(parts, ScriptedWriter())
    assert "query_hr_analytics" in tool_names(parts[0], "carol.reyes", svc)
    assert "query_hr_analytics" not in tool_names(parts[0], "bob.chen", svc)
    assert "query_hr_analytics" not in tool_names(parts[0], "alice.park", svc)
    assert "query_hr_analytics" not in tool_names(parts[0], "carol.reyes", None)


def test_the_tool_returns_the_sql_and_caps_rows(parts: tuple[Session, AnalyticsRunner]) -> None:
    session, _ = parts
    svc = service(parts, ScriptedWriter("SELECT first_name FROM v_employees"))
    deps = ToolDeps(
        session,
        user(session, "carol.reyes"),
        FakePolicies(),
        FakeHolidays(),
        FakeRates(),
        payroll=PAYROLL,
        today=TODAY,
        analytics=svc,
    )
    analytics_tool = next(t for t in build_tools(deps) if t.name == "query_hr_analytics")

    out = json.loads(analytics_tool.invoke({"question": "everyone's first name"}))

    assert out["sql"].startswith("SELECT first_name FROM v_employees")
    assert len(out["rows"]) <= 50
    assert out["more_rows"] is False  # 30 employees
