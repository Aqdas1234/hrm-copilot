"""Shared helpers for agent tests: real tools for a seeded user, fakes for services."""

from datetime import date

from fastapi.testclient import TestClient
from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from hrm.agents.tools import ToolDeps, build_tools
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.integrations.mock_payroll import create_mock_payroll_app
from hrm.integrations.payroll import PayrollClient
from hrm.rag.models import Category, Chunk, RetrievedChunk
from tests.fakes import FakeHolidays, FakeRates

TODAY = date(2026, 10, 22)


class FakePolicies:
    def search(
        self, query: str, k: int | None = None, *, category: Category | None = None
    ) -> list[RetrievedChunk]:
        chunk = Chunk(
            chunk_id="time-off-policy:pto-ground-rules:0",
            doc_id="time-off-policy",
            category="time_off",
            audience="all",
            section_path=("PTO policy", "PTO ground rules"),
            text="Inform your Manager a minimum of 5 business days before your leave.",
            token_count=12,
        )
        return [RetrievedChunk(chunk=chunk, score=1.0, rank=1)]


def user_ctx(db: Session, name: str) -> UserContext:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    return UserContext.from_user(user)


def tools_for(db: Session, name: str) -> dict[str, BaseTool]:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    salaries = {e.employee_code: (15_000_000, "USD") for e in [user.employee]}
    payroll = PayrollClient(
        TestClient(create_mock_payroll_app(salaries, failure_rate=0, today=TODAY)),
        base_url="http://payroll.test",
    )
    deps = ToolDeps(
        db, UserContext.from_user(user), FakePolicies(), FakeHolidays(), FakeRates(), payroll, TODAY
    )
    return {t.name: t for t in build_tools(deps)}
