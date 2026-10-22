"""Answer an HR admin's statistics question: model writes SQL, code checks it, it runs.

    question -> model (structured: sql + explanation) -> sql_guard -> read-only runner
                      ^ one retry, told why the query was refused

Every query and every refusal is audited with the question and the SQL.
"""

from datetime import date
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from hrm.analytics.runner import AnalyticsResult, AnalyticsRunner
from hrm.analytics.schema import describe
from hrm.analytics.sql_guard import UnsafeQueryError, check_query
from hrm.auth.context import UserContext
from hrm.core.errors import HRMError
from hrm.db import repository as repo

ATTEMPTS = 2

_PROMPT = """\
You write one MySQL 8 SELECT query that answers an HR question. Today is {today}.

You can read only these views:
{schema}

Rules:
- One SELECT (WITH at the start is fine). Only the views above; no other tables.
- Money columns are in cents: divide by 100 for whole units.
- Salaries exist only as group statistics in v_salary_bands; never per person.
- Give columns readable aliases. Add ORDER BY when the order matters.
- If the question cannot be answered from these views, return a query that selects a
  single column named `note` with a short explanation as text.
"""


class SQLDraft(BaseModel):
    sql: str = Field(description="One MySQL SELECT query.")
    explanation: str = Field(description="One sentence: what the query computes.")


class SQLWriter(Protocol):
    """A chat model bound to SQLDraft (llm.with_structured_output(SQLDraft)), or a fake."""

    def invoke(self, input: list[BaseMessage]) -> Any: ...


class AnalyticsAnswer(AnalyticsResult):
    explanation: str


class AnalyticsQueryError(HRMError):
    """The checked query failed in the database (timeout, permission)."""


class AnalyticsService:
    def __init__(
        self,
        session: Session,
        user: UserContext,
        writer: SQLWriter,
        runner: AnalyticsRunner,
        today: date,
    ) -> None:
        self._session = session
        self._user = user
        self._writer = writer
        self._runner = runner
        self._today = today

    def answer(self, question: str) -> AnalyticsAnswer:
        draft, checked = self._write_checked_sql(question)
        try:
            result = self._runner.run(checked)
        except DBAPIError as exc:  # timeout, or the account may not read something
            self._audit("analytics.failed", question, checked, error=type(exc.orig).__name__)
            raise AnalyticsQueryError(
                "The query could not run (too slow or not permitted)."
            ) from exc
        self._audit("analytics.query", question, result.sql, rows=len(result.rows))
        return AnalyticsAnswer(**result.model_dump(), explanation=draft.explanation)

    def _write_checked_sql(self, question: str) -> tuple[SQLDraft, str]:
        messages: list[BaseMessage] = [
            SystemMessage(content=_PROMPT.format(today=self._today, schema=describe())),
            HumanMessage(content=question),
        ]
        for attempt in range(1, ATTEMPTS + 1):
            draft = self._writer.invoke(messages)
            if not isinstance(draft, SQLDraft):
                raise AnalyticsQueryError("The model did not return a query.")
            try:
                return draft, check_query(draft.sql)
            except UnsafeQueryError as exc:
                self._audit("analytics.refused", question, draft.sql, reason=str(exc))
                if attempt == ATTEMPTS:
                    raise
                messages.append(
                    HumanMessage(
                        content=f"This query was refused: {exc}\n{draft.sql}\n"
                        "Write a corrected query."
                    )
                )
        raise AssertionError("unreachable")

    def _audit(self, action: str, question: str, sql: str, **details: object) -> None:
        repo.record_audit(
            self._session,
            actor_user_id=self._user.user_id,
            action=action,
            entity="analytics",
            details={"question": question, "sql": sql, **details},
        )
