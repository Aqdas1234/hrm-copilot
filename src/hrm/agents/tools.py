"""Tools the agent may call, built per request for one logged-in user.

Security model (each point is covered by tests):
- Identity is captured here, in code. No tool about "my" data takes a user or employee
  argument, so no prompt can redirect it to someone else's data.
- Tools are filtered by role: only managers and HR admins get the team tools.
- Every tool goes through the permission-checked services (HRAccess, LeaveService,
  ExpenseService); none touches the database directly.
- Tools that change data carry metadata {"writes": True}; the agent graph asks the user
  to confirm before running them.
- Expected failures (refused, rule broken, service down) come back as "ERROR: ..." text the
  agent can explain. Unexpected exceptions are bugs and are not hidden.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any, Literal, ParamSpec

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hrm.auth.access import HRAccess
from hrm.auth.context import UserContext
from hrm.core.errors import HRMError
from hrm.db import repository as repo
from hrm.db.models import RequestStatus, Role
from hrm.integrations.payroll import PayrollClient
from hrm.rag.tool import PolicySearch, make_policy_search_tool
from hrm.workflows.expenses import ExpenseService, RateLookup
from hrm.workflows.leave import HolidayLookup, LeaveService

WRITES = {"writes": True}
MAX_EXPENSE = Decimal("1000000")

LeaveCode = Literal["pto", "parental", "unpaid"]
P = ParamSpec("P")


@dataclass(frozen=True)
class ToolDeps:
    """Everything the tools need, prepared once per request by the caller."""

    session: Session
    user: UserContext
    policies: PolicySearch
    holidays: HolidayLookup
    rates: RateLookup
    payroll: PayrollClient
    today: date


def build_tools(deps: ToolDeps) -> list[BaseTool]:
    """The tools for this user: everyone's tools, plus team tools for managers and HR."""
    tools = [make_policy_search_tool(deps.policies), *_self_service_tools(deps)]
    if _manages_people(deps):
        tools += _team_tools(deps)
    return tools


def is_write(tool_: BaseTool) -> bool:
    return bool((tool_.metadata or {}).get("writes"))


def _mark_writes(*tools: BaseTool) -> None:
    """Tag tools that change data; the agent graph asks the user to confirm before them."""
    for t in tools:
        t.metadata = {**(t.metadata or {}), **WRITES}


# --- tools for everyone (about the logged-in user only) ------------------------------------


def _self_service_tools(deps: ToolDeps) -> list[BaseTool]:
    access = HRAccess(deps.session, deps.user)
    leave = LeaveService(deps.session, deps.user, deps.holidays, deps.today)
    expenses = ExpenseService(deps.session, deps.user, deps.rates, deps.today)

    @tool(parse_docstring=True)
    @_safe
    def get_my_profile() -> str:
        """Your own HR profile: name, job title, department, country, manager, start date."""
        return _json(access.me())

    @tool(parse_docstring=True)
    @_safe
    def get_my_salary() -> str:
        """Your own current base salary (yearly, in your local currency). Access is audited."""
        return _json(
            access.salary(deps.user.employee_id, reason="self-service via assistant", on=deps.today)
        )

    @tool(parse_docstring=True)
    @_safe
    def get_my_leave_balance(leave_type: LeaveCode = "pto", year: int | None = None) -> str:
        """How many days of a leave type you have used and how many you are allowed.

        Args:
            leave_type: "pto" (flexible paid time off, no set limit), "parental" or "unpaid".
            year: Calendar year; defaults to the current year.
        """
        year = year or deps.today.year
        balance = repo.leave_balance(deps.session, deps.user.employee_id, leave_type, year)
        return _json(
            {
                "leave_type": leave_type,
                "year": year,
                "used_days": float(balance.used_days) if balance else 0.0,
                "allowed_days": float(balance.allowed_days)
                if balance and balance.allowed_days is not None
                else None,
                "note": "allowed_days null means no set limit" if leave_type == "pto" else None,
            }
        )

    @tool(parse_docstring=True)
    @_safe
    def list_my_leave_requests(
        status: Literal["pending", "approved", "rejected", "cancelled"] | None = None,
    ) -> str:
        """Your leave requests, newest first.

        Args:
            status: Only requests with this status; all if omitted.
        """
        found = access.leave_requests(
            deps.user.employee_id, status=RequestStatus(status) if status else None
        )
        return _json(found)

    @tool(parse_docstring=True)
    @_safe
    def get_public_holidays(year: int | None = None) -> str:
        """Public holidays in your country (the ones that do not count as leave days).

        Args:
            year: Calendar year; defaults to the current year.
        """
        country = access.me().country
        holidays = deps.holidays.holidays(country, year or deps.today.year)
        return _json(
            {"country": country, "holidays": [{"date": h.day, "name": h.name} for h in holidays]}
        )

    @tool(parse_docstring=True)
    @_safe
    def check_leave_request(leave_type: LeaveCode, start_date: str, end_date: str) -> str:
        """Check a leave request WITHOUT filing it: working days, warnings, or why not allowed.
        Always use this before request_leave and show the result to the user.

        Args:
            leave_type: "pto", "parental" or "unpaid".
            start_date: First day off, YYYY-MM-DD.
            end_date: Last day off, YYYY-MM-DD.
        """
        return _json(leave.preview(leave_type, _date(start_date), _date(end_date)))

    @tool(parse_docstring=True)
    @_safe
    def request_leave(
        leave_type: LeaveCode, start_date: str, end_date: str, reason: str | None = None
    ) -> str:
        """File a leave request for your manager to approve. Changes data: only after the user
        has seen the check_leave_request result and agreed.

        Args:
            leave_type: "pto", "parental" or "unpaid".
            start_date: First day off, YYYY-MM-DD.
            end_date: Last day off, YYYY-MM-DD.
            reason: Optional short reason for your manager.
        """
        return _json(
            leave.request(leave_type, _date(start_date), _date(end_date), _text(reason, 300))
        )

    @tool(parse_docstring=True)
    @_safe
    def cancel_my_leave_request(request_id: int) -> str:
        """Cancel one of your own pending leave requests. Changes data.

        Args:
            request_id: The id from list_my_leave_requests.
        """
        return _json(leave.cancel(request_id))

    @tool(parse_docstring=True)
    @_safe
    def list_my_expense_claims() -> str:
        """Your expense claims, newest purchase first, with amounts in USD."""
        return _json(expenses.my_claims())

    @tool(parse_docstring=True)
    @_safe
    def submit_expense_claim(
        description: str, amount: str, currency: str, purchase_date: str
    ) -> str:
        """Submit an expense claim for reimbursement. Changes data: confirm the details first.

        Args:
            description: What was bought, e.g. "External monitor".
            amount: The amount paid in the original currency, e.g. "249.99".
            currency: 3-letter currency code, e.g. "USD", "EUR", "PKR".
            purchase_date: Date of purchase, YYYY-MM-DD (must be within the last 60 days).
        """
        return _json(
            expenses.submit(
                _text(description, 200) or "", _cents(amount), currency, _date(purchase_date)
            )
        )

    @tool(parse_docstring=True)
    @_safe
    def get_my_payslip(year: int, month: int) -> str:
        """Your payslip for a past month: gross, tax and net pay.

        Args:
            year: e.g. 2026.
            month: 1-12. The current month is not paid yet.
        """
        return _json(deps.payroll.payslip(access.me().employee_code, year, month))

    _mark_writes(request_leave, cancel_my_leave_request, submit_expense_claim)
    return [
        get_my_profile,
        get_my_salary,
        get_my_leave_balance,
        list_my_leave_requests,
        get_public_holidays,
        check_leave_request,
        request_leave,
        cancel_my_leave_request,
        list_my_expense_claims,
        submit_expense_claim,
        get_my_payslip,
    ]


# --- tools for managers and HR admins ------------------------------------------------------


def _team_tools(deps: ToolDeps) -> list[BaseTool]:
    access = HRAccess(deps.session, deps.user)
    leave = LeaveService(deps.session, deps.user, deps.holidays, deps.today)
    expenses = ExpenseService(deps.session, deps.user, deps.rates, deps.today)

    @tool(parse_docstring=True)
    @_safe
    def list_my_team() -> str:
        """Your direct reports (profiles only; salaries are never shown to managers)."""
        return _json(access.my_team())

    @tool(parse_docstring=True)
    @_safe
    def list_pending_approvals() -> str:
        """Leave requests and expense claims waiting for your decision."""
        return _json(
            {
                "leave_requests": access.pending_approvals(),
                "expense_claims": expenses.pending_approvals(),
            }
        )

    @tool(parse_docstring=True)
    @_safe
    def decide_leave_request(
        request_id: int, decision: Literal["approve", "reject"], note: str | None = None
    ) -> str:
        """Approve or reject a leave request waiting for you. Changes data.

        Args:
            request_id: The id from list_pending_approvals.
            decision: "approve" or "reject".
            note: Reason for the employee; required when rejecting.
        """
        if decision == "approve":
            return _json(leave.approve(request_id, _text(note, 300)))
        return _json(leave.reject(request_id, _required_note(note)))

    @tool(parse_docstring=True)
    @_safe
    def decide_expense_claim(
        claim_id: int, decision: Literal["approve", "reject"], note: str | None = None
    ) -> str:
        """Approve or reject an expense claim waiting for you. Changes data.

        Args:
            claim_id: The id from list_pending_approvals.
            decision: "approve" or "reject".
            note: Reason for the employee; required when rejecting.
        """
        if decision == "approve":
            return _json(expenses.approve(claim_id, _text(note, 300)))
        return _json(expenses.reject(claim_id, _required_note(note)))

    _mark_writes(decide_leave_request, decide_expense_claim)
    return [list_my_team, list_pending_approvals, decide_leave_request, decide_expense_claim]


def _manages_people(deps: ToolDeps) -> bool:
    if deps.user.role in (Role.MANAGER, Role.HR_ADMIN):
        return True
    return bool(repo.direct_reports(deps.session, deps.user.employee_id))


# --- helpers ---------------------------------------------------------------------------------


class ToolInputError(HRMError):
    """An argument the model passed is not usable (bad date, amount, ...)."""


def _safe(fn: Callable[P, str]) -> Callable[P, str]:
    """Turn expected failures into a message the agent can relay; let bugs surface."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except HRMError as exc:
            return f"ERROR: {exc}"

    return wrapper


def _json(value: Any) -> str:
    def encode(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, (date, Decimal)):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        raise TypeError(f"Cannot encode {type(obj).__name__}")

    if isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    return json.dumps(value, default=encode, separators=(",", ":"))


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ToolInputError(f"'{value}' is not a date; use YYYY-MM-DD.") from exc


def _cents(amount: str) -> int:
    try:
        value = Decimal(amount.strip().replace(",", ""))
    except InvalidOperation as exc:
        raise ToolInputError(f"'{amount}' is not an amount; use digits like 249.99.") from exc
    if not Decimal(0) < value <= MAX_EXPENSE:
        raise ToolInputError("The amount must be above 0 and at most 1,000,000.")
    return int((value * 100).quantize(Decimal(1)))


def _text(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) > limit:
        raise ToolInputError(f"Please keep it under {limit} characters.")
    return value or None


def _required_note(note: str | None) -> str:
    text = _text(note, 300)
    if not text:
        raise ToolInputError("A rejection needs a short reason for the employee.")
    return text
