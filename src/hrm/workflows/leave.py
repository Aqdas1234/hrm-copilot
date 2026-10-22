"""Leave actions: request, approve, reject, cancel.

These connect the pure rules (leave_rules) to the database and to permissions:
- anyone may request leave for themselves;
- only the assigned approver (the manager) or an HR admin may approve or reject;
- nobody may decide on their own request, not even an HR admin;
- only pending requests can be decided or cancelled, and only by the right person;
- approving adds the working days to the employee's balance for that year.
Every action is audited; decision notes are kept in the audit log.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import NoReturn, Protocol

from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError, LeaveRequestView
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.models import LeaveBalance, LeaveRequest, RequestStatus
from hrm.integrations.holidays import Holiday
from hrm.workflows.leave_rules import (
    LeaveContext,
    LeaveRequestDraft,
    LeaveRuleError,
    Period,
    Violation,
    plan_leave,
)


class HolidayLookup(Protocol):
    """HolidayService, or a fake in tests."""

    def holidays(self, country: str, year: int) -> list[Holiday]: ...


class LeaveOutcome(LeaveRequestView):
    warnings: list[str]


class LeaveService:
    def __init__(
        self, session: Session, ctx: UserContext, holidays: HolidayLookup, today: date
    ) -> None:
        self._session = session
        self._ctx = ctx
        self._holidays = holidays
        self._today = today

    def request(
        self, leave_code: str, start: date, end: date, reason: str | None = None
    ) -> LeaveOutcome:
        employee = repo.get_employee(self._session, self._ctx.employee_id)
        leave_type = repo.leave_type(self._session, leave_code)
        if employee is None or leave_type is None:
            raise LeaveRuleError(
                [Violation("unknown_leave_type", f"Unknown leave type '{leave_code}'.")]
            )
        if start.year != end.year:
            raise LeaveRuleError(
                [
                    Violation(
                        "spans_years",
                        "Please split leave that crosses the new year into two requests.",
                    )
                ]
            )

        balance = repo.leave_balance(self._session, employee.id, leave_code, start.year)
        holidays = frozenset(h.day for h in self._holidays.holidays(employee.country, start.year))
        context = LeaveContext(
            today=self._today,
            holidays=holidays,
            existing=tuple(
                Period(r.start_date, r.end_date)
                for r in repo.active_leave_for(self._session, employee.id)
            ),
            allowed_days=balance.allowed_days if balance else leave_type.annual_days,
            used_days=balance.used_days if balance else Decimal(0),
            manager_id=employee.manager_id,
        )
        plan = plan_leave(LeaveRequestDraft(leave_code, start, end), context)

        request = LeaveRequest(
            employee_id=employee.id,
            leave_type=leave_type,
            start_date=start,
            end_date=end,
            working_days=plan.day_count,
            reason=reason,
            status=RequestStatus.PENDING,
            approver_id=plan.approver_id,
        )
        self._session.add(request)
        self._session.flush()
        self._audit(
            "leave.request", request, {"days": str(plan.day_count), "warnings": plan.warnings}
        )
        return LeaveOutcome(**LeaveRequestView.of(request).model_dump(), warnings=plan.warnings)

    def approve(self, request_id: int, note: str | None = None) -> LeaveRequestView:
        request = self._decidable(request_id)
        balance = repo.leave_balance(
            self._session, request.employee_id, request.leave_type.code, request.start_date.year
        )
        if balance is None:  # first use of this leave type this year
            balance = LeaveBalance(
                employee_id=request.employee_id,
                leave_type_id=request.leave_type_id,
                year=request.start_date.year,
                allowed_days=request.leave_type.annual_days,
                used_days=Decimal(0),
            )
            self._session.add(balance)
        balance.used_days += request.working_days
        return self._decide(request, RequestStatus.APPROVED, "leave.approve", note)

    def reject(self, request_id: int, note: str) -> LeaveRequestView:
        return self._decide(
            self._decidable(request_id), RequestStatus.REJECTED, "leave.reject", note
        )

    def cancel(self, request_id: int) -> LeaveRequestView:
        request = repo.get_leave_request(self._session, request_id)
        if request is None or request.employee_id != self._ctx.employee_id:
            self._deny("leave.cancel", request_id)
        self._require_pending(request)
        return self._decide(request, RequestStatus.CANCELLED, "leave.cancel", None)

    # --- helpers --------------------------------------------------------------------------

    def _decidable(self, request_id: int) -> LeaveRequest:
        request = repo.get_leave_request(self._session, request_id)
        allowed = (
            request is not None
            and request.employee_id != self._ctx.employee_id
            and (request.approver_id == self._ctx.employee_id or self._ctx.is_hr_admin)
        )
        if request is None or not allowed:
            self._deny("leave.decide", request_id)
        self._require_pending(request)
        return request

    @staticmethod
    def _require_pending(request: LeaveRequest) -> None:
        if request.status is not RequestStatus.PENDING:
            raise LeaveRuleError(
                [
                    Violation(
                        "not_pending",
                        f"This request is already {request.status.value}.",
                    )
                ]
            )

    def _decide(
        self, request: LeaveRequest, status: RequestStatus, action: str, note: str | None
    ) -> LeaveRequestView:
        request.status = status
        request.decided_at = datetime.now(UTC)
        self._session.flush()
        self._audit(action, request, {"note": note} if note else {})
        return LeaveRequestView.of(request)

    def _audit(self, action: str, request: LeaveRequest, details: dict[str, object]) -> None:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action=action,
            entity="leave_request",
            entity_id=request.id,
            details=details,
        )

    def _deny(self, action: str, request_id: int) -> NoReturn:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action="access.denied",
            entity="leave_request",
            entity_id=request_id,
            details={"attempted": action},
        )
        raise AccessDeniedError("Leave request not found or not accessible.")
