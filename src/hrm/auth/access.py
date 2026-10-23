"""The permission layer: the only way callers (API, workflows, agent tools) read HR data.

Rules (deny by default):
- Everyone may see their own profile, leave and salary.
- A direct manager may see a report's profile and leave, never their salary.
- An HR admin may see everyone's profile, leave and salary.
- Every salary read and every refused access is written to the audit log.
- "Not found" and "not allowed" raise the same error, so ids cannot be probed.

Results are view models that only contain what the caller may see; salary is never part
of a profile, it has its own audited call.
"""

from datetime import date
from typing import NoReturn

from pydantic import BaseModel
from sqlalchemy.orm import Session

from hrm.auth.context import UserContext
from hrm.core.errors import HRMError
from hrm.db import repository as repo
from hrm.db.models import Employee, LeaveRequest, RequestStatus


class AccessDeniedError(HRMError):
    """Not found, or not allowed (deliberately indistinguishable)."""


class EmployeeView(BaseModel):
    id: int
    employee_code: str
    name: str
    email: str
    job_title: str
    department: str
    country: str
    manager_name: str | None
    start_date: date

    @classmethod
    def of(cls, e: Employee) -> "EmployeeView":
        return cls(
            id=e.id,
            employee_code=e.employee_code,
            name=e.full_name,
            email=e.email,
            job_title=e.job_title,
            department=e.department.name,
            country=e.country,
            manager_name=e.manager.full_name if e.manager else None,
            start_date=e.start_date,
        )


class SalaryView(BaseModel):
    employee_id: int
    yearly_amount: float  # in `currency`, e.g. 145000.0
    currency: str
    effective_from: date


class LeaveRequestView(BaseModel):
    id: int
    employee_id: int
    leave_type: str
    start_date: date
    end_date: date
    working_days: float
    status: RequestStatus
    reason: str | None

    @classmethod
    def of(cls, r: LeaveRequest) -> "LeaveRequestView":
        return cls(
            id=r.id,
            employee_id=r.employee_id,
            leave_type=r.leave_type.code,
            start_date=r.start_date,
            end_date=r.end_date,
            working_days=float(r.working_days),
            status=r.status,
            reason=r.reason,
        )


class HRAccess:
    """HR data as seen by one user. Create one per request."""

    def __init__(self, session: Session, ctx: UserContext) -> None:
        self._session = session
        self._ctx = ctx

    # --- rules ----------------------------------------------------------------------------

    def can_view_work_data(self, employee_id: int) -> bool:
        """Profile and leave: self, direct manager, or HR admin."""
        return (
            employee_id == self._ctx.employee_id
            or self._ctx.is_hr_admin
            or repo.is_direct_manager(self._session, self._ctx.employee_id, employee_id)
        )

    def can_view_salary(self, employee_id: int) -> bool:
        """Salary: self or HR admin only. Managers never see their reports' pay."""
        return employee_id == self._ctx.employee_id or self._ctx.is_hr_admin

    # --- reads ----------------------------------------------------------------------------

    def me(self) -> EmployeeView:
        return self.employee(self._ctx.employee_id)

    def employee(self, employee_id: int) -> EmployeeView:
        return EmployeeView.of(self._allowed_employee(employee_id, "employee.read"))

    def my_team(self) -> list[EmployeeView]:
        return [
            EmployeeView.of(e) for e in repo.direct_reports(self._session, self._ctx.employee_id)
        ]

    def all_employees(self) -> list[EmployeeView]:
        """All active employees — HR admins only."""
        if not self._ctx.is_hr_admin:
            self._deny("employees.list", self._ctx.employee_id)
        return [EmployeeView.of(e) for e in repo.all_employees(self._session)]

    def leave_requests(
        self, employee_id: int, *, status: RequestStatus | None = None
    ) -> list[LeaveRequestView]:
        self._allowed_employee(employee_id, "leave.read")
        requests = repo.leave_requests_for(self._session, employee_id, status=status)
        return [LeaveRequestView.of(r) for r in requests]

    def pending_approvals(self) -> list[LeaveRequestView]:
        """Leave requests waiting for this user's decision."""
        requests = repo.pending_leave_for_approver(self._session, self._ctx.employee_id)
        return [LeaveRequestView.of(r) for r in requests]

    def salary(self, employee_id: int, *, reason: str, on: date | None = None) -> SalaryView:
        """Audited every time; `reason` is recorded (e.g. "annual review")."""
        if not self.can_view_salary(employee_id):
            self._deny("salary.read", employee_id)
        comp = repo.current_compensation(self._session, employee_id, on=on or date.today())
        if comp is None:
            self._deny("salary.read", employee_id)
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action="salary.read",
            entity="employee",
            entity_id=employee_id,
            details={"reason": reason},
        )
        return SalaryView(
            employee_id=employee_id,
            yearly_amount=comp.base_salary_cents / 100,
            currency=comp.currency,
            effective_from=comp.effective_from,
        )

    # --- helpers --------------------------------------------------------------------------

    def _allowed_employee(self, employee_id: int, action: str) -> Employee:
        employee = repo.get_employee(self._session, employee_id)
        if employee is None or not self.can_view_work_data(employee_id):
            self._deny(action, employee_id)
        return employee

    def _deny(self, action: str, employee_id: int) -> NoReturn:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action="access.denied",
            entity="employee",
            entity_id=employee_id,
            details={"attempted": action},
        )
        raise AccessDeniedError("Employee not found or not accessible.")
