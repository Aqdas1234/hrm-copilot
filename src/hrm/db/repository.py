"""Data access: small, typed queries over the HR database.

All SQL lives here, so permissions, workflows and agent tools never build queries themselves.
These functions do NOT check permissions; they are raw access. Callers must go through the
permission layer (hrm.auth), which decides who may see what. Sensitive reads (salary) are
marked as such and must be audited by the caller.
"""

from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.db.models import (
    AuditLog,
    Compensation,
    Employee,
    ExchangeRate,
    ExpenseClaim,
    LeaveBalance,
    LeaveRequest,
    LeaveType,
    PublicHoliday,
    RateSource,
    RequestStatus,
    User,
)

# --- people -----------------------------------------------------------------------------


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.strip().lower()))


def get_employee(session: Session, employee_id: int) -> Employee | None:
    return session.get(Employee, employee_id)


def direct_reports(session: Session, manager_id: int) -> list[Employee]:
    return list(
        session.scalars(
            select(Employee)
            .where(Employee.manager_id == manager_id, Employee.is_active)
            .order_by(Employee.last_name, Employee.first_name)
        )
    )


def is_direct_manager(session: Session, manager_id: int, employee_id: int) -> bool:
    employee = session.get(Employee, employee_id)
    return employee is not None and employee.manager_id == manager_id


def current_compensation(session: Session, employee_id: int, on: date) -> Compensation | None:
    """SENSITIVE: salary. Only for HR admins, and every call must be audited by the caller."""
    return session.scalar(
        select(Compensation)
        .where(Compensation.employee_id == employee_id, Compensation.effective_from <= on)
        .order_by(Compensation.effective_from.desc())
        .limit(1)
    )


# --- leave ------------------------------------------------------------------------------


def leave_type(session: Session, code: str) -> LeaveType | None:
    return session.scalar(select(LeaveType).where(LeaveType.code == code))


def leave_balance(session: Session, employee_id: int, code: str, year: int) -> LeaveBalance | None:
    return session.scalar(
        select(LeaveBalance)
        .join(LeaveType)
        .where(
            LeaveBalance.employee_id == employee_id,
            LeaveType.code == code,
            LeaveBalance.year == year,
        )
    )


def leave_requests_for(
    session: Session, employee_id: int, *, status: RequestStatus | None = None
) -> list[LeaveRequest]:
    query = select(LeaveRequest).where(LeaveRequest.employee_id == employee_id)
    if status is not None:
        query = query.where(LeaveRequest.status == status)
    return list(session.scalars(query.order_by(LeaveRequest.start_date.desc())))


def pending_leave_for_approver(session: Session, approver_id: int) -> list[LeaveRequest]:
    return list(
        session.scalars(
            select(LeaveRequest)
            .where(
                LeaveRequest.approver_id == approver_id,
                LeaveRequest.status == RequestStatus.PENDING,
            )
            .order_by(LeaveRequest.start_date)
        )
    )


def holidays_between(session: Session, country: str, start: date, end: date) -> list[PublicHoliday]:
    return list(
        session.scalars(
            select(PublicHoliday)
            .where(
                PublicHoliday.country == country.upper(),
                PublicHoliday.day >= start,
                PublicHoliday.day <= end,
            )
            .order_by(PublicHoliday.day)
        )
    )


def add_holidays(session: Session, holidays: list[PublicHoliday]) -> None:
    session.add_all(holidays)
    session.flush()


# --- exchange rates ---------------------------------------------------------------------


def cached_rate(session: Session, currency: str, day: date) -> ExchangeRate | None:
    return session.scalar(
        select(ExchangeRate).where(
            ExchangeRate.currency == currency,
            ExchangeRate.day == day,
            ExchangeRate.source == RateSource.API,
        )
    )


def manual_rate(session: Session, currency: str, on: date) -> ExchangeRate | None:
    """The newest manually set rate valid on `on` (set on or before that day)."""
    return session.scalar(
        select(ExchangeRate)
        .where(
            ExchangeRate.currency == currency,
            ExchangeRate.source == RateSource.MANUAL,
            ExchangeRate.day <= on,
        )
        .order_by(ExchangeRate.day.desc())
        .limit(1)
    )


def add_rate(session: Session, rate: ExchangeRate) -> None:
    session.add(rate)
    session.flush()


# --- expenses -----------------------------------------------------------------------------


def expense_claims_for(session: Session, employee_id: int) -> list[ExpenseClaim]:
    return list(
        session.scalars(
            select(ExpenseClaim)
            .where(ExpenseClaim.employee_id == employee_id)
            .order_by(ExpenseClaim.purchase_date.desc())
        )
    )


# --- audit --------------------------------------------------------------------------------


def record_audit(
    session: Session,
    *,
    actor_user_id: int | None,
    action: str,
    entity: str,
    entity_id: int | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """Append one audit entry ("who did or saw what"). The log is never updated or deleted."""
    entry = AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        entity=entity,
        entity_id=entity_id,
        details=details or {},
    )
    session.add(entry)
    session.flush()
    return entry


def audit_trail(session: Session, *, entity: str, entity_id: int) -> list[AuditLog]:
    return list(
        session.scalars(
            select(AuditLog)
            .where(AuditLog.entity == entity, AuditLog.entity_id == entity_id)
            .order_by(AuditLog.at, AuditLog.id)
        )
    )
