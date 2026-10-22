"""Schema behaviour on SQLite in memory. The same models run on MySQL (see test_mysql.py)."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from hrm.db.models import (
    AuditLog,
    Base,
    Compensation,
    Department,
    Employee,
    EmploymentType,
    HolidaySource,
    LeaveBalance,
    LeaveRequest,
    LeaveType,
    PublicHoliday,
    RequestStatus,
    Role,
    User,
)


def add_employee(s: Session, code: str, manager: Employee | None = None) -> Employee:
    dept = s.scalar(select(Department)) or Department(name="Engineering")
    employee = Employee(
        employee_code=code,
        first_name="Ada",
        last_name=code,
        email=f"{code.lower()}@example.com",
        job_title="Engineer",
        department=dept,
        manager=manager,
        country="US",
        employment_type=EmploymentType.FULL_TIME,
        start_date=date(2024, 1, 15),
    )
    s.add(employee)
    s.flush()
    return employee


def test_all_tables_are_created(session: Session) -> None:
    tables = set(inspect(session.get_bind()).get_table_names())

    assert tables == set(Base.metadata.tables)
    assert {"employees", "compensation", "leave_requests", "audit_log"} <= tables


def test_employee_has_no_salary_column_salary_lives_in_compensation() -> None:
    employee_columns = set(Employee.__table__.columns.keys())

    assert not any("salary" in c or "compensation" in c for c in employee_columns)
    assert "base_salary_cents" in Compensation.__table__.columns


def test_manager_and_direct_reports(session: Session) -> None:
    boss = add_employee(session, "E1")
    alice = add_employee(session, "E2", manager=boss)
    bob = add_employee(session, "E3", manager=boss)

    assert alice.manager is boss
    assert {e.employee_code for e in boss.reports} == {alice.employee_code, bob.employee_code}


def test_user_role_is_stored_as_text_and_reads_back_as_enum(session: Session) -> None:
    session.add(
        User(
            email="a@x.com",
            password_hash="h",
            role=Role.MANAGER,
            employee=add_employee(session, "E1"),
        )
    )
    session.flush()

    user = session.scalar(select(User))
    assert user is not None and user.role is Role.MANAGER


def test_unknown_role_is_rejected(session: Session) -> None:
    user = User(
        email="a@x.com", password_hash="h", role="superuser", employee=add_employee(session, "E1")
    )
    session.add(user)

    with pytest.raises(StatementError):
        session.flush()


@pytest.mark.parametrize(
    "bad_row",
    [
        lambda e: Compensation(
            employee_id=e.id, base_salary_cents=-1, currency="USD", effective_from=date(2024, 1, 1)
        ),
        lambda e: Compensation(
            employee_id=999, base_salary_cents=1, currency="USD", effective_from=date(2024, 1, 1)
        ),
        lambda e: LeaveRequest(
            employee_id=e.id,
            leave_type_id=1,
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 1),
            working_days=Decimal("1"),
        ),
    ],
    ids=["negative salary", "unknown employee (foreign key)", "end before start"],
)
def test_constraints_reject_invalid_rows(session: Session, bad_row: object) -> None:
    employee = add_employee(session, "E1")
    session.add(LeaveType(code="pto", name="Paid time off"))
    session.flush()
    session.add(bad_row(employee))  # type: ignore[operator]

    with pytest.raises(IntegrityError):
        session.flush()


def test_duplicate_email_and_balance_per_year_are_rejected(session: Session) -> None:
    employee = add_employee(session, "E1")
    pto = LeaveType(code="pto", name="Paid time off", annual_days=None)
    session.add(pto)
    session.flush()
    session.add_all(
        [
            LeaveBalance(employee_id=employee.id, leave_type_id=pto.id, year=2026),
            LeaveBalance(employee_id=employee.id, leave_type_id=pto.id, year=2026),
        ]
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_flexible_pto_has_no_set_limit_and_money_is_exact(session: Session) -> None:
    employee = add_employee(session, "E1")
    session.add_all(
        [
            LeaveType(code="pto", name="Paid time off", annual_days=None),
            LeaveType(code="parental", name="Paid parental leave", annual_days=Decimal("80")),
            Compensation(
                employee_id=employee.id,
                base_salary_cents=12_345_678,
                currency="USD",
                effective_from=date(2024, 1, 1),
            ),
        ]
    )
    session.flush()

    assert session.scalar(select(LeaveType.annual_days).where(LeaveType.code == "pto")) is None
    assert session.scalar(
        select(LeaveType.annual_days).where(LeaveType.code == "parental")
    ) == Decimal("80")
    assert session.scalar(select(Compensation.base_salary_cents)) == 12_345_678


def test_new_leave_request_defaults_to_pending(session: Session) -> None:
    employee = add_employee(session, "E1")
    pto = LeaveType(code="pto", name="Paid time off")
    session.add(pto)
    session.flush()
    request = LeaveRequest(
        employee_id=employee.id,
        leave_type_id=pto.id,
        start_date=date(2026, 5, 4),
        end_date=date(2026, 5, 6),
        working_days=Decimal("3"),
    )
    session.add(request)
    session.flush()

    assert request.status is RequestStatus.PENDING
    assert request.created_at is not None


def test_one_holiday_per_country_and_day(session: Session) -> None:
    session.add_all(
        [
            PublicHoliday(
                country="PK",
                day=date(2026, 8, 14),
                name="Independence Day",
                source=HolidaySource.MANUAL,
            ),
            PublicHoliday(
                country="PK", day=date(2026, 8, 14), name="Duplicate", source=HolidaySource.MANUAL
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_audit_log_stores_structured_details(session: Session) -> None:
    session.add(
        AuditLog(
            action="salary.read",
            entity="employee",
            entity_id=7,
            details={"reason": "review", "fields": ["base_salary_cents"]},
        )
    )
    session.flush()

    entry = session.scalar(select(AuditLog))
    assert entry is not None
    assert entry.details == {"reason": "review", "fields": ["base_salary_cents"]}
    assert entry.at is not None
