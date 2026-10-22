"""Every cell of the access rules, including the requests that must be refused."""

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError, EmployeeView, HRAccess
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.models import AuditLog, RequestStatus
from hrm.db.seed import seed

TODAY = date(2026, 10, 22)


@pytest.fixture
def db(session: Session) -> Session:
    seed(session, today=TODAY, demo_password="x")
    return session


def as_user(db: Session, name: str) -> HRAccess:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    return HRAccess(db, UserContext.from_user(user))


def emp_id(db: Session, name: str) -> int:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    return user.employee_id


def audit_actions(db: Session) -> list[str]:
    return [a for a in db.scalars(select(AuditLog.action).order_by(AuditLog.id)) if a != "db.seed"]


# --- employee (Alice) ---------------------------------------------------------------------


def test_employee_sees_own_profile_leave_and_salary(db: Session) -> None:
    alice = as_user(db, "alice.park")

    assert alice.me().name == "Alice Park"
    assert alice.me().manager_name == "Bob Chen"
    assert len(alice.leave_requests(emp_id(db, "alice.park"))) == 2
    assert (
        alice.salary(emp_id(db, "alice.park"), reason="checking my pay", on=TODAY).currency == "USD"
    )


@pytest.mark.parametrize("other", ["bob.chen", "hira.khan", "carol.reyes"])
def test_employee_cannot_see_anyone_else(db: Session, other: str) -> None:
    alice = as_user(db, "alice.park")

    with pytest.raises(AccessDeniedError):
        alice.employee(emp_id(db, other))
    with pytest.raises(AccessDeniedError):
        alice.leave_requests(emp_id(db, other))
    with pytest.raises(AccessDeniedError):
        alice.salary(emp_id(db, other), reason="curious", on=TODAY)


# --- manager (Bob) ------------------------------------------------------------------------


def test_manager_sees_reports_profile_and_leave(db: Session) -> None:
    bob = as_user(db, "bob.chen")

    assert bob.employee(emp_id(db, "alice.park")).name == "Alice Park"
    pending = bob.leave_requests(emp_id(db, "hira.khan"), status=RequestStatus.PENDING)
    assert len(pending) == 1
    assert {v.name for v in bob.my_team()} >= {"Alice Park", "Hira Khan"}
    assert {r.employee_id for r in bob.pending_approvals()} == {
        emp_id(db, "alice.park"),
        emp_id(db, "hira.khan"),
    }


def test_manager_never_sees_reports_salary(db: Session) -> None:
    bob = as_user(db, "bob.chen")

    with pytest.raises(AccessDeniedError):
        bob.salary(emp_id(db, "alice.park"), reason="raise planning", on=TODAY)


def test_manager_cannot_see_people_outside_the_team(db: Session) -> None:
    bob = as_user(db, "bob.chen")

    with pytest.raises(AccessDeniedError):
        bob.employee(emp_id(db, "carol.reyes"))


# --- HR admin (Carol) -----------------------------------------------------------------------


def test_hr_admin_sees_everyone_including_salary_and_it_is_audited(db: Session) -> None:
    carol = as_user(db, "carol.reyes")

    assert carol.employee(emp_id(db, "hira.khan")).country == "PK"
    salary = carol.salary(emp_id(db, "hira.khan"), reason="annual review", on=TODAY)

    assert salary.currency == "PKR" and salary.yearly_amount >= 3_000_000
    entry = db.scalar(select(AuditLog).where(AuditLog.action == "salary.read"))
    assert entry is not None
    assert entry.entity_id == emp_id(db, "hira.khan")
    assert entry.details == {"reason": "annual review"}


# --- cross-cutting -------------------------------------------------------------------------


def test_profile_view_never_contains_salary() -> None:
    assert not any("salary" in f or "compensation" in f for f in EmployeeView.model_fields)


def test_unknown_and_forbidden_ids_raise_the_same_error(db: Session) -> None:
    alice = as_user(db, "alice.park")

    with pytest.raises(AccessDeniedError) as unknown:
        alice.employee(999_999)
    with pytest.raises(AccessDeniedError) as forbidden:
        alice.employee(emp_id(db, "carol.reyes"))

    assert str(unknown.value) == str(forbidden.value)  # cannot probe which ids exist


def test_every_refusal_is_audited_with_what_was_attempted(db: Session) -> None:
    alice = as_user(db, "alice.park")
    target = emp_id(db, "bob.chen")

    for attempt in (alice.employee, lambda i: alice.salary(i, reason="x", on=TODAY)):
        with pytest.raises(AccessDeniedError):
            attempt(target)

    denied = db.scalars(select(AuditLog).where(AuditLog.action == "access.denied")).all()
    assert [d.details["attempted"] for d in denied] == ["employee.read", "salary.read"]
    assert all(d.entity_id == target for d in denied)


def test_ordinary_reads_are_not_audited_only_sensitive_ones(db: Session) -> None:
    alice = as_user(db, "alice.park")

    alice.me()
    alice.leave_requests(emp_id(db, "alice.park"))

    assert audit_actions(db) == []
