"""Repository queries against the deterministic seed (SQLite in memory)."""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from hrm.db import repository as repo
from hrm.db.models import Compensation, RequestStatus
from hrm.db.seed import seed

TODAY = date(2026, 10, 22)


@pytest.fixture
def db(session: Session) -> Session:
    seed(session, today=TODAY, demo_password="x")
    return session


def ids(db: Session) -> dict[str, int]:
    found = {}
    for name in ["alice.park", "bob.chen", "hira.khan", "carol.reyes"]:
        user = repo.get_user_by_email(db, f"{name}@example.com")
        assert user is not None
        found[name.split(".")[0]] = user.employee_id
    return found


def test_get_user_by_email_is_case_and_space_insensitive(db: Session) -> None:
    user = repo.get_user_by_email(db, "  Alice.Park@Example.com ")

    assert user is not None and user.employee.first_name == "Alice"
    assert repo.get_user_by_email(db, "nobody@example.com") is None


def test_direct_reports_and_direct_manager_check(db: Session) -> None:
    p = ids(db)

    reports = {e.first_name for e in repo.direct_reports(db, p["bob"])}

    assert {"Alice", "Hira"} <= reports
    assert repo.is_direct_manager(db, p["bob"], p["alice"])
    assert not repo.is_direct_manager(db, p["alice"], p["bob"])  # not the other way round
    assert not repo.is_direct_manager(db, p["carol"], p["alice"])  # HR is not Alice's manager


def test_current_compensation_picks_the_latest_effective_row(db: Session) -> None:
    p = ids(db)
    db.add(
        Compensation(
            employee_id=p["alice"],
            base_salary_cents=99_900_000,
            currency="USD",
            effective_from=date(2027, 1, 1),
        )
    )  # a future raise
    db.flush()

    now = repo.current_compensation(db, p["alice"], on=TODAY)
    later = repo.current_compensation(db, p["alice"], on=date(2027, 6, 1))

    assert now is not None and now.base_salary_cents != 99_900_000
    assert later is not None and later.base_salary_cents == 99_900_000


def test_leave_balance_and_types(db: Session) -> None:
    p = ids(db)

    pto = repo.leave_balance(db, p["alice"], "pto", TODAY.year)
    parental = repo.leave_balance(db, p["alice"], "parental", TODAY.year)

    assert pto is not None and pto.allowed_days is None  # flexible PTO
    assert parental is not None and parental.allowed_days == 80
    assert repo.leave_balance(db, p["alice"], "pto", 1999) is None
    assert repo.leave_type(db, "unpaid") is not None


def test_leave_requests_by_status_and_pending_for_the_approver(db: Session) -> None:
    p = ids(db)

    alice_all = repo.leave_requests_for(db, p["alice"])
    alice_pending = repo.leave_requests_for(db, p["alice"], status=RequestStatus.PENDING)
    bob_queue = repo.pending_leave_for_approver(db, p["bob"])

    assert len(alice_all) == 2 and len(alice_pending) == 1
    assert {r.employee_id for r in bob_queue} == {p["alice"], p["hira"]}
    assert all(r.status is RequestStatus.PENDING for r in bob_queue)
    assert repo.pending_leave_for_approver(db, p["alice"]) == []


def test_holidays_between_dates_for_a_country(db: Session) -> None:
    found = repo.holidays_between(db, "pk", date(2026, 8, 1), date(2026, 12, 31))

    assert [h.name for h in found] == ["Independence Day", "Iqbal Day", "Quaid-e-Azam Day"]
    assert repo.holidays_between(db, "US", date(2026, 1, 1), date(2026, 12, 31)) == []


def test_expense_claims_newest_first(db: Session) -> None:
    claims = repo.expense_claims_for(db, ids(db)["alice"])

    assert [c.description for c in claims] == ["Laptop stand"]


def test_audit_entries_are_appended_and_read_back_in_order(db: Session) -> None:
    p = ids(db)
    carol_user = repo.get_user_by_email(db, "carol.reyes@example.com")
    assert carol_user is not None

    repo.record_audit(
        db,
        actor_user_id=carol_user.id,
        action="salary.read",
        entity="employee",
        entity_id=p["alice"],
        details={"reason": "annual review"},
    )
    repo.record_audit(
        db,
        actor_user_id=carol_user.id,
        action="employee.read",
        entity="employee",
        entity_id=p["alice"],
    )

    trail = repo.audit_trail(db, entity="employee", entity_id=p["alice"])
    assert [e.action for e in trail] == ["salary.read", "employee.read"]
    assert trail[0].details == {"reason": "annual review"}
    assert trail[1].details == {}
