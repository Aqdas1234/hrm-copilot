"""Leave actions on the seeded database. Today: Thursday 2026-10-22.

Seed facts used here: Alice and Hira report to Bob; Alice has a pending request for
Thu 12 - Fri 13 Nov; Carol is HR admin.
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.models import AuditLog, RequestStatus
from hrm.db.seed import seed
from hrm.workflows.leave import LeaveService
from hrm.workflows.leave_rules import LeaveRuleError
from tests.fakes import FakeHolidays

TODAY = date(2026, 10, 22)


@pytest.fixture
def db(session: Session) -> Session:
    seed(session, today=TODAY, demo_password="x")
    return session


def as_user(db: Session, name: str) -> LeaveService:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    return LeaveService(db, UserContext.from_user(user), FakeHolidays(), TODAY)


def pending_of(db: Session, name: str) -> int:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    [request] = repo.leave_requests_for(db, user.employee_id, status=RequestStatus.PENDING)
    return request.id


def used_pto(db: Session, name: str) -> Decimal:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    balance = repo.leave_balance(db, user.employee_id, "pto", 2026)
    assert balance is not None
    return balance.used_days


def rule_codes(error: pytest.ExceptionInfo[LeaveRuleError]) -> list[str]:
    return [v.code for v in error.value.violations]


# --- the happy path ------------------------------------------------------------------------


def test_request_then_manager_approves_and_the_balance_is_updated(db: Session) -> None:
    before = used_pto(db, "alice.park")

    outcome = as_user(db, "alice.park").request(
        "pto", date(2026, 11, 2), date(2026, 11, 4), "Visiting family"
    )
    assert (
        outcome.status is RequestStatus.PENDING
        and outcome.working_days == 3
        and outcome.warnings == []
    )

    approved = as_user(db, "bob.chen").approve(outcome.id, note="Enjoy!")

    assert approved.status is RequestStatus.APPROVED
    assert used_pto(db, "alice.park") == before + 3
    actions = db.scalars(
        select(AuditLog.action).where(
            AuditLog.entity_id == outcome.id, AuditLog.entity == "leave_request"
        )
    ).all()
    assert actions == ["leave.request", "leave.approve"]


def test_holidays_of_the_employees_country_are_not_counted(db: Session) -> None:
    outcome = as_user(db, "hira.khan").request("pto", date(2026, 11, 9), date(2026, 11, 11))

    assert outcome.working_days == 2  # Iqbal Day (Mon 9 Nov) is a holiday in Pakistan


def test_short_notice_request_is_stored_with_its_warning(db: Session) -> None:
    outcome = as_user(db, "alice.park").request("pto", date(2026, 10, 26), date(2026, 10, 26))

    assert any("Short notice" in w for w in outcome.warnings)
    entry = db.scalar(select(AuditLog).where(AuditLog.action == "leave.request"))
    assert entry is not None and entry.details["warnings"] == outcome.warnings


def test_reject_with_a_note_does_not_touch_the_balance(db: Session) -> None:
    before = used_pto(db, "alice.park")

    rejected = as_user(db, "bob.chen").reject(pending_of(db, "alice.park"), note="Release week")

    assert rejected.status is RequestStatus.REJECTED
    assert used_pto(db, "alice.park") == before
    entry = db.scalar(select(AuditLog).where(AuditLog.action == "leave.reject"))
    assert entry is not None and entry.details == {"note": "Release week"}


def test_employee_cancels_own_pending_request(db: Session) -> None:
    cancelled = as_user(db, "alice.park").cancel(pending_of(db, "alice.park"))

    assert cancelled.status is RequestStatus.CANCELLED


def test_hr_admin_may_decide_for_any_team(db: Session) -> None:
    approved = as_user(db, "carol.reyes").approve(pending_of(db, "hira.khan"))

    assert approved.status is RequestStatus.APPROVED


# --- permission boundaries ----------------------------------------------------------------


def test_nobody_decides_on_their_own_request(db: Session) -> None:
    alice = as_user(db, "alice.park")
    request_id = pending_of(db, "alice.park")

    with pytest.raises(AccessDeniedError):
        alice.approve(request_id)


def test_hr_admin_cannot_approve_their_own_request(db: Session) -> None:
    carol = as_user(db, "carol.reyes")
    own = carol.request("pto", date(2026, 11, 2), date(2026, 11, 3))

    with pytest.raises(AccessDeniedError):
        carol.approve(own.id)


def test_colleague_cannot_approve_or_cancel_someone_elses_request(db: Session) -> None:
    hira = as_user(db, "hira.khan")
    alice_request = pending_of(db, "alice.park")

    with pytest.raises(AccessDeniedError):
        hira.approve(alice_request)
    with pytest.raises(AccessDeniedError):
        hira.cancel(alice_request)
    denied = db.scalars(select(AuditLog).where(AuditLog.action == "access.denied")).all()
    assert [d.details["attempted"] for d in denied] == ["leave.decide", "leave.cancel"]


def test_unknown_request_looks_the_same_as_a_forbidden_one(db: Session) -> None:
    with pytest.raises(AccessDeniedError) as unknown:
        as_user(db, "bob.chen").approve(999_999)
    with pytest.raises(AccessDeniedError) as forbidden:
        as_user(db, "hira.khan").approve(pending_of(db, "alice.park"))

    assert str(unknown.value) == str(forbidden.value)


# --- rule violations ------------------------------------------------------------------------


def test_cannot_decide_twice(db: Session) -> None:
    bob = as_user(db, "bob.chen")
    request_id = pending_of(db, "alice.park")
    bob.approve(request_id)

    with pytest.raises(LeaveRuleError) as caught:
        bob.reject(request_id, note="changed my mind")

    assert rule_codes(caught) == ["not_pending"]


def test_overlap_with_the_seeded_pending_request(db: Session) -> None:
    with pytest.raises(LeaveRuleError) as caught:
        as_user(db, "alice.park").request("pto", date(2026, 11, 13), date(2026, 11, 16))

    assert rule_codes(caught) == ["overlap"]


def test_cancelled_requests_no_longer_block_new_ones(db: Session) -> None:
    alice = as_user(db, "alice.park")
    alice.cancel(pending_of(db, "alice.park"))

    assert alice.request("pto", date(2026, 11, 13), date(2026, 11, 16)).working_days == 2


def test_parental_leave_is_limited_to_the_allowance(db: Session) -> None:
    alice = repo.get_user_by_email(db, "alice.park@example.com")
    assert alice is not None
    balance = repo.leave_balance(db, alice.employee_id, "parental", 2026)
    assert balance is not None
    balance.used_days = Decimal(78)  # 2 of 80 days left

    with pytest.raises(LeaveRuleError) as caught:
        as_user(db, "alice.park").request(
            "parental", date(2026, 11, 2), date(2026, 11, 6)
        )  # 5 days

    assert rule_codes(caught) == ["over_allowance"]
    assert "Only 2 days" in caught.value.violations[0].message


def test_unknown_leave_type_and_year_crossing_requests_are_refused(db: Session) -> None:
    alice = as_user(db, "alice.park")

    with pytest.raises(LeaveRuleError) as unknown:
        alice.request("sabbatical", date(2026, 11, 2), date(2026, 11, 3))
    with pytest.raises(LeaveRuleError) as crossing:
        alice.request("pto", date(2026, 12, 30), date(2027, 1, 4))

    assert rule_codes(unknown) == ["unknown_leave_type"]
    assert rule_codes(crossing) == ["spans_years"]
