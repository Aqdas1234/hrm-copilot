"""Expense rules and actions on the seeded database. Today: Thursday 2026-10-22.

Seed facts: Hira (reports to Bob) has a pending PKR monitor claim; Bob's own pending
laptop claim has no approver (Bob has no manager); Carol is HR admin.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.models import AuditLog, RateSource, RequestStatus
from hrm.db.seed import seed
from hrm.integrations.currency import Conversion
from hrm.workflows.expense_rules import ExpensePlan, ExpenseRuleError, check_expense
from hrm.workflows.expenses import ExpenseService

TODAY = date(2026, 10, 22)
RATES = {"USD": Decimal(1), "EUR": Decimal("1.1225"), "PKR": Decimal("0.0036")}


class FakeRates:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, date]] = []

    def to_usd(self, amount_cents: int, currency: str, on: date) -> Conversion:
        self.calls.append((amount_cents, currency, on))
        rate = RATES[currency]
        return Conversion(
            amount_cents,
            currency,
            int(amount_cents * rate),
            rate,
            on,
            None if currency == "USD" else RateSource.API,
        )


@pytest.fixture
def db(session: Session) -> Session:
    seed(session, today=TODAY, demo_password="x")
    return session


def as_user(db: Session, name: str, rates: FakeRates | None = None) -> ExpenseService:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    return ExpenseService(db, UserContext.from_user(user), rates or FakeRates(), TODAY)


def pending_claim(db: Session, name: str) -> int:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    [claim] = [
        c
        for c in repo.expense_claims_for(db, user.employee_id)
        if c.status is RequestStatus.PENDING
    ]
    return claim.id


def codes(error: pytest.ExceptionInfo[ExpenseRuleError]) -> list[str]:
    return [v.code for v in error.value.violations]


# --- the pure rules -------------------------------------------------------------------------


def rules(**overrides: object) -> ExpensePlan:
    values: dict[str, object] = {
        "description": "Monitor",
        "amount_cents": 10_000,
        "purchase_date": TODAY - timedelta(days=5),
        "usd_cents": 10_000,
        "today": TODAY,
        "has_manager": True,
    }
    values.update(overrides)
    return check_expense(**values)  # type: ignore[arg-type]


def test_filing_deadline_is_60_days() -> None:
    rules(purchase_date=TODAY - timedelta(days=60))  # day 60: still fine

    with pytest.raises(ExpenseRuleError) as caught:
        rules(purchase_date=TODAY - timedelta(days=61))

    assert codes(caught) == ["too_late"]
    assert "61 days old" in str(caught.value)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"description": "  "}, "no_description"),
        ({"amount_cents": 0}, "amount_not_positive"),
        ({"purchase_date": TODAY + timedelta(days=1)}, "future_purchase"),
    ],
)
def test_invalid_claims(overrides: dict[str, object], code: str) -> None:
    with pytest.raises(ExpenseRuleError) as caught:
        rules(**overrides)

    assert codes(caught) == [code]


def test_over_1000_usd_is_flagged_not_refused() -> None:
    at_limit = rules(usd_cents=100_000)
    over = rules(usd_cents=100_001)

    assert not at_limit.is_company_property
    assert over.is_company_property
    assert any("company property" in w for w in over.warnings)


# --- actions --------------------------------------------------------------------------------


def test_submit_converts_at_the_purchase_date_and_routes_to_the_manager(db: Session) -> None:
    rates = FakeRates()
    purchase = TODAY - timedelta(days=10)

    outcome = as_user(db, "alice.park", rates).submit("Conference ticket", 25_000, "eur", purchase)

    assert rates.calls == [(25_000, "EUR", purchase)]  # normalized before the lookup
    assert outcome.amount_usd_cents == 28_062 and outcome.currency == "EUR"
    assert outcome.status is RequestStatus.PENDING and outcome.warnings == []
    claim = repo.get_expense_claim(db, outcome.id)
    alice = repo.get_user_by_email(db, "alice.park@example.com")
    assert (
        claim is not None and alice is not None and claim.approver_id == alice.employee.manager_id
    )
    entry = db.scalar(select(AuditLog).where(AuditLog.action == "expense.submit"))
    assert entry is not None and entry.details["rate"] == "1.1225"


def test_expensive_item_is_flagged_for_the_approver(db: Session) -> None:
    outcome = as_user(db, "alice.park").submit("Laptop", 145_000, "USD", TODAY)

    assert outcome.is_company_property
    assert any("company property" in w for w in outcome.warnings)


def test_too_old_claim_is_refused_before_any_rate_lookup(db: Session) -> None:
    rates = FakeRates()

    with pytest.raises(ExpenseRuleError) as caught:
        as_user(db, "alice.park", rates).submit(
            "Old taxi", 4_000, "EUR", TODAY - timedelta(days=90)
        )

    assert codes(caught) == ["too_late"]
    assert rates.calls == []


def test_manager_approves_and_rejects_team_claims(db: Session) -> None:
    approved = as_user(db, "bob.chen").approve(pending_claim(db, "hira.khan"), note="OK")
    new = as_user(db, "hira.khan").submit("Desk lamp", 800_000, "PKR", TODAY)
    rejected = as_user(db, "bob.chen").reject(new.id, note="Not a work item")

    assert approved.status is RequestStatus.APPROVED
    assert rejected.status is RequestStatus.REJECTED


def test_no_one_approves_their_own_claim_and_colleagues_cannot_decide(db: Session) -> None:
    with pytest.raises(AccessDeniedError):
        as_user(db, "bob.chen").approve(pending_claim(db, "bob.chen"))  # own laptop
    with pytest.raises(AccessDeniedError):
        as_user(db, "alice.park").approve(pending_claim(db, "hira.khan"))  # a colleague


def test_claim_without_approver_is_decided_by_hr(db: Session) -> None:
    approved = as_user(db, "carol.reyes").approve(pending_claim(db, "bob.chen"))

    assert approved.status is RequestStatus.APPROVED


def test_cancel_own_pending_claim_only(db: Session) -> None:
    hira_claim = pending_claim(db, "hira.khan")

    with pytest.raises(AccessDeniedError):
        as_user(db, "alice.park").cancel(hira_claim)
    assert as_user(db, "hira.khan").cancel(hira_claim).status is RequestStatus.CANCELLED


def test_decided_claims_cannot_be_decided_again(db: Session) -> None:
    bob = as_user(db, "bob.chen")
    claim = pending_claim(db, "hira.khan")
    bob.approve(claim)

    with pytest.raises(ExpenseRuleError) as caught:
        bob.reject(claim, note="second thoughts")

    assert codes(caught) == ["not_pending"]


@pytest.mark.parametrize("currency", ["EURO", "E1R", ""])
def test_invalid_currency_codes_are_refused(db: Session, currency: str) -> None:
    with pytest.raises(ExpenseRuleError) as caught:
        as_user(db, "alice.park").submit("Taxi", 2_000, currency, TODAY)

    assert codes(caught) == ["bad_currency"]


def test_my_claims_lists_only_my_own(db: Session) -> None:
    claims = as_user(db, "alice.park").my_claims()

    assert [c.description for c in claims] == ["Laptop stand"]
