"""Expense claims: submit, approve, reject, cancel.

Same permission rules as leave: the employee submits for themselves; only the assigned
approver (the manager) or an HR admin decides; nobody decides on their own claim; only
open claims can be decided or cancelled. Amounts are converted to USD at the rate of
the purchase date and stored with the claim. Every action is audited.

Claims over $1,000 need two approvals: the manager's moves the claim to pending_hr, then
any HR admin gives the final one. An HR admin approving at the first step counts as both
(they are HR). Either step can reject; the employee can cancel until the final decision.
"""

from datetime import UTC, date, datetime
from typing import NoReturn, Protocol

from pydantic import BaseModel
from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.models import ExpenseClaim, RequestStatus
from hrm.integrations.currency import Conversion
from hrm.workflows.expense_rules import ExpenseRuleError, check_expense, needs_hr_sign_off
from hrm.workflows.leave_rules import Violation

OPEN = (RequestStatus.PENDING, RequestStatus.PENDING_HR)


class RateLookup(Protocol):
    """CurrencyService, or a fake in tests."""

    def to_usd(self, amount_cents: int, currency: str, on: date) -> Conversion: ...


class ExpenseView(BaseModel):
    id: int
    employee_id: int
    description: str
    amount_cents: int
    currency: str
    amount_usd_cents: int
    purchase_date: date
    is_company_property: bool
    status: RequestStatus

    @classmethod
    def of(cls, c: ExpenseClaim) -> "ExpenseView":
        return cls(
            id=c.id,
            employee_id=c.employee_id,
            description=c.description,
            amount_cents=c.amount_cents,
            currency=c.currency,
            amount_usd_cents=c.amount_usd_cents,
            purchase_date=c.purchase_date,
            is_company_property=c.is_company_property,
            status=c.status,
        )


def pending_expense_approvals(session: Session, ctx: UserContext) -> list[ExpenseView]:
    """Claims waiting for this user's decision: as their manager, and for HR admins also
    every claim waiting for HR's approval (except their own)."""
    claims = repo.pending_expenses_for_approver(session, ctx.employee_id)
    if ctx.is_hr_admin:
        claims += [
            c for c in repo.expenses_waiting_for_hr(session) if c.employee_id != ctx.employee_id
        ]
    return [ExpenseView.of(c) for c in claims]


class ExpenseOutcome(ExpenseView):
    warnings: list[str]


class ExpenseService:
    def __init__(self, session: Session, ctx: UserContext, rates: RateLookup, today: date) -> None:
        self._session = session
        self._ctx = ctx
        self._rates = rates
        self._today = today

    def submit(
        self, description: str, amount_cents: int, currency: str, purchase_date: date
    ) -> ExpenseOutcome:
        employee = repo.get_employee(self._session, self._ctx.employee_id)
        if employee is None:
            raise ExpenseRuleError(
                [Violation("no_employee", "Your employee record was not found.")]
            )
        currency = currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ExpenseRuleError(
                [Violation("bad_currency", "Use a 3-letter currency code, e.g. USD.")]
            )
        # Rules that need no conversion first, so a bad claim never triggers a rate lookup.
        check_expense(
            description=description,
            amount_cents=amount_cents,
            purchase_date=purchase_date,
            usd_cents=0,
            today=self._today,
            has_manager=True,
        )
        conversion = self._rates.to_usd(amount_cents, currency, purchase_date)
        plan = check_expense(
            description=description,
            amount_cents=amount_cents,
            purchase_date=purchase_date,
            usd_cents=conversion.usd_cents,
            today=self._today,
            has_manager=employee.manager_id is not None,
        )
        claim = ExpenseClaim(
            employee_id=employee.id,
            description=description.strip(),
            amount_cents=amount_cents,
            currency=conversion.currency,
            amount_usd_cents=conversion.usd_cents,
            purchase_date=purchase_date,
            is_company_property=plan.is_company_property,
            status=RequestStatus.PENDING,
            approver_id=employee.manager_id,
        )
        self._session.add(claim)
        self._session.flush()
        self._audit(
            "expense.submit",
            claim,
            {
                "usd_cents": conversion.usd_cents,
                "rate": str(conversion.usd_per_unit),
                "rate_published_on": conversion.published_on.isoformat(),
                "warnings": plan.warnings,
            },
        )
        return ExpenseOutcome(**ExpenseView.of(claim).model_dump(), warnings=plan.warnings)

    def my_claims(self) -> list[ExpenseView]:
        return [
            ExpenseView.of(c) for c in repo.expense_claims_for(self._session, self._ctx.employee_id)
        ]

    def pending_approvals(self) -> list[ExpenseView]:
        return pending_expense_approvals(self._session, self._ctx)

    def approve(self, claim_id: int, note: str | None = None) -> ExpenseView:
        claim = self._decidable(claim_id)
        manager_step = claim.status is RequestStatus.PENDING and not self._ctx.is_hr_admin
        if manager_step and needs_hr_sign_off(claim.amount_usd_cents):
            return self._decide(claim, RequestStatus.PENDING_HR, "expense.approve_manager", note)
        return self._decide(claim, RequestStatus.APPROVED, "expense.approve", note)

    def reject(self, claim_id: int, note: str) -> ExpenseView:
        return self._decide(
            self._decidable(claim_id), RequestStatus.REJECTED, "expense.reject", note
        )

    def cancel(self, claim_id: int) -> ExpenseView:
        claim = repo.get_expense_claim(self._session, claim_id)
        if claim is None or claim.employee_id != self._ctx.employee_id:
            self._deny("expense.cancel", claim_id)
        self._require_pending(claim)
        return self._decide(claim, RequestStatus.CANCELLED, "expense.cancel", None)

    # --- helpers --------------------------------------------------------------------------

    def _decidable(self, claim_id: int) -> ExpenseClaim:
        claim = repo.get_expense_claim(self._session, claim_id)
        if claim is None or claim.employee_id == self._ctx.employee_id:
            self._deny("expense.decide", claim_id)
        is_approver = claim.approver_id == self._ctx.employee_id
        if claim.status is RequestStatus.PENDING_HR and not self._ctx.is_hr_admin:
            if is_approver:
                raise ExpenseRuleError(
                    [Violation("waiting_for_hr", "You approved this claim; HR decides it now.")]
                )
            self._deny("expense.decide", claim_id)
        if not (is_approver or self._ctx.is_hr_admin):
            self._deny("expense.decide", claim_id)
        self._require_pending(claim)
        return claim

    @staticmethod
    def _require_pending(claim: ExpenseClaim) -> None:
        if claim.status not in OPEN:
            raise ExpenseRuleError(
                [Violation("not_pending", f"This claim is already {claim.status.value}.")]
            )

    def _decide(
        self, claim: ExpenseClaim, status: RequestStatus, action: str, note: str | None
    ) -> ExpenseView:
        claim.status = status
        if status not in OPEN:  # the manager's step is not the decision
            claim.decided_at = datetime.now(UTC)
        self._session.flush()
        self._audit(action, claim, {"note": note} if note else {})
        return ExpenseView.of(claim)

    def _audit(self, action: str, claim: ExpenseClaim, details: dict[str, object]) -> None:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action=action,
            entity="expense_claim",
            entity_id=claim.id,
            details=details,
        )

    def _deny(self, action: str, claim_id: int) -> NoReturn:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action="access.denied",
            entity="expense_claim",
            entity_id=claim_id,
            details={"attempted": action},
        )
        raise AccessDeniedError("Expense claim not found or not accessible.")
