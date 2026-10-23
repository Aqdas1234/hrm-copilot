"""Expense rules from the "Spending company money" policy (pure, deterministic).

- "Please file an expense report within 60 days of an eligible purchase": a deadline,
  so later claims are refused.
- "If you spend more than $1,000 on any given physical item, it will be considered
  property of the company": we cannot know if the purchase is a physical item, so claims
  over $1,000 are flagged for the approver rather than refused.
- The policy brings in People Ops above its spending limits; here every claim over $1,000
  needs HR's approval after the manager's.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

from hrm.workflows.leave_rules import LeaveRuleError, Violation

FILING_DEADLINE_DAYS = 60
COMPANY_PROPERTY_USD_CENTS = 1_000 * 100
HR_SIGN_OFF_USD_CENTS = 1_000 * 100

ExpenseRuleError = LeaveRuleError  # same shape: a list of violations


@dataclass(frozen=True)
class ExpensePlan:
    is_company_property: bool
    warnings: list[str] = field(default_factory=list)


def check_expense(
    *,
    description: str,
    amount_cents: int,
    purchase_date: date,
    usd_cents: int,
    today: date,
    has_manager: bool,
) -> ExpensePlan:
    violations: list[Violation] = []
    if not description.strip():
        violations.append(Violation("no_description", "Please describe what was bought."))
    if amount_cents <= 0:
        violations.append(Violation("amount_not_positive", "The amount must be greater than zero."))
    if purchase_date > today:
        violations.append(Violation("future_purchase", "The purchase date is in the future."))
    elif purchase_date < today - timedelta(days=FILING_DEADLINE_DAYS):
        violations.append(
            Violation(
                "too_late",
                f"Expenses must be filed within {FILING_DEADLINE_DAYS} days of the purchase "
                f"(this one is {(today - purchase_date).days} days old). "
                "Please contact People Ops.",
            )
        )
    if violations:
        raise ExpenseRuleError(violations)

    warnings: list[str] = []
    is_property = usd_cents > COMPANY_PROPERTY_USD_CENTS
    if is_property:
        warnings.append(
            "Over $1,000: if this is a physical item, it becomes company property (policy)."
        )
    if needs_hr_sign_off(usd_cents):
        warnings.append("Over $1,000: after your manager approves, HR gives the final approval.")
    if not has_manager:
        warnings.append("No manager on file: the People Team will route this claim.")
    return ExpensePlan(is_property, warnings)


def needs_hr_sign_off(usd_cents: int) -> bool:
    return usd_cents > HR_SIGN_OFF_USD_CENTS
