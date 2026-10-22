"""Leave rules as plain, deterministic Python (no database, no network, no LLM).

Everything the rules need is passed in, so each rule is easy to test and to review
against the time-off policy. The policy's notice rule is a guideline ("as a rule of thumb
... if practicable"), so short notice is a warning, not a refusal.

Violations block a request; warnings are shown to the employee and the approver.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from hrm.core.errors import HRMError

NOTICE_BUSINESS_DAYS = 5  # time-off policy: inform the manager 5 business days ahead
COVERAGE_BUSINESS_DAYS = 5  # policy: "more than 5 business days" -> set up Assign Roles


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


class LeaveRuleError(HRMError):
    """The request breaks one or more rules; all of them are listed."""

    def __init__(self, violations: list[Violation]) -> None:
        super().__init__("; ".join(v.message for v in violations))
        self.violations = violations


@dataclass(frozen=True)
class Period:
    start: date
    end: date


@dataclass(frozen=True)
class LeaveRequestDraft:
    leave_code: str  # "pto", "parental", "unpaid"
    start: date
    end: date


@dataclass(frozen=True)
class LeaveContext:
    """What the rules need to know about the employee, prepared by the caller."""

    today: date
    holidays: frozenset[date]  # the employee's country, covering the requested dates
    existing: tuple[Period, ...]  # pending and approved requests
    allowed_days: Decimal | None  # yearly allowance for this leave type; None = no limit
    used_days: Decimal
    manager_id: int | None


@dataclass(frozen=True)
class LeavePlan:
    working_days: tuple[date, ...]
    day_count: Decimal
    approver_id: int | None
    warnings: list[str] = field(default_factory=list)


def working_days(start: date, end: date, holidays: Iterable[date] = ()) -> list[date]:
    """Weekdays from start to end (inclusive) that are not public holidays."""
    off = set(holidays)
    days, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in off:
            days.append(day)
        day += timedelta(days=1)
    return days


def business_days_between(after: date, before: date, holidays: Iterable[date] = ()) -> int:
    """Working days strictly after `after` and strictly before `before`."""
    return len(working_days(after + timedelta(days=1), before - timedelta(days=1), holidays))


def plan_leave(draft: LeaveRequestDraft, ctx: LeaveContext) -> LeavePlan:
    """Check a leave request against the rules. Raises LeaveRuleError with every violation."""
    violations: list[Violation] = []
    if draft.end < draft.start:
        violations.append(Violation("dates_reversed", "The end date is before the start date."))
    if draft.start < ctx.today:
        violations.append(Violation("in_the_past", "Leave cannot start in the past."))

    days = working_days(draft.start, draft.end, ctx.holidays) if draft.end >= draft.start else []
    if draft.end >= draft.start and not days:
        violations.append(
            Violation(
                "no_working_days",
                "The dates contain no working days (only weekends or holidays).",
            )
        )

    overlapping = [p for p in ctx.existing if p.start <= draft.end and draft.start <= p.end]
    if overlapping:
        first = overlapping[0]
        violations.append(
            Violation(
                "overlap",
                f"Overlaps an existing request ({first.start} to {first.end}).",
            )
        )

    count = Decimal(len(days))
    if ctx.allowed_days is not None and ctx.used_days + count > ctx.allowed_days:
        left = max(ctx.allowed_days - ctx.used_days, Decimal(0))
        violations.append(
            Violation(
                "over_allowance",
                f"Only {left} days of {draft.leave_code} leave remain; this request needs {count}.",
            )
        )

    if violations:
        raise LeaveRuleError(violations)

    warnings: list[str] = []
    if draft.leave_code == "pto":
        notice = business_days_between(ctx.today, draft.start, ctx.holidays)
        if notice < NOTICE_BUSINESS_DAYS:
            warnings.append(
                f"Short notice: {notice} business days before the leave; the policy asks for "
                f"{NOTICE_BUSINESS_DAYS} where practicable. Please tell your manager now."
            )
        if len(days) > COVERAGE_BUSINESS_DAYS:
            warnings.append(
                "More than 5 business days: the policy encourages setting up coverage with "
                "the Assign Roles feature in PTO by Deel."
            )
    if ctx.manager_id is None:
        warnings.append("No manager on file: the People Team will route this request.")

    return LeavePlan(tuple(days), count, ctx.manager_id, warnings)
