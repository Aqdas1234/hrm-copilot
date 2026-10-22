"""Leave rules on a real calendar. Today: Thursday 2026-10-22."""

from datetime import date
from decimal import Decimal

import pytest

from hrm.workflows.leave_rules import (
    LeaveContext,
    LeaveRequestDraft,
    LeaveRuleError,
    Period,
    business_days_between,
    plan_leave,
    working_days,
)

TODAY = date(2026, 10, 22)  # Thursday
IQBAL_DAY = date(2026, 11, 9)  # Monday, a public holiday in Pakistan
PK_HOLIDAYS = frozenset({IQBAL_DAY})


def ctx(**overrides: object) -> LeaveContext:
    values: dict[str, object] = {
        "today": TODAY,
        "holidays": frozenset(),
        "existing": (),
        "allowed_days": None,
        "used_days": Decimal(0),
        "manager_id": 42,
    }
    values.update(overrides)
    return LeaveContext(**values)  # type: ignore[arg-type]


def pto(start: date, end: date) -> LeaveRequestDraft:
    return LeaveRequestDraft("pto", start, end)


def violation_codes(draft: LeaveRequestDraft, context: LeaveContext) -> list[str]:
    with pytest.raises(LeaveRuleError) as caught:
        plan_leave(draft, context)
    return [v.code for v in caught.value.violations]


# --- counting days ------------------------------------------------------------------------


def test_working_days_skip_weekends_and_holidays() -> None:
    # Fri 6 Nov .. Tue 10 Nov: Fri, (Sat, Sun), Iqbal Day Mon, Tue
    days = working_days(date(2026, 11, 6), date(2026, 11, 10), PK_HOLIDAYS)

    assert days == [date(2026, 11, 6), date(2026, 11, 10)]


def test_business_days_between_excludes_both_ends() -> None:
    # after Thu 22 Oct, before Thu 29 Oct: Fri 23, Mon 26, Tue 27, Wed 28
    assert business_days_between(TODAY, date(2026, 10, 29)) == 4


# --- valid requests -------------------------------------------------------------------------


def test_simple_request_with_good_notice_has_no_warnings() -> None:
    plan = plan_leave(pto(date(2026, 11, 2), date(2026, 11, 4)), ctx())  # Mon-Wed, 6 days' notice

    assert plan.day_count == 3
    assert plan.approver_id == 42
    assert plan.warnings == []


def test_holiday_inside_the_leave_is_not_counted() -> None:
    plan = plan_leave(pto(date(2026, 11, 9), date(2026, 11, 13)), ctx(holidays=PK_HOLIDAYS))

    assert plan.day_count == 4  # Mon is Iqbal Day
    assert IQBAL_DAY not in plan.working_days


def test_request_starting_on_a_weekend_counts_only_weekdays() -> None:
    plan = plan_leave(pto(date(2026, 10, 31), date(2026, 11, 3)), ctx())  # Sat .. Tue

    assert plan.day_count == 2


# --- warnings (allowed, but flagged) -----------------------------------------------------


def test_short_notice_is_a_warning_not_a_refusal() -> None:
    plan = plan_leave(pto(date(2026, 10, 26), date(2026, 10, 26)), ctx())  # next Monday

    assert plan.day_count == 1
    assert any("Short notice: 1 business days" in w for w in plan.warnings)


def test_holidays_do_not_count_as_notice() -> None:
    # after Thu 5 Nov, before Wed 11 Nov: Fri 6, (Mon 9 = Iqbal Day), Tue 10 -> 2 business days
    plan = plan_leave(
        pto(date(2026, 11, 11), date(2026, 11, 11)),
        ctx(today=date(2026, 11, 5), holidays=PK_HOLIDAYS),
    )

    assert any("Short notice: 2 business days" in w for w in plan.warnings)


def test_long_leave_suggests_coverage() -> None:
    plan = plan_leave(pto(date(2026, 11, 2), date(2026, 11, 13)), ctx())  # 10 working days

    assert any("Assign Roles" in w for w in plan.warnings)


def test_no_manager_routes_to_the_people_team() -> None:
    plan = plan_leave(pto(date(2026, 11, 2), date(2026, 11, 2)), ctx(manager_id=None))

    assert plan.approver_id is None
    assert any("People Team" in w for w in plan.warnings)


def test_notice_warnings_apply_to_pto_only() -> None:
    draft = LeaveRequestDraft("parental", date(2026, 10, 26), date(2026, 10, 30))

    plan = plan_leave(draft, ctx(allowed_days=Decimal(80)))

    assert plan.warnings == []


# --- violations (refused) -----------------------------------------------------------------


def test_reversed_dates() -> None:
    assert violation_codes(pto(date(2026, 11, 4), date(2026, 11, 2)), ctx()) == ["dates_reversed"]


def test_leave_in_the_past() -> None:
    assert "in_the_past" in violation_codes(pto(date(2026, 10, 20), date(2026, 10, 21)), ctx())


def test_only_weekend_or_holiday_dates() -> None:
    weekend = pto(date(2026, 10, 24), date(2026, 10, 25))
    holiday = pto(IQBAL_DAY, IQBAL_DAY)

    assert violation_codes(weekend, ctx()) == ["no_working_days"]
    assert violation_codes(holiday, ctx(holidays=PK_HOLIDAYS)) == ["no_working_days"]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2026, 11, 2), date(2026, 11, 4)),  # same days
        (date(2026, 10, 30), date(2026, 11, 2)),  # overlaps the first day
        (date(2026, 11, 4), date(2026, 11, 6)),  # overlaps the last day
        (date(2026, 11, 3), date(2026, 11, 3)),  # inside
    ],
)
def test_overlap_with_an_existing_request(start: date, end: date) -> None:
    existing = (Period(date(2026, 11, 2), date(2026, 11, 4)),)

    assert violation_codes(pto(start, end), ctx(existing=existing)) == ["overlap"]


def test_adjacent_requests_do_not_overlap() -> None:
    existing = (Period(date(2026, 11, 2), date(2026, 11, 4)),)

    assert (
        plan_leave(pto(date(2026, 11, 5), date(2026, 11, 6)), ctx(existing=existing)).day_count == 2
    )


def test_parental_leave_beyond_the_allowance() -> None:
    draft = LeaveRequestDraft("parental", date(2026, 11, 2), date(2026, 11, 13))  # 10 days

    codes = violation_codes(draft, ctx(allowed_days=Decimal(80), used_days=Decimal(75)))

    assert codes == ["over_allowance"]


def test_parental_leave_using_exactly_the_remaining_days_is_fine() -> None:
    draft = LeaveRequestDraft("parental", date(2026, 11, 2), date(2026, 11, 6))  # 5 days

    plan = plan_leave(draft, ctx(allowed_days=Decimal(80), used_days=Decimal(75)))

    assert plan.day_count == 5


def test_flexible_pto_has_no_allowance_limit() -> None:
    plan = plan_leave(pto(date(2026, 11, 2), date(2026, 12, 31)), ctx(used_days=Decimal(40)))

    assert plan.day_count > 40


def test_all_violations_are_reported_at_once() -> None:
    draft = pto(date(2026, 10, 10), date(2026, 10, 11))  # in the past AND a weekend

    assert sorted(violation_codes(draft, ctx())) == ["in_the_past", "no_working_days"]
