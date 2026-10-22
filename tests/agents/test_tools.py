"""Agent tools against the seeded database, with fakes for models and external services.

Today: Thursday 2026-10-22. Alice (US) and Hira (PK) report to Bob; Carol is HR admin.
"""

import json
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from hrm.agents.tools import is_write
from hrm.db.models import RequestStatus
from tests.agents.helpers import tools_for

SELF_TOOLS = {
    "search_hr_policies",
    "get_my_profile",
    "get_my_salary",
    "get_my_leave_balance",
    "list_my_leave_requests",
    "get_public_holidays",
    "check_leave_request",
    "request_leave",
    "cancel_my_leave_request",
    "list_my_expense_claims",
    "submit_expense_claim",
    "get_my_payslip",
    "open_hr_ticket",
    "list_my_tickets",
}
TEAM_TOOLS = {
    "list_my_team",
    "list_pending_approvals",
    "decide_leave_request",
    "decide_expense_claim",
}
HR_ADMIN_TOOLS = {"list_open_tickets", "update_hr_ticket"}  # + analytics when configured
WRITE_TOOLS = {
    "open_hr_ticket",
    "update_hr_ticket",
    "request_leave",
    "cancel_my_leave_request",
    "submit_expense_claim",
    "decide_leave_request",
    "decide_expense_claim",
}


def call(tools: dict[str, BaseTool], name: str, **args: Any) -> Any:
    out = tools[name].invoke(args)
    return out if out.startswith("ERROR:") else json.loads(out)


# --- structure: identity, roles, write markers ----------------------------------------------


def test_employees_get_self_service_tools_only(db: Session) -> None:
    assert set(tools_for(db, "alice.park")) == SELF_TOOLS


def test_managers_also_get_team_tools(db: Session) -> None:
    assert set(tools_for(db, "bob.chen")) == SELF_TOOLS | TEAM_TOOLS


def test_hr_admins_also_get_the_ticket_queue(db: Session) -> None:
    assert set(tools_for(db, "carol.reyes")) == SELF_TOOLS | TEAM_TOOLS | HR_ADMIN_TOOLS


def test_no_self_service_tool_accepts_an_identity_argument(db: Session) -> None:
    identity_words = ("user", "employee", "email", "person", "who")
    for name, t in tools_for(db, "carol.reyes").items():
        if name in SELF_TOOLS:
            assert not [a for a in t.args if any(w in a.lower() for w in identity_words)], name


def test_exactly_the_data_changing_tools_are_marked_as_writes(db: Session) -> None:
    # The HR admin has every tool.
    marked = {name for name, t in tools_for(db, "carol.reyes").items() if is_write(t)}

    assert marked == WRITE_TOOLS


def test_every_tool_has_a_description_for_the_model(db: Session) -> None:
    for name, t in tools_for(db, "bob.chen").items():
        assert len(t.description) > 30, name


# --- behaviour --------------------------------------------------------------------------------


def test_profile_and_salary_are_always_the_logged_in_users(db: Session) -> None:
    alice = tools_for(db, "alice.park")

    assert call(alice, "get_my_profile")["name"] == "Alice Park"
    assert call(alice, "get_my_salary")["currency"] == "USD"


def test_leave_balance_explains_flexible_pto(db: Session) -> None:
    balance = call(tools_for(db, "alice.park"), "get_my_leave_balance")

    assert balance["allowed_days"] is None and "no set limit" in balance["note"]


def test_holidays_are_for_the_users_own_country(db: Session) -> None:
    holidays = call(tools_for(db, "hira.khan"), "get_public_holidays", year=2026)

    assert holidays["country"] == "PK"
    assert holidays["holidays"][0]["name"] == "Iqbal Day"


def test_check_leave_does_not_save_but_request_leave_does(db: Session) -> None:
    alice = tools_for(db, "alice.park")
    before = len(call(alice, "list_my_leave_requests"))

    preview = call(
        alice,
        "check_leave_request",
        leave_type="pto",
        start_date="2026-11-02",
        end_date="2026-11-04",
    )
    assert preview["working_days"] == 3
    assert len(call(alice, "list_my_leave_requests")) == before

    filed = call(
        alice, "request_leave", leave_type="pto", start_date="2026-11-02", end_date="2026-11-04"
    )
    assert filed["status"] == "pending"
    assert len(call(alice, "list_my_leave_requests")) == before + 1


def test_submit_expense_converts_the_amount(db: Session) -> None:
    claim = call(
        tools_for(db, "alice.park"),
        "submit_expense_claim",
        description="Keyboard",
        amount="80.50",
        currency="eur",
        purchase_date="2026-10-20",
    )

    assert claim["amount_cents"] == 8_050 and claim["amount_usd_cents"] == 9_036


def test_payslip_comes_from_payroll_for_the_users_own_employee_code(db: Session) -> None:
    slip = call(tools_for(db, "alice.park"), "get_my_payslip", year=2026, month=9)

    assert slip["employee_code"] == "E0003" and slip["gross_cents"] == 1_250_000


def test_manager_sees_and_decides_pending_approvals(db: Session) -> None:
    bob = tools_for(db, "bob.chen")
    pending = call(bob, "list_pending_approvals")
    request_id = pending["leave_requests"][0]["id"]

    decided = call(bob, "decide_leave_request", request_id=request_id, decision="approve")

    assert {"leave_requests", "expense_claims"} <= pending.keys()
    assert decided["status"] == RequestStatus.APPROVED.value


def test_policy_search_is_available_to_everyone(db: Session) -> None:
    out = tools_for(db, "alice.park")["search_hr_policies"].invoke({"query": "notice before PTO"})

    assert "5 business days" in out


# --- errors become messages the agent can relay ------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        (
            "check_leave_request",
            {"leave_type": "pto", "start_date": "next friday", "end_date": "2026-11-04"},
            "is not a date",
        ),
        (
            "check_leave_request",
            {"leave_type": "pto", "start_date": "2026-10-24", "end_date": "2026-10-25"},
            "no working days",
        ),
        (
            "submit_expense_claim",
            {
                "description": "Taxi",
                "amount": "-5",
                "currency": "USD",
                "purchase_date": "2026-10-20",
            },
            "above 0",
        ),
        (
            "submit_expense_claim",
            {
                "description": "Taxi",
                "amount": "lots",
                "currency": "USD",
                "purchase_date": "2026-10-20",
            },
            "not an amount",
        ),
        ("cancel_my_leave_request", {"request_id": 999_999}, "not found or not accessible"),
    ],
)
def test_expected_failures_are_returned_as_error_messages(
    db: Session, tool: str, args: dict[str, Any], expected: str
) -> None:
    out = tools_for(db, "alice.park")[tool].invoke(args)

    assert out.startswith("ERROR:") and expected in out


def test_rejecting_requires_a_reason(db: Session) -> None:
    bob = tools_for(db, "bob.chen")
    request_id = call(bob, "list_pending_approvals")["leave_requests"][0]["id"]

    out = bob["decide_leave_request"].invoke({"request_id": request_id, "decision": "reject"})

    assert out.startswith("ERROR:") and "reason" in out


def test_a_manager_cannot_decide_requests_outside_their_queue(db: Session) -> None:
    carol_request = call(
        tools_for(db, "carol.reyes"),
        "request_leave",
        leave_type="pto",
        start_date="2026-11-02",
        end_date="2026-11-03",
    )["id"]

    out = tools_for(db, "bob.chen")["decide_leave_request"].invoke(
        {"request_id": carol_request, "decision": "approve"}
    )

    assert out == "ERROR: Leave request not found or not accessible."


# --- HR tickets ----------------------------------------------------------------------------


def test_an_employee_opens_and_lists_their_own_tickets(db: Session) -> None:
    alice = tools_for(db, "alice.park")
    opened = call(alice, "open_hr_ticket", category="payroll", subject="Tax code", description="x")

    assert opened["status"] == "open"
    assert [t["id"] for t in call(alice, "list_my_tickets")] == [opened["id"]]
    assert call(tools_for(db, "hira.khan"), "list_my_tickets") == []


def test_hr_works_the_queue_and_closing_needs_a_resolution(db: Session) -> None:
    ticket = call(
        tools_for(db, "alice.park"),
        "open_hr_ticket",
        category="access",
        subject="s",
        description="b",
    )
    carol = tools_for(db, "carol.reyes")

    assert [t["id"] for t in call(carol, "list_open_tickets")] == [ticket["id"]]
    error = call(carol, "update_hr_ticket", ticket_id=ticket["id"], status="closed")
    assert error.startswith("ERROR:") and "resolution" in error
    closed = call(
        carol, "update_hr_ticket", ticket_id=ticket["id"], status="closed", resolution="Done."
    )
    assert closed["status"] == "closed"
    assert call(carol, "list_open_tickets") == []
