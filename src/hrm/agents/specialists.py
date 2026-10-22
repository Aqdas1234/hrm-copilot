"""The specialist agents of router mode, declared as data.

A specialist is the shared agent loop (graph.agent_loop) with a narrower tool set and a
focus line added to the system prompt. Fewer tools means a shorter prompt (cheaper) and
fewer wrong choices for a small model; that is the claim B7 measures against the single
agent.

Role filtering still happens first (tools.build_tools): a specialist only gets the tools
the user already has, so "people" has no approval tools for an employee.

Every specialist also gets search_hr_policies: "can I take 3 days next week?" is a leave
action *and* a policy question.
"""

from dataclasses import dataclass
from typing import Literal

SpecialistName = Literal["policy", "leave", "pay", "people", "general"]

POLICY_TOOL = "search_hr_policies"


@dataclass(frozen=True)
class Specialist:
    name: SpecialistName
    description: str  # what the router reads to pick one
    tools: frozenset[str] | None  # None: all of the user's tools
    focus: str  # added to the system prompt


SPECIALISTS: tuple[Specialist, ...] = (
    Specialist(
        name="policy",
        description=(
            "Questions about company policy in general: PTO rules, parental leave, benefits, "
            "expense and stipend rules, remote work, conduct. No personal data needed."
        ),
        tools=frozenset({POLICY_TOOL}),
        focus="You answer company policy questions from the policy documents.",
    ),
    Specialist(
        name="leave",
        description=(
            "The user's own time off: leave balance, their leave requests, public holidays, "
            "checking, requesting or cancelling leave."
        ),
        tools=frozenset(
            {
                POLICY_TOOL,
                "get_my_leave_balance",
                "list_my_leave_requests",
                "get_public_holidays",
                "check_leave_request",
                "request_leave",
                "cancel_my_leave_request",
            }
        ),
        focus="You help the user with their own time off.",
    ),
    Specialist(
        name="pay",
        description=(
            "The user's own pay and expenses: salary, payslips, listing or submitting "
            "expense claims."
        ),
        tools=frozenset(
            {
                POLICY_TOOL,
                "get_my_salary",
                "get_my_payslip",
                "list_my_expense_claims",
                "submit_expense_claim",
            }
        ),
        focus="You help the user with their own pay, payslips and expense claims.",
    ),
    Specialist(
        name="people",
        description=(
            "The user's HR profile, their team, and HR tickets (asking HR for help); for "
            "managers: approving or rejecting their reports' leave and expense claims; for HR "
            "admins: the ticket queue and workforce statistics."
        ),
        tools=frozenset(
            {
                POLICY_TOOL,
                "get_my_profile",
                "list_my_team",
                "list_pending_approvals",
                "decide_leave_request",
                "decide_expense_claim",
                "open_hr_ticket",
                "list_my_tickets",
                "list_open_tickets",
                "update_hr_ticket",
                "query_hr_analytics",
            }
        ),
        focus="You help with the user's profile, their team and their approvals.",
    ),
    Specialist(
        name="general",
        description="Requests that span several of the areas above, or fit none of them.",
        tools=None,
        focus="You can use all of the user's tools.",
    ),
)

FALLBACK: SpecialistName = "general"
