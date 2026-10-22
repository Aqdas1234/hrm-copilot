"""The analytics views, described for the SQL checker and the model's prompt.

The views are created by migration 0002. A test checks this description matches them
column for column, so the model is never told about a column that does not exist.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class View:
    name: str
    description: str
    columns: dict[str, str]  # column -> meaning


VIEWS: tuple[View, ...] = (
    View(
        "v_employees",
        "One row per employee (no contact data, no salary).",
        {
            "employee_id": "id",
            "employee_code": "e.g. E0003",
            "first_name": "",
            "last_name": "",
            "job_title": "",
            "department": "e.g. Engineering, People, Sales",
            "country": "ISO code, e.g. US, PK, GB, DE",
            "employment_type": "full_time | part_time | contractor",
            "start_date": "date",
            "is_active": "boolean",
            "manager_id": "employee_id of the manager, or NULL",
        },
    ),
    View(
        "v_leave_requests",
        "Leave requests (reasons are not included).",
        {
            "request_id": "id",
            "employee_id": "",
            "leave_type": "pto | parental | unpaid",
            "start_date": "date",
            "end_date": "date",
            "working_days": "number of working days requested",
            "status": "pending | approved | rejected | cancelled",
            "created_at": "timestamp",
        },
    ),
    View(
        "v_leave_balances",
        "Leave used per employee, leave type and year.",
        {
            "employee_id": "",
            "leave_type": "pto | parental | unpaid",
            "year": "",
            "allowed_days": "NULL means no set limit",
            "used_days": "",
        },
    ),
    View(
        "v_expense_claims",
        "Expense claims (descriptions are not included).",
        {
            "claim_id": "id",
            "employee_id": "",
            "currency": "currency paid in",
            "amount_usd_cents": "amount in US cents",
            "purchase_date": "date",
            "is_company_property": "boolean",
            "status": "pending | approved | rejected | cancelled",
        },
    ),
    View(
        "v_salary_bands",
        "Current yearly salary statistics per department, country and currency; only "
        "groups of 3 or more people. Individual salaries are not available.",
        {
            "department": "",
            "country": "",
            "currency": "",
            "employees": "people in the group (at least 3)",
            "min_salary_cents": "yearly, in cents of the currency",
            "max_salary_cents": "yearly, in cents of the currency",
            "avg_salary_cents": "yearly, in cents of the currency",
        },
    ),
)

VIEW_NAMES = frozenset(v.name for v in VIEWS)


def describe() -> str:
    """The schema as text, for the model's prompt."""
    lines = []
    for view in VIEWS:
        lines.append(f"{view.name}: {view.description}")
        for column, meaning in view.columns.items():
            lines.append(f"  - {column}{f': {meaning}' if meaning else ''}")
    return "\n".join(lines)
