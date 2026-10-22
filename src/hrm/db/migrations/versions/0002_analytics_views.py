"""Analytics views: the only data HR admins' SQL questions can read.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-22

Rules for what a view may expose (see hrm/analytics/schema.py):
- no free text people wrote (leave reasons, expense descriptions): it can carry injection;
- no contact data (email) and no credentials;
- salaries only as statistics over groups of at least 3 people, never per person.
The SQL is spelled out here, not imported, so this migration always builds the same views.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VIEWS = {
    "v_employees": """
        SELECT e.id AS employee_id, e.employee_code, e.first_name, e.last_name,
               e.job_title, d.name AS department, e.country, e.employment_type,
               e.start_date, e.is_active, e.manager_id
        FROM employees e JOIN departments d ON d.id = e.department_id
    """,
    "v_leave_requests": """
        SELECT r.id AS request_id, r.employee_id, t.code AS leave_type, r.start_date,
               r.end_date, r.working_days, r.status, r.created_at
        FROM leave_requests r JOIN leave_types t ON t.id = r.leave_type_id
    """,
    "v_leave_balances": """
        SELECT b.employee_id, t.code AS leave_type, b.year, b.allowed_days, b.used_days
        FROM leave_balances b JOIN leave_types t ON t.id = b.leave_type_id
    """,
    "v_expense_claims": """
        SELECT c.id AS claim_id, c.employee_id, c.currency, c.amount_usd_cents,
               c.purchase_date, c.is_company_property, c.status
        FROM expense_claims c
    """,
    # Current salary per employee, aggregated; groups smaller than 3 are left out so no
    # single person's salary can be read or derived.
    "v_salary_bands": """
        SELECT d.name AS department, e.country, c.currency,
               COUNT(*) AS employees,
               MIN(c.base_salary_cents) AS min_salary_cents,
               MAX(c.base_salary_cents) AS max_salary_cents,
               AVG(c.base_salary_cents) AS avg_salary_cents
        FROM compensation c
        JOIN employees e ON e.id = c.employee_id
        JOIN departments d ON d.id = e.department_id
        WHERE e.is_active
          AND c.effective_from = (
              SELECT MAX(c2.effective_from) FROM compensation c2
              WHERE c2.employee_id = c.employee_id AND c2.effective_from <= CURRENT_DATE
          )
        GROUP BY d.name, e.country, c.currency
        HAVING COUNT(*) >= 3
    """,
}


def upgrade() -> None:
    for name, sql in VIEWS.items():
        op.execute(f"CREATE VIEW {name} AS {sql}")


def downgrade() -> None:
    for name in reversed(VIEWS):
        op.execute(f"DROP VIEW {name}")
