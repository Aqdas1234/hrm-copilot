"""Expense claims over $1,000 need HR's approval after the manager's: status pending_hr

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-23 15:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BEFORE = ("pending", "approved", "rejected", "cancelled")
AFTER = ("pending", "pending_hr", "approved", "rejected", "cancelled")
# Only expense claims: the leave table keeps refusing pending_hr.


def upgrade() -> None:
    _status_check(AFTER)


def downgrade() -> None:
    # The old schema has no second step: send those claims back to the manager.
    op.execute("UPDATE expense_claims SET status = 'pending' WHERE status = 'pending_hr'")
    _status_check(BEFORE)


def _status_check(values: tuple[str, ...]) -> None:
    with op.batch_alter_table("expense_claims", schema=None) as batch_op:
        # Dropped by its full name; created without the prefix, which the naming
        # convention adds ("ck_expense_claims_").
        batch_op.drop_constraint(op.f("ck_expense_claims_expense_status"), type_="check")
        batch_op.create_check_constraint("expense_status", sa.column("status").in_(values))
