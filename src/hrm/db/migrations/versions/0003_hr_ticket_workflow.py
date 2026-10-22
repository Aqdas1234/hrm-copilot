"""HR ticket workflow: categories from a fixed list, resolution and timestamps

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-22 20:30:31.799858

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CATEGORIES = ("payroll", "benefits", "leave", "access", "other")


def upgrade() -> None:
    # No tickets could be created before this revision, so no row can break the new CHECK.
    with op.batch_alter_table("hr_tickets", schema=None) as batch_op:
        batch_op.add_column(sa.Column("resolution", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.alter_column(
            "category",
            existing_type=sa.VARCHAR(length=50),
            type_=sa.String(length=20),
            existing_nullable=False,
        )
        # Written out: alter_column does not add an Enum's CHECK on MySQL.
        # Name without prefix: the naming convention adds "ck_hr_tickets_".
        batch_op.create_check_constraint("ticket_category", sa.column("category").in_(CATEGORIES))


def downgrade() -> None:
    with op.batch_alter_table("hr_tickets", schema=None) as batch_op:
        batch_op.drop_constraint("ticket_category", type_="check")
        batch_op.alter_column(
            "category",
            existing_type=sa.String(length=20),
            type_=sa.VARCHAR(length=50),
            existing_nullable=False,
        )
        batch_op.drop_column("closed_at")
        batch_op.drop_column("updated_at")
        batch_op.drop_column("resolution")
