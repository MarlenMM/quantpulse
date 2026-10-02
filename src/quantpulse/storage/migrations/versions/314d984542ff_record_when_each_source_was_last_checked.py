"""Record when each upstream source was last checked

Finding 36. The freshness strip printed "Institutional ownership: 169 days ago"
for the newest quarter SEC had published, which read as neglect. It now names
the period ("Q1 2026 filings") and says "the newest SEC publishes" -- a claim
only a check can support, because SEC's publication lag is usually days and on
2026-10-02 was more than a month. The weekly 13F step writes one row here on
each successful check.

Revision ID: 314d984542ff
Revises: a34e1d9c2b70
Create Date: 2026-10-02 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "314d984542ff"
down_revision: str | Sequence[str] | None = "a34e1d9c2b70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "source_checks",
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("checked_on", sa.Date(), nullable=False),
        sa.Column("newest_period", sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint("source", name=op.f("pk_source_checks")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("source_checks")
