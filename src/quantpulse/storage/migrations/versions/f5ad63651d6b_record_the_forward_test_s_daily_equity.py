"""Record the forward test's daily equity

`backtest_results` answers "what would this have returned", and a backtest can
always be met with "you fitted that" -- no amount of bootstrap rigour fully
settles it. This table is the other half: a record written forward, one row per
refresh, of what a paper account following the *published* ratings was actually
worth on a day whose outcome nobody knew when the positions were chosen.

It also carries the signal `backtest_results` cannot. That table ranks
`momentum_category`, because five of the composite's seven categories hold
weeks of stored history rather than years -- so the Buy/Sell rating the whole
app publishes has never itself been testable. These rows are how that history
starts accumulating.

Arriving with its writer (`scripts/refresh_data.py`'s `paper_trading` step) per
the project's "schema alongside its writer" convention.

Revision ID: f5ad63651d6b
Revises: b4e17c9a2f01
Create Date: 2026-09-07 20:41:26.906136

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f5ad63651d6b"
down_revision: str | Sequence[str] | None = "b4e17c9a2f01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Keyed on `run_date` alone: the refresh writes at most one row per trading
    # day, and re-running a day must correct that day rather than append a
    # second, contradictory equity for it.
    op.create_table(
        "paper_trading_snapshots",
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("positions_held", sa.Integer(), nullable=False),
        sa.Column("benchmark_close", sa.Float(), nullable=True),
        sa.Column("rebalanced", sa.Boolean(), nullable=False),
        sa.Column("orders_submitted", sa.Integer(), nullable=False),
        sa.Column("orders_rejected", sa.Integer(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=True),
        sa.Column("signal_name", sa.String(length=50), nullable=False),
        sa.Column("profile", sa.String(length=30), nullable=False),
        sa.Column("target_positions", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("run_date", name=op.f("pk_paper_trading_snapshots")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("paper_trading_snapshots")
