"""Add hit_rate_windows to forecasts

A hit rate is pooled across a sample of symbols, so the number of graded pairs
is symbols x folds -- but twenty stocks graded over the same three one-year
windows is three pieces of evidence, not sixty. Measured on real history over
the nightly's read window, the 5-day horizon grades 163 distinct windows and
the 20-day horizon 40, while the 63-day horizon manages 12 and the 1-year
horizon none at all. The 1-year "60% hit rate vs 52% naive" the ML model
produced was twenty correlated readings of a single year, published as a bare
percentage with nothing to say so.

This column records how many distinct out-of-sample windows a stored rate was
measured over, so both front ends can show it; rates measured over fewer than
`backtest.MIN_GRADED_WINDOWS` windows are stored as null rather than published
at all (Section 7.6, Section 22).

Revision ID: a91d4e5c37b2
Revises: cfef1ca2960f
Create Date: 2026-08-05 06:20:11.402913

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a91d4e5c37b2"
down_revision: str | Sequence[str] | None = "cfef1ca2960f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # As with the sibling column migrations: autogenerate keeps proposing
    # `uq_pattern_signals_symbol` on `pattern_signals`, which is pre-existing
    # drift from the retroactive `naming_convention` change and unrelated to
    # this column. SQLite cannot ALTER TABLE ADD CONSTRAINT outside batch mode,
    # so it stays out rather than being smuggled into an unrelated migration.
    op.add_column("forecasts", sa.Column("hit_rate_windows", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("forecasts", "hit_rate_windows")
