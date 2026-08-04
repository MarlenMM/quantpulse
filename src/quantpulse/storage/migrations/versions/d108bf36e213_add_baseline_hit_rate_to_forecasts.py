"""Add baseline_hit_rate to forecasts

A model's own directional hit-rate is close to meaningless on its own -- it has
to be read against the naive null's rate over the same folds. Measured on real
history the baseline scored exactly the fraction of periods that happened to be
up (63.6% at h=63) while ARIMA scored 50.0%, so a bare "53%" both reads as
modest skill and hides a model doing worse than the null it exists to beat.
`backtest.walk_forward_accuracy` already computed this figure; it had nowhere
to be stored, so it never reached either front end (Section 7.6).

Revision ID: d108bf36e213
Revises: 29891defa460
Create Date: 2026-08-04 11:57:43.217128

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d108bf36e213"
down_revision: str | Sequence[str] | None = "29891defa460"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Autogenerate also proposed creating `uq_pattern_signals_symbol` on
    # `pattern_signals`. That is pre-existing drift from the retroactive
    # `naming_convention` change, unrelated to this column, and SQLite cannot
    # ALTER TABLE ADD CONSTRAINT without batch mode -- so it is deliberately
    # left out rather than smuggled into an unrelated migration.
    op.add_column("forecasts", sa.Column("baseline_hit_rate", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("forecasts", "baseline_hit_rate")
