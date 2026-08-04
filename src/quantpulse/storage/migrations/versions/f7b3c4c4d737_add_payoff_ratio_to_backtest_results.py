"""Add payoff_ratio to backtest_results

A Kelly position size needs two numbers from a real track record: how often the
strategy wins, and how much a win pays relative to what a loss costs. `win_rate`
was already stored; the payoff ratio was measured nowhere, which is why
`optimization.kelly_position_fraction` had no caller despite being a Section 27
requirement and its own docstring pointing at `backtest.py` for the input.
`backtest.payoff_ratio` now derives it from the same `period_returns` series
every other headline metric comes from.

Revision ID: f7b3c4c4d737
Revises: d108bf36e213
Create Date: 2026-08-04 17:26:05.560687

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f7b3c4c4d737"
down_revision: str | Sequence[str] | None = "d108bf36e213"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Autogenerate again proposed `uq_pattern_signals_symbol` on
    # `pattern_signals`. That is pre-existing drift on databases created before
    # the retroactive `naming_convention` change (a freshly migrated DB shows
    # none), unrelated to this column, and SQLite cannot ALTER TABLE ADD
    # CONSTRAINT without batch mode -- so it stays out of this migration too.
    op.add_column("backtest_results", sa.Column("payoff_ratio", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("backtest_results", "payoff_ratio")
