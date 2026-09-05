"""Record which signal a backtest ranked, and the excess-return Kelly inputs

Three columns, both halves of one honesty problem on the Track Record page.

`signal_name` records what was actually ranked. The page called itself a
"followed the algorithm's ratings" track record while every stored run was
ranked by a hand-rolled trailing return, and nothing in the row could have told
a reader otherwise. Storing it per run means a row stays interpretable across a
change of signal -- which has now happened once (to `scoring.score_momentum`,
the app's own momentum category scorer) and will happen again when
`composite_scores` has the years of history the full rating would need.

`excess_win_rate`/`excess_payoff_ratio` are the same two measures taken against
the benchmark instead of against zero. The Kelly block sizes a position from
them, and on absolute returns it returned a confident positive fraction for a
run that trailed its own buy-and-hold benchmark -- measuring the market's return
and calling it the strategy's edge. Existing rows keep NULL: they were produced
by a different signal and were never measured this way, and back-filling a
number nobody computed is exactly the kind of quiet fabrication this table's
`assumed_txn_cost` column exists to prevent.

Revision ID: b4e17c9a2f01
Revises: aba9f58ad26c
Create Date: 2026-09-05 22:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4e17c9a2f01"
down_revision: str | Sequence[str] | None = "aba9f58ad26c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # As in `f7b3c4c4d737`, autogenerate proposes `uq_pattern_signals_symbol` on
    # `pattern_signals`. That is pre-existing drift on databases created before
    # the retroactive `naming_convention` change, unrelated to these columns, and
    # SQLite cannot ALTER TABLE ADD CONSTRAINT outside batch mode -- so it stays
    # out of this migration too.
    op.add_column("backtest_results", sa.Column("excess_win_rate", sa.Float(), nullable=True))
    op.add_column("backtest_results", sa.Column("excess_payoff_ratio", sa.Float(), nullable=True))
    op.add_column("backtest_results", sa.Column("signal_name", sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("backtest_results", "signal_name")
    op.drop_column("backtest_results", "excess_payoff_ratio")
    op.drop_column("backtest_results", "excess_win_rate")
