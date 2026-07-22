"""Phase 7 schema: forecasts + backtest_results tables

Revision ID: 398983318da7
Revises: 999438f3b47e
Create Date: 2026-07-22 19:50:08.221524

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "398983318da7"
down_revision: Union[str, Sequence[str], None] = "999438f3b47e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "forecasts",
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("generated_date", sa.Date(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=20), nullable=False),
        sa.Column("point_return", sa.Float(), nullable=False),
        sa.Column("point_price", sa.Float(), nullable=True),
        sa.Column("lower_price", sa.Float(), nullable=True),
        sa.Column("upper_price", sa.Float(), nullable=True),
        sa.Column("historical_hit_rate", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["symbol"],
            ["tickers.symbol"],
        ),
        sa.PrimaryKeyConstraint("symbol", "generated_date", "horizon_days", "model_name"),
    )
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("cadence", sa.String(length=10), nullable=False),
        sa.Column("n_periods", sa.Integer(), nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("cagr", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("benchmark_cagr", sa.Float(), nullable=True),
        sa.Column("benchmark_sharpe", sa.Float(), nullable=True),
        sa.Column("avg_turnover", sa.Float(), nullable=True),
        sa.Column("assumed_txn_cost", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("backtest_results")
    op.drop_table("forecasts")
