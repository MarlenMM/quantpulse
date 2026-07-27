"""Phase 7 schema: bootstrap CI columns on backtest_results

Revision ID: 268835023399
Revises: 398983318da7
Create Date: 2026-07-27 14:22:14.936729

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "268835023399"
down_revision: Union[str, Sequence[str], None] = "398983318da7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CI_COLUMNS = (
    "sharpe_ci_low",
    "sharpe_ci_high",
    "cagr_ci_low",
    "cagr_ci_high",
    "ci_confidence_level",
)


def upgrade() -> None:
    """Upgrade schema."""
    # All nullable: runs already stored (and future runs too short to bootstrap
    # honestly) legitimately have no interval, and a null reads as "not
    # established" rather than a fabricated bound.
    with op.batch_alter_table("backtest_results") as batch_op:
        for column in _CI_COLUMNS:
            batch_op.add_column(sa.Column(column, sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("backtest_results") as batch_op:
        for column in reversed(_CI_COLUMNS):
            batch_op.drop_column(column)
