"""Add a coverage tier to tickers

Separates "how much do we know about this symbol" from "is it in the index we
track". `ranked` symbols are fetched and scored nightly; `catalogue` symbols are
searchable and analysable on demand, and nothing else.

`server_default='ranked'` is the load-bearing part: every row that already
exists is a scored S&P 500 constituent, so backfilling them as ranked is both
correct and the only value that leaves existing behaviour untouched.

Revision ID: aba9f58ad26c
Revises: a91d4e5c37b2
Create Date: 2026-08-08 17:47:57.664160

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "aba9f58ad26c"
down_revision: Union[str, Sequence[str], None] = "a91d4e5c37b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "tickers",
        sa.Column("coverage", sa.String(length=16), server_default="ranked", nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema.

    Dropping the column also drops the distinction, so a downgraded database
    treats every catalogue row as if it were ranked. Callers that care should
    delete catalogue rows first -- there is no way to express "these were only
    ever names" once the column is gone.
    """
    op.drop_column("tickers", "coverage")
