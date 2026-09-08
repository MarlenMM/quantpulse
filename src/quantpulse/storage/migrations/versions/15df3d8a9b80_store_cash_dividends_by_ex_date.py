"""Store cash dividends by ex-date

The Portfolio Manager could tell you what you hold and what it is worth, and
nothing about what it has *paid* you. Dividend yield existed only as a
fundamental metric -- a forward-looking ratio on a company, not a record of cash
this portfolio actually received.

Keyed on `(symbol, ex_date)` and append-only: a declared dividend does not
change, and the weekly refetch of a name's whole history has to recognise what
is already stored rather than duplicate it.

`ex_date` rather than the pay date, because entitlement turns on the ex-date --
`portfolio.performance.dividend_income` decides on exactly this column, and the
pay date would credit income to a holder who had already sold.

Arriving with its writer (`scripts/refresh_data.py`'s `dividends` step) per the
project's "schema alongside its writer" convention.

Revision ID: 15df3d8a9b80
Revises: f5ad63651d6b
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "15df3d8a9b80"
down_revision: str | Sequence[str] | None = "f5ad63651d6b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "dividends",
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["symbol"], ["tickers.symbol"], name=op.f("fk_dividends_symbol_tickers")
        ),
        sa.PrimaryKeyConstraint("symbol", "ex_date", name=op.f("pk_dividends")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("dividends")
