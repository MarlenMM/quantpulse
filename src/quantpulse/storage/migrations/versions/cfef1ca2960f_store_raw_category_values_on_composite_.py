"""Store raw category values on composite_scores

The `*_score` columns are cross-sectional percentiles, and that transform is
lossy in exactly the way that matters here: a rank cannot be turned back into an
absolute reading. So `rating_mode="absolute"` was correct code that no interface
could reach -- a stored row could only ever be re-scored in relative mode.

These seven columns carry each category's value on its own native scale (0-100
fixed readings for fundamental/technical/analyst/smart_money, a [-1, 1] polarity
for sentiment/industry_macro, a raw mean/std ratio for momentum), which is
everything `build_composite` needs to re-run either mode from a stored row.

They live on this profile-keyed row rather than in a separate table because they
are genuinely profile-dependent: the conservative profile's
`prefer_low_volatility` makes `score_momentum` return negative volatility rather
than a risk-adjusted return, so "the raw momentum for AAPL today" has no single
profile-independent answer.

Nullable, so every already-stored row stays valid and simply has no absolute
reading available until the next nightly rewrites it.

Revision ID: cfef1ca2960f
Revises: f7b3c4c4d737
Create Date: 2026-08-04 18:10:54.829831

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cfef1ca2960f"
down_revision: str | Sequence[str] | None = "f7b3c4c4d737"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RAW_COLUMNS = (
    "fundamental_raw",
    "technical_raw",
    "analyst_raw",
    "sentiment_raw",
    "momentum_raw",
    "industry_macro_raw",
    "smart_money_raw",
)


def upgrade() -> None:
    """Upgrade schema."""
    # Autogenerate again proposed `uq_pattern_signals_symbol` on
    # `pattern_signals`: pre-existing drift on databases created before the
    # retroactive `naming_convention` change (a freshly migrated one shows
    # none), unrelated to these columns, and SQLite cannot ALTER TABLE ADD
    # CONSTRAINT without batch mode. Deliberately excluded, as in the two
    # preceding migrations.
    for column in _RAW_COLUMNS:
        op.add_column("composite_scores", sa.Column(column, sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    for column in reversed(_RAW_COLUMNS):
        op.drop_column("composite_scores", column)
