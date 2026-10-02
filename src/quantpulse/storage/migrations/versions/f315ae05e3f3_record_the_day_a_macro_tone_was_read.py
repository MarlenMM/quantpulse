"""Record the day a market-regime row's macro tone was read

Handoff item A. GDELT refuses the nightly's macro-tone request on roughly four
nights in ten. The owner's call: retry, then carry the newest stored reading
forward if it is at most three trading sessions old -- which needs the day each
tone was read, so a carried value cannot be carried again past that limit and
the coverage sentence can name it. Existing rows keep NULL and read as their own
date.

Revision ID: f315ae05e3f3
Revises: 314d984542ff
Create Date: 2026-10-02 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f315ae05e3f3"
down_revision: str | Sequence[str] | None = "314d984542ff"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("market_regime", sa.Column("macro_tone_as_of", sa.Date(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("market_regime") as batch:
        batch.drop_column("macro_tone_as_of")
