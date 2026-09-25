"""Store each forecast's edge over naive, its interval, and its place in the stock's history

Finding 34. Every graded forecast was shown with a bare hit rate pooled across
twenty names -- "51.8%" beside a +46.8% twenty-day forecast reads as evidence
for that number. Measured on the published data, no model's edge over the naive
forecast excluded zero at any graded horizon (GBR 20-day: +3.5 points, 90%
interval -0.9 to +7.8). These columns carry that edge and its window-bootstrapped
interval with each row, and where the point forecast sits among the stock's own
past moves of the same length (a percentile and an outside-the-range flag).

Existing rows keep NULL: they were written before this was measured, and the
front ends say "not yet measured" rather than invent it.

Revision ID: a34e1d9c2b70
Revises: 15df3d8a9b80
Create Date: 2026-09-26 01:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a34e1d9c2b70"
down_revision: str | Sequence[str] | None = "15df3d8a9b80"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = (
    ("edge_vs_naive", sa.Float()),
    ("edge_ci_low", sa.Float()),
    ("edge_ci_high", sa.Float()),
    ("own_history_percentile", sa.Float()),
    ("outside_own_history", sa.Boolean()),
)


def upgrade() -> None:
    """Upgrade schema."""
    for name, kind in _COLUMNS:
        op.add_column("forecasts", sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("forecasts") as batch:
        for name, _kind in reversed(_COLUMNS):
            batch.drop_column(name)
