"""Add durable worker inbox.

Revision ID: workerinbox001
Revises:
"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlalchemy_utils

from alembic import op

revision: str = "workerinbox001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "inbox_event",
        sa.Column("event_id", sqlalchemy_utils.types.uuid.UUIDType(), nullable=False),
        sa.Column("topic", sa.String(length=255), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("idx_inbox_state_locked", "inbox_event", ["state", "locked_at"])


def downgrade() -> None:
    op.drop_index("idx_inbox_state_locked", table_name="inbox_event")
    op.drop_table("inbox_event")
