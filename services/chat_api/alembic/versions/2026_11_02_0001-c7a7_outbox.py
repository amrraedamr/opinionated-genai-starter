"""Add chat outbox.

Revision ID: c7a7outbox001
Revises: 5d517c618371
"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlalchemy_utils

from alembic import op

revision: str = "c7a7outbox001"
down_revision: Union[str, Sequence[str], None] = "5d517c618371"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "outbox_event",
        sa.Column("id", sqlalchemy_utils.types.uuid.UUIDType(), nullable=False),
        sa.Column("topic", sa.String(length=255), nullable=False),
        sa.Column("event", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_chat_outbox_pending", "outbox_event", ["published_at", "available_at"])


def downgrade() -> None:
    op.drop_index("idx_chat_outbox_pending", table_name="outbox_event")
    op.drop_table("outbox_event")
