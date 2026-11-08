"""Add identity outbox.

Revision ID: 1doutbox001
Revises: 519ec68cf8c3
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlalchemy_utils

from alembic import op

revision: str = "1doutbox001"
down_revision: Union[str, Sequence[str], None] = "519ec68cf8c3"
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
    op.create_index(
        "idx_identity_outbox_pending", "outbox_event", ["published_at", "available_at"]
    )


def downgrade() -> None:
    op.drop_index("idx_identity_outbox_pending", table_name="outbox_event")
    op.drop_table("outbox_event")
