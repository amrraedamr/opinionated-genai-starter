"""User identity and nullable password_hash

Revision ID: a1b2c3d4e5f6
Revises: 74a469122a52
Create Date: 2026-08-16 00:01:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlalchemy_utils

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "74a469122a52"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.alter_column(
            "password_hash",
            existing_type=sa.String(length=255),
            nullable=True,
        )

    op.create_table(
        "user_identity",
        sa.Column("id", sqlalchemy_utils.types.uuid.UUIDType(), nullable=False),
        sa.Column("user_id", sqlalchemy_utils.types.uuid.UUIDType(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("email_at_link", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "subject",
            name="uq_user_identity_provider_subject",
        ),
    )
    with op.batch_alter_table("user_identity", schema=None) as batch_op:
        batch_op.create_index("idx_user_identity_user_id", ["user_id"], unique=False)
        batch_op.create_index("idx_user_identity_provider", ["provider"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("user_identity", schema=None) as batch_op:
        batch_op.drop_index("idx_user_identity_provider")
        batch_op.drop_index("idx_user_identity_user_id")

    op.drop_table("user_identity")

    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.alter_column(
            "password_hash",
            existing_type=sa.String(length=255),
            nullable=False,
        )
