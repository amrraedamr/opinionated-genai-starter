"""Give each user the tenant their tokens name.

Revision ID: 2tenant0001
Revises: 1doutbox001
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "2tenant0001"
down_revision: Union[str, Sequence[str], None] = "1doutbox001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column(
            "tenant_id", sa.String(length=255), nullable=False, server_default="default"
        ),
    )


def downgrade() -> None:
    op.drop_column("user", "tenant_id")
