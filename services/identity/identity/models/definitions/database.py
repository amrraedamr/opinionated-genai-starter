from enum import StrEnum


class DatabaseTable(StrEnum):
    """Tables owned by identity."""

    USER = "user"
    USER_IDENTITY = "user_identity"
