"""Which environment variable selects a backend, for an error to name."""

from enum import StrEnum


class BackendSetting(StrEnum):
    """The variable an operator changes to pick a different backend."""

    SQL = "SQL_BACKEND"
    BROKER = "BROKER"
