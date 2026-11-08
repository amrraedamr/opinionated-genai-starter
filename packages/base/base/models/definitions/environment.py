"""Where a service is running, and what that permits."""

from enum import StrEnum


class Environment(StrEnum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


DEBUG_ENVIRONMENTS: frozenset[Environment] = frozenset(
    {Environment.LOCAL, Environment.DEVELOPMENT}
)
"""Where an exception may be shown to whoever provoked it. Named as those
that may, so one added later discloses nothing until someone says so."""
