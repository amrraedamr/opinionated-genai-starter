from enum import StrEnum


class FailureKind(StrEnum):
    """What went wrong, in terms no client or domain owns.

    An SDK handler decides which kind an exception is; a domain decides which
    error class a kind becomes. Stating it once is what lets one SDK serve
    several domains without repeating the classification.
    """

    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


RETRYABLE_FAILURES: frozenset[FailureKind] = frozenset(
    {
        FailureKind.RATE_LIMIT,
        FailureKind.SERVER,
        FailureKind.TIMEOUT,
        FailureKind.CONNECTION,
    }
)
"""Kinds worth trying again. A kind is retryable whoever raised it.

The four describe the moment rather than the request, so the same call a little
later plausibly succeeds. Everything else describes the request itself, and a
second attempt would arrive at the identical answer.
"""
