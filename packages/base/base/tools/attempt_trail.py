"""What one outbound call went through, for the length of that call.

A ContextVar rather than handler state because one handler serves many
concurrent calls, and each task needs its own.
"""

import contextvars
from typing import NamedTuple


class AttemptTrail(NamedTuple):
    """How many tries one call took, and what went wrong on the way."""

    attempts: int = 0
    errors: tuple[str, ...] = ()
    """One entry per attempt that failed, in order. A call that succeeded
    first time has none; a call that recovered shows what it recovered from."""


trail: contextvars.ContextVar[AttemptTrail] = contextvars.ContextVar(
    "provider_call_trail"
)


def start_attempt_trail() -> None:
    """Begin a new call, discarding whatever the last one left behind."""
    trail.set(AttemptTrail())


def mark_attempt() -> None:
    """Count one try against the call in progress."""
    current = trail.get(AttemptTrail())
    trail.set(current._replace(attempts=current.attempts + 1))


def mark_attempt_failure(error: Exception) -> None:
    """Record what an attempt failed with."""
    current = trail.get(AttemptTrail())
    trail.set(
        current._replace(errors=(*current.errors, f"{type(error).__name__}: {error}"))
    )


def take_attempt_trail() -> AttemptTrail:
    """Read what the call that just finished went through, and forget it.

    Reading clears it so a later record cannot inherit an earlier call's trail.
    """
    current = trail.get(AttemptTrail())
    trail.set(AttemptTrail())
    return current
