from collections.abc import Awaitable, Callable
from enum import StrEnum

from contracts.events import Event

EventHandler = Callable[[Event], Awaitable[None]]
"""What an inbox runs for one event."""


class InboxState(StrEnum):
    """Where one consumed event has got to."""

    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
