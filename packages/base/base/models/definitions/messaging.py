from enum import StrEnum


class InboxState(StrEnum):
    """Where one consumed event has got to."""

    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
