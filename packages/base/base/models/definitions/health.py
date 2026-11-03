from enum import StrEnum


class ProbeStatus(StrEnum):
    """Outcome of a single capability readiness probe.

    ``DEGRADED`` means the capability answered but something is not right —
    the service can still take traffic. ``FAILED`` means it is unusable, and
    ``/readyz`` answers 503 so an orchestrator stops routing to this instance.
    """

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
