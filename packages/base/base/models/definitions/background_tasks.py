from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundTaskBackend(StrEnum):
    IN_MEMORY = "in_memory"
    CELERY = "celery"
    ARQ = "arq"
