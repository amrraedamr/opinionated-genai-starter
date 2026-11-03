from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from base.models.definitions.background_tasks import BackgroundTaskBackend
from base.models.view_models import BaseModel

T = TypeVar("T")


class TaskHandle(BaseModel, Generic[T]):
    id: UUID
    name: str | None = None
    backend: BackgroundTaskBackend
    created_at: datetime
    metadata: dict[str, Any] | None = None
