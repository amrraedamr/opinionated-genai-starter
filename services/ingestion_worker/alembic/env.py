import ingestion_worker.models.inbox  # noqa: F401
from base.tools.migrations import run_migrations
from ingestion_worker.models import WorkerBase
from ingestion_worker.settings import settings

run_migrations(WorkerBase.metadata, settings)
