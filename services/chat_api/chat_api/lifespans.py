"""Lifespan hooks this service contributes to its composition root."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from base.services.background_tasks.in_memory import InMemoryBackgroundTaskService


@asynccontextmanager
async def background_tasks_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Drain in-flight background tasks before the process exits."""
    del app
    try:
        yield
    finally:
        await InMemoryBackgroundTaskService().ashutdown(timeout=10)
