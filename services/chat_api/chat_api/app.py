# ruff: noqa: E402, I001
"""Composition root for chat_api."""

from base.services.configuration import configure
from chat_api.settings import settings

# Register the service's settings before importing anything that reads them.
configure(settings)

from collections.abc import AsyncGenerator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from base.app import create_app  # noqa: E402
from base.capabilities.broker import BrokerDependency  # noqa: E402
from base.capabilities.prompts import Prompts  # noqa: E402
from base.capabilities.sql import SQLDependency  # noqa: E402
from base.capabilities.terminal_capture import TerminalCapture  # noqa: E402
from base.services.auth import principal_dependency  # noqa: E402
from base.services.background_tasks.in_memory import InMemoryBackgroundTaskService  # noqa: E402
from base.services.common.meta import Stored, scope_id  # noqa: E402
from base.services.logging.tee_stream import TerminalCaptureService  # noqa: E402
from base.tools.stream import StreamIterator  # noqa: E402
from chat_api import routers  # noqa: E402
from chat_api.messaging import BROKER, messaging_lifespan  # noqa: E402
from chat_api.prompts import templates  # noqa: E402
from chat_api.stores.prompt import PromptStore  # noqa: E402


@asynccontextmanager
async def background_tasks_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    del app
    try:
        yield
    finally:
        await InMemoryBackgroundTaskService().ashutdown(timeout=10)


app = create_app(
    settings=settings,
    routers=routers,
    prefix=settings.api_prefix,
    capabilities=[
        TerminalCapture(settings.service_name),
        Prompts(templates=[templates], store=PromptStore()),
        SQLDependency(),
        BrokerDependency(BROKER),
    ],
    cors_origins=settings.cors_origins_list or None,
    lifespan_hooks=[
        messaging_lifespan,
        background_tasks_lifespan,
    ],
)


@app.post(
    "/logs",
    dependencies=[principal_dependency],
)
async def get_logs() -> EventSourceResponse:
    request_scope_id = scope_id.get()
    capture_service = TerminalCaptureService(settings.service_name)
    stream_iterator = StreamIterator(source=capture_service.subscribe())
    return EventSourceResponse(
        stream_iterator,
        media_type="text/event-stream",
        background=BackgroundTask(Stored.delete_scope, request_scope_id),
    )


def start() -> None:
    from base.tools.webserver import run_webserver

    run_webserver(app_path="chat_api.app:app", settings=settings)
