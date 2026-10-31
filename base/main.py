from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from base import routers
from base.services.auth.dependencies import validate_user_dependency
from base.services.common.meta import Stored, scope_id
from base.services.configuration import settings
from base.services.logging.console import ConsoleLoggerService
from base.services.logging.tee_stream import TerminalCaptureService
from base.services.prompt.loader import PromptLoader
from base.tools.exception_handlers import AppExceptionHandler
from base.tools.routers import init_routers
from base.tools.stream import StreamIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger = ConsoleLoggerService()
    logger.info("Application starting")

    capture_service = TerminalCaptureService(settings.app_name)
    capture_service.start()

    try:
        await PromptLoader().initialize()
        yield
    finally:
        logger.info("Application shutting down")
        await PromptLoader().shutdown()
        await capture_service.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def register_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    scope_id.set(uuid4())
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        should_delete = response is None or not response.headers.get(
            "content-type", ""
        ).startswith("text/event-stream")
        if should_delete:
            Stored.delete_scope()


init_routers(
    app,
    routers=routers,
    prefix="/api/7r",
)


AppExceptionHandler().register(app)


@app.get("/", tags=["Main"])
def root():
    return {
        "message": "Welcome to the Opinionated GenAI Starter API!",
        "version": "0.1",
    }


@app.get("/health-check", tags=["Main"])
def health_check():
    return {"status": "ok"}


@app.post(
    "/logs",
    dependencies=[validate_user_dependency],
)
async def get_logs():
    capture_service = TerminalCaptureService(settings.app_name)
    stream_iterator = StreamIterator(source=capture_service.subscribe())
    return EventSourceResponse(stream_iterator, media_type="text/event-stream")
