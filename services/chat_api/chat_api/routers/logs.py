"""Streams this service's captured stdout/stderr over SSE."""

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from base.services.auth import principal_dependency
from base.services.common.meta import Stored, scope_id
from base.services.logging.tee_stream import TerminalCaptureService
from base.tools.stream import StreamIterator
from chat_api.settings import settings

router = APIRouter(prefix="/logs", tags=["logs"])


@router.post("", dependencies=[principal_dependency])
async def get_logs() -> EventSourceResponse:
    """Subscribe to the terminal capture channel for the lifetime of the stream."""
    request_scope_id = scope_id.get()
    capture_service = TerminalCaptureService(settings.service_name)
    stream_iterator = StreamIterator(source=capture_service.subscribe())
    return EventSourceResponse(
        stream_iterator,
        media_type="text/event-stream",
        background=BackgroundTask(Stored.delete_scope, request_scope_id),
    )
