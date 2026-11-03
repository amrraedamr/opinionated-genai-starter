from fastapi.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from base.models.view_models import ErrorResponse


async def reject(
    scope: Scope,
    receive: Receive,
    send: Send,
    message: str,
    status_code: int,
    headers: dict[str, str] | None = None,
) -> None:
    """Send an `ErrorResponse` from inside raw ASGI middleware."""
    response = JSONResponse(
        ErrorResponse(error=message).serialize(exclude_none=True),
        status_code=status_code,
        headers=headers,
    )
    await response(scope, receive, send)
