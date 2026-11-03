from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response

from base.services.common.meta import Stored, scope_id
from base.tools import correlation

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

CallNext = Callable[[Request], Awaitable[Response]]


async def request_context(request: Request, call_next: CallNext) -> Response:
    """Open a request scope, adopt a correlation id, and set security headers.

    A streamed response outlives this function, so its scope is torn down by
    the producer rather than here.
    """
    scope_id.set(uuid4())
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    incoming = correlation.adopt(
        request.headers.get(correlation.HEADER),
        request.headers.get(correlation.TRACEPARENT),
    )
    response: Response | None = None
    try:
        response = await call_next(request)
        response.headers[correlation.HEADER] = str(incoming)
        response.headers["X-Request-ID"] = request_id
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response
    finally:
        streaming = response is not None and response.headers.get(
            "content-type", ""
        ).startswith("text/event-stream")
        if not streaming:
            Stored.delete_scope()
