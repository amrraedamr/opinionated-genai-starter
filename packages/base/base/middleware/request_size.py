from starlette.types import ASGIApp, Message, Receive, Scope, Send

from base.middleware.responses import reject


class _BodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """Reject oversized bodies by declared length, then by what actually arrives."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                body_size = int(content_length)
            except ValueError:
                await reject(scope, receive, send, "Invalid Content-Length", 400)
                return
            if body_size > self.max_bytes:
                await reject(scope, receive, send, "Request body too large", 413)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await reject(scope, receive, send, "Request body too large", 413)
