"""ASGI middleware shared by every HTTP service."""

from base.tools.errors import MissingExtraError

try:
    import fastapi  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("server", "fastapi", feature="HTTP middleware") from exc

from base.middleware.request_context import SECURITY_HEADERS, request_context
from base.middleware.request_size import RequestSizeLimitMiddleware
from base.middleware.traffic_control import TrafficControlMiddleware

__all__ = [
    "SECURITY_HEADERS",
    "RequestSizeLimitMiddleware",
    "TrafficControlMiddleware",
    "request_context",
]
