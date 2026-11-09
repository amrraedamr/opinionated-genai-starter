"""A websocket's equivalent of a request scope, which it has never had.

`request_context` is registered with `app.middleware("http")`, so a socket
gets none of it: no scope id, so a scoped service resolves whatever the last
HTTP request left behind; no correlation id, so nothing a socket logs can be
tied to anything else; and no span, so a conversation held over minutes is
invisible.

A socket is also longer-lived than a request, which makes each of those worse
rather than better.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from base.models.view_models.realtime import ConnectionTicket
from base.services.common.meta import Stored, scope_id
from base.tools.correlation import adopt_correlation_id, correlation_id
from base.tools.tenancy import TenantContext
from base.tools.tracing import OptionalTelemetry


class RealtimeContext:
    """Opens and tears down the scope a connection runs inside."""

    @staticmethod
    @asynccontextmanager
    async def connection(
        ticket: ConnectionTicket, *, channel: str | None = None
    ) -> AsyncGenerator[None]:
        """Hold a scope for the life of one socket.

        The scope is torn down on the way out whatever happened, because a
        socket that drops does not come back to clean up after itself, and a
        leaked scope keeps a service instance alive per dead connection.
        """
        scope_id.set(uuid4())
        adopt_correlation_id()
        with TenantContext.scope(ticket.tenant_id):
            with OptionalTelemetry.continued_span(
                "realtime connection",
                None,
                kind="server",
                attributes={"realtime.channel": channel} if channel else None,
            ):
                try:
                    yield
                finally:
                    Stored.delete_scope()

    @staticmethod
    def current_correlation_id() -> str:
        """What this connection should stamp on everything it logs."""
        current = correlation_id.get(None)
        return str(current) if current is not None else ""


__all__ = ["RealtimeContext"]
