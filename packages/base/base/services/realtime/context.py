"""Request-style scope, correlation id and span for a websocket connection."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from base.models.view_models.realtime import ConnectionTicket
from base.services.common.meta import Stored, scope_id
from base.tools.correlation import adopt_correlation_id, correlation_id
from base.tools.tenancy import CallerContext, TenantContext
from base.tools.tracing import OptionalTelemetry


class RealtimeContext:
    """Opens and tears down the scope a connection runs inside."""

    @staticmethod
    @asynccontextmanager
    async def connection(
        ticket: ConnectionTicket, *, channel: str | None = None
    ) -> AsyncGenerator[None]:
        """Hold a scope for the life of one socket."""
        scope_id.set(uuid4())
        adopt_correlation_id()
        caller = str(ticket.user_id) if ticket.user_id else ticket.subject
        with TenantContext.scope(ticket.tenant_id), CallerContext.scope(caller):
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
