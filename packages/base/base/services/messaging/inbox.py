"""Consumer side of reliable delivery.

Delivery is at-least-once, so the same event arrives more than once. A claim
is recorded in the consumer's own database before the handler runs and the
outcome after it, so duplicates are dropped and a crash mid-handler is retried
rather than lost.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import select
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="Inbox") from exc

from base.models.definitions.messaging import InboxState
from base.services.messaging.protocol import InboxRow
from base.services.sql.service import SQLService
from contracts.events import Event

EventHandler = Callable[[Event], Awaitable[None]]


class InboxProcessor:
    def __init__(
        self,
        model: type[InboxRow],
        *,
        claim_timeout_seconds: int = 300,
    ) -> None:
        self._model = model
        self._claim_timeout = timedelta(seconds=claim_timeout_seconds)
        self._sql = SQLService()

    def wrap(self, handler: EventHandler) -> EventHandler:
        """Make a handler idempotent across redelivery and restarts."""

        async def process(event: Event) -> None:
            if not await self._aclaim(event):
                return
            try:
                await handler(event)
            except Exception as exc:
                await self._afinish(event, InboxState.FAILED, exc)
                raise
            await self._afinish(event, InboxState.PROCESSED)

        return process

    async def _aclaim(self, event: Event) -> bool:
        """Whether this process should run the handler for this event."""
        now = datetime.now(UTC)
        async with self._sql.atransaction() as session:
            row = await session.scalar(
                select(self._model)
                .where(self._model.event_id == event.meta.event_id)
                .with_for_update()
            )
            if row is None:
                session.add(
                    self._model(
                        event_id=event.meta.event_id,
                        topic=event.topic,
                        state=InboxState.PROCESSING,
                        locked_at=now,
                    )
                )
                return True
            if row.state == InboxState.PROCESSED:
                return False
            # A claim older than the timeout belongs to a process that died.
            if (
                row.state == InboxState.PROCESSING
                and row.locked_at > now - self._claim_timeout
            ):
                return False
            row.state = InboxState.PROCESSING
            row.locked_at = now
            row.attempts += 1
            row.last_error = None
            return True

    async def _afinish(
        self,
        event: Event,
        state: InboxState,
        error: Exception | None = None,
    ) -> None:
        async with self._sql.atransaction() as session:
            row = await session.get(self._model, event.meta.event_id)
            if row is None:
                raise RuntimeError("Inbox claim disappeared")
            row.state = state
            row.processed_at = (
                datetime.now(UTC) if state == InboxState.PROCESSED else None
            )
            row.last_error = (
                None if error is None else f"{type(error).__name__}: {error}"[:1000]
            )
