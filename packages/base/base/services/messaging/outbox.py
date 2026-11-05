"""Producer side of reliable delivery.

A domain write and its event commit together in one transaction, then a
dispatcher publishes what was committed. Publishing from the request handler
instead would lose the event whenever the broker is unavailable.
"""

import asyncio
from datetime import UTC, datetime, timedelta

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="Outbox") from exc

from base.services.broker import AbstractBroker
from base.services.logging.console import ConsoleLoggerService
from base.services.messaging.protocol import OutboxRow
from base.services.sql.service import SQLService
from contracts.events import Event

MAX_BACKOFF_SECONDS = 60


class OutboxDispatcher:
    def __init__(
        self,
        broker: AbstractBroker,
        model: type[OutboxRow],
        *,
        poll_seconds: float = 0.5,
        batch_size: int = 50,
    ) -> None:
        self._broker = broker
        self._model = model
        self._poll_seconds = poll_seconds
        self._batch_size = batch_size
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._logger = ConsoleLoggerService()

    async def astart(self) -> None:
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="outbox-dispatcher")

    async def astop(self) -> None:
        self._stopping.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=5)
        except TimeoutError:
            self._task.cancel()
        except Exception:
            self._logger.exception("Outbox dispatcher stopped on an error")
        finally:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        """A failed poll must not kill the dispatcher: a dead task would stop
        publishing silently and forever."""
        while not self._stopping.is_set():
            try:
                dispatched = await self.adispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception("Outbox dispatch failed; retrying")
                dispatched = 0
            if not dispatched:
                try:
                    await asyncio.wait_for(self._stopping.wait(), self._poll_seconds)
                except TimeoutError:
                    pass

    async def adispatch_once(self) -> int:
        """Publish one batch of committed events. Returns how many were claimed."""
        now = datetime.now(UTC)
        async with SQLService().atransaction() as session:
            rows = list(
                (
                    await session.scalars(
                        select(self._model)
                        .where(
                            self._model.published_at.is_(None),
                            self._model.available_at <= now,
                        )
                        .order_by(self._model.occurred_at)
                        .limit(self._batch_size)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for row in rows:
                try:
                    await self._broker.apublish(Event.model_validate(row.event))
                    row.published_at = datetime.now(UTC)
                    row.last_error = None
                except Exception as exc:
                    row.attempts += 1
                    row.last_error = f"{type(exc).__name__}: {exc}"[:1000]
                    row.available_at = datetime.now(UTC) + timedelta(
                        seconds=min(2 ** max(row.attempts - 1, 0), MAX_BACKOFF_SECONDS)
                    )
                    self._logger.exception("Outbox publication failed for %s", row.id)
            await session.flush()
            return len(rows)


def add_event(session: AsyncSession, model: type[OutboxRow], event: Event) -> None:
    """Stage an event in the caller's transaction, so it commits with the write."""
    session.add(
        model(
            id=event.meta.event_id,
            topic=event.topic,
            event=event.model_dump(mode="json"),
            occurred_at=event.meta.occurred_at,
        )
    )
