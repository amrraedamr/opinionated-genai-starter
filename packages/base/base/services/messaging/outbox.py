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

from base.models.definitions.telemetry import (
    MESSAGING_DESTINATION,
    MESSAGING_MESSAGE_ID,
    MESSAGING_OPERATION,
)
from base.services.broker import AbstractBroker
from base.services.logging.console import ConsoleLoggerService
from base.services.messaging.protocol import OutboxRow
from base.services.sql.service import SQLService
from base.tools.tenancy import TenantContext
from base.tools.tracing import OptionalTelemetry
from contracts.events import Event

MAX_BACKOFF_SECONDS = 60


DEFAULT_MAX_ATTEMPTS: int = 10
"""Attempts before a row is left alone.

A broker that refuses one event forever would otherwise take a slot in every
batch for as long as the service runs. The row keeps its error so whoever
investigates can replay it.
"""


class OutboxDispatcher:
    def __init__(
        self,
        broker: AbstractBroker,
        model: type[OutboxRow],
        *,
        poll_seconds: float = 0.5,
        batch_size: int = 50,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self._broker = broker
        self._model = model
        self._poll_seconds = poll_seconds
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._logger = ConsoleLoggerService()
        self._telemetry = OptionalTelemetry()

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
                            # A row that has used its attempts stays in the
                            # table with its error, and stops being claimed.
                            self._model.attempts < self._max_attempts,
                        )
                        .order_by(self._model.occurred_at)
                        .limit(self._batch_size)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for row in rows:
                event = Event.model_validate(row.event)
                with self._telemetry.continued_span(
                    f"outbox dispatch {event.topic}",
                    event.meta.traceparent,
                    kind="producer",
                    attributes={
                        MESSAGING_DESTINATION: event.topic,
                        MESSAGING_MESSAGE_ID: str(event.meta.event_id),
                        MESSAGING_OPERATION: "publish",
                    },
                ):
                    try:
                        await self._broker.apublish(event)
                        row.published_at = datetime.now(UTC)
                        row.last_error = None
                    except Exception as exc:
                        row.attempts += 1
                        row.last_error = f"{type(exc).__name__}: {exc}"[:1000]
                        row.available_at = datetime.now(UTC) + timedelta(
                            seconds=min(
                                2 ** max(row.attempts - 1, 0), MAX_BACKOFF_SECONDS
                            )
                        )
                        if row.attempts >= self._max_attempts:
                            self._logger.error(
                                f"Outbox row {row.id} parked after {row.attempts} "
                                f"attempts; it will not be retried: {row.last_error}"
                            )
                        else:
                            self._logger.exception(
                                "Outbox publication failed for %s", row.id
                            )
            await session.flush()
            return len(rows)


def add_event(session: AsyncSession, model: type[OutboxRow], event: Event) -> None:
    """Stage an event in the caller's transaction, stamped with trace and tenant."""
    updates: dict[str, str] = {}
    if event.meta.traceparent is None:
        traceparent = OptionalTelemetry.current_traceparent()
        if traceparent is not None:
            updates["traceparent"] = traceparent
    if event.meta.tenant_id is None:
        tenant = TenantContext.get()
        if tenant is not None:
            updates["tenant_id"] = str(tenant)
    stamped = (
        event.model_copy(update={"meta": event.meta.model_copy(update=updates)})
        if updates
        else event
    )
    session.add(
        model(
            id=stamped.meta.event_id,
            topic=stamped.topic,
            event=stamped.model_dump(mode="json"),
            occurred_at=stamped.meta.occurred_at,
        )
    )
