"""Composition root for ingestion_worker.

No HTTP surface: this service consumes from the broker and nothing else.
"""

import asyncio
import signal

from base.services.configuration import configure
from ingestion_worker.settings import settings

configure(settings)

from base.services.broker import AbstractBroker  # noqa: E402
from base.services.logging.console import ConsoleLoggerService  # noqa: E402
from base.services.messaging import InboxProcessor  # noqa: E402
from base.services.sql.service import SQLService  # noqa: E402
from contracts.v1 import chat, identity  # noqa: E402
from ingestion_worker import handlers  # noqa: E402
from ingestion_worker.models.inbox import InboxEventDTO  # noqa: E402

logger = ConsoleLoggerService()


def build_broker() -> AbstractBroker:
    if settings.broker == "redis":
        from base.services.broker.redis_streams import RedisStreamsBroker

        return RedisStreamsBroker()
    from base.services.broker.memory import InMemoryBroker

    return InMemoryBroker()


async def arun(broker: AbstractBroker | None = None) -> AbstractBroker:
    """Subscribe every handler and begin consuming."""
    broker = broker or build_broker()
    group = settings.consumer_group
    inbox = InboxProcessor(
        InboxEventDTO,
        claim_timeout_seconds=settings.inbox_claim_timeout_seconds,
    )
    await broker.asubscribe(
        identity.USER_REGISTERED,
        group,
        inbox.wrap(handlers.ahandle_user_registered),
    )
    await broker.asubscribe(
        chat.CONVERSATION_CREATED,
        group,
        inbox.wrap(handlers.ahandle_conversation_created),
    )
    await broker.astart()
    logger.info(f"{settings.service_name} consuming as group {group}")
    return broker


def start() -> None:
    async def _main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stop.set)
        broker = await arun()
        try:
            await stop.wait()
        finally:
            await broker.astop()
            await SQLService().adispose()

    asyncio.run(_main())
