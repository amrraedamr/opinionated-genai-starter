from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from base.services.broker import AbstractBroker
from base.services.broker.memory import InMemoryBroker
from base.services.messaging import OutboxDispatcher
from identity.models.orm.outbox import OutboxEventDTO
from identity.settings import settings


def build_broker() -> AbstractBroker:
    if settings.broker == "redis":
        from base.services.broker.redis_streams import RedisStreamsBroker

        return RedisStreamsBroker()
    return InMemoryBroker()


BROKER = build_broker()


@asynccontextmanager
async def messaging_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    del app
    broker = BROKER
    dispatcher = OutboxDispatcher(broker, OutboxEventDTO)
    await dispatcher.astart()
    try:
        yield
    finally:
        await dispatcher.astop()
        await broker.astop()
