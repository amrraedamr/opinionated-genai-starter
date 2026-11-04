from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from base.services.broker import AbstractBroker
from base.services.broker.factory import build_broker
from base.services.messaging import OutboxDispatcher
from chat_api.models.orm.outbox import OutboxEventDTO
from chat_api.settings import settings

BROKER: AbstractBroker = build_broker(settings)


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
