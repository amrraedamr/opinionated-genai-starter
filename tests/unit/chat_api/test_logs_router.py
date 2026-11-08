"""The log stream: a subscription held open for as long as the client listens.

Two things matter and neither is the payload. The route is behind the same
guard as everything else, declared as a dependency rather than read from a
signature. And the request scope is torn down by the stream rather than by
the middleware, which lets go of a `text/event-stream` response early — so
the teardown rides on the response's background task.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sse_starlette.sse import EventSourceResponse

from base.middleware import request_context
from base.models.definitions.cache import CacheChannels
from base.services.auth.bearer import current_principal
from base.services.cache.memory import InMemoryCacheService
from base.services.common.meta import Stored, scope_id
from base.tools.exception_handlers import AppExceptionHandler
from base.tools.stream import StreamIterator
from chat_api.routers import logs as logs_router


@pytest.fixture
def app(chat_settings: Any) -> FastAPI:
    """Only the log route, with the request scope the stream relies on."""
    application = FastAPI()
    application.middleware("http")(request_context)
    application.include_router(logs_router.router, prefix="/api")
    AppExceptionHandler().register(application)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def channel() -> str:
    return CacheChannels.LOGS.for_app(logs_router.settings.service_name)


# The guard


def test_subscribing_without_a_token_is_refused(client: TestClient) -> None:
    assert client.post("/api/logs").status_code == 401


def test_subscribing_with_a_token_that_is_not_one_is_refused(
    client: TestClient,
) -> None:
    response = client.post("/api/logs", headers={"Authorization": "Bearer nonsense"})
    assert response.status_code == 401


def test_the_guard_is_declared_rather_than_taken_as_an_argument() -> None:
    """The handler needs no caller, so the dependency is on the route itself."""
    route = next(
        r
        for r in logs_router.router.routes
        if r.path == "/logs"  # type: ignore[attr-defined]
    )
    assert [d.call for d in route.dependant.dependencies] == [current_principal]
    assert logs_router.get_logs.__code__.co_argcount == 0


# The stream


async def test_the_response_is_an_event_stream(chat_settings: Any) -> None:
    scope_id.set(uuid.uuid4())
    response = await logs_router.get_logs()

    assert isinstance(response, EventSourceResponse)
    assert response.media_type == "text/event-stream"


async def test_the_stream_tears_its_own_request_scope_down(
    chat_settings: Any,
) -> None:
    """The middleware lets go of a streamed response before it has drained."""
    scope = uuid.uuid4()
    scope_id.set(scope)
    response = await logs_router.get_logs()

    assert response.background is not None
    assert response.background.func is Stored.delete_scope
    assert response.background.args == (scope,)


async def test_the_subscription_carries_what_the_service_printed(
    chat_settings: Any, channel: str
) -> None:
    scope_id.set(uuid.uuid4())
    response = await logs_router.get_logs()
    assert isinstance(response.body_iterator, StreamIterator)

    waiting = asyncio.create_task(anext(response.body_iterator))
    for _ in range(5):
        await asyncio.sleep(0)
    InMemoryCacheService().publish(channel, "a captured line")

    async with asyncio.timeout(5):
        assert await waiting == "a captured line"


async def test_each_subscriber_gets_its_own_copy(
    chat_settings: Any, channel: str
) -> None:
    scope_id.set(uuid.uuid4())
    first = (await logs_router.get_logs()).body_iterator
    second = (await logs_router.get_logs()).body_iterator

    waiting = [asyncio.create_task(anext(it)) for it in (first, second)]
    for _ in range(5):
        await asyncio.sleep(0)
    InMemoryCacheService().publish(channel, "a captured line")

    async with asyncio.timeout(5):
        assert await asyncio.gather(*waiting) == [
            "a captured line",
            "a captured line",
        ]


async def test_the_channel_belongs_to_this_service_alone(
    chat_settings: Any, channel: str
) -> None:
    """Two services sharing a process must not read each other's output."""
    scope_id.set(uuid.uuid4())
    response = await logs_router.get_logs()
    waiting = asyncio.create_task(anext(response.body_iterator))
    for _ in range(5):
        await asyncio.sleep(0)

    InMemoryCacheService().publish(CacheChannels.LOGS.for_app("another"), "theirs")
    InMemoryCacheService().publish(channel, "mine")

    async with asyncio.timeout(5):
        assert await waiting == "mine"
