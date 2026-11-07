"""One cache fixture, two backends.

The point of the suite next door is that `InMemoryCacheService` and
`RedisCacheService` answer the same way, so both are driven through the same
fixture and the same assertions. The Redis parameter carries the `integration`
marker, so with no containers the in-process half still runs and reports.

Every composed key a test touches is recorded and deleted afterwards: these
tests share a real database with whatever else is using it.
"""

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest

from base.services.cache import AbstractCacheService
from base.services.common.meta import purge_all

BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param("redis", id="redis", marks=pytest.mark.integration),
]


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    """Drop every singleton around each test, pools included."""
    purge_all()
    yield
    purge_all()


def recorder(touched: set[str]) -> Callable[..., str]:
    """`_get_key`, remembering every key it composes so it can be deleted."""
    original = AbstractCacheService._get_key

    def recording_get_key(key: Any, namespace: UUID | str | None = None) -> str:
        composed = original(key, namespace)
        touched.add(composed)
        return composed

    return recording_get_key


@pytest.fixture(params=BACKENDS)
async def cache(request: pytest.FixtureRequest) -> AsyncIterator[AbstractCacheService]:
    """A cache service of the requested backend, cleaned up afterwards."""
    touched: set[str] = set()

    if request.param == "memory":
        from base.services.cache.memory import InMemoryCacheService

        request.getfixturevalue("make_settings")()
        service: AbstractCacheService = InMemoryCacheService()
    else:
        from base.services.cache.redis import RedisCacheService

        request.getfixturevalue("infra_settings")
        service = RedisCacheService()

    service._get_key = recorder(touched)  # type: ignore[method-assign]
    try:
        yield service
    finally:
        if request.param == "redis":
            from base.clients.redis_client import RedisClient

            client = RedisClient()
            if touched:
                await client.async_client.delete(*touched)
            await client.close_all()


@pytest.fixture
async def redis_cache(infra_settings: Any) -> AsyncIterator[Any]:
    """The Redis cache alone, for behaviour the in-process one cannot show."""
    del infra_settings
    from base.clients.redis_client import RedisClient
    from base.services.cache.redis import RedisCacheService

    touched: set[str] = set()
    service = RedisCacheService()
    service._get_key = recorder(touched)  # type: ignore[method-assign]
    try:
        yield service
    finally:
        client = RedisClient()
        if touched:
            await client.async_client.delete(*touched)
        await client.close_all()


@pytest.fixture
def namespace() -> UUID:
    """A namespace no other test shares."""
    return uuid4()


@pytest.fixture
def other_namespace() -> UUID:
    """A second namespace, for proving two do not collide."""
    return uuid4()
