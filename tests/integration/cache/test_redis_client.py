"""The Redis client against the real server.

`ping` and `aping` are what a readiness probe calls, so what they answer when
the server is there, and when the client is pointed somewhere it is not, is
the whole of their contract.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from base.clients.redis_client import RedisClient
from base.services.common.meta import purge_all

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    purge_all()
    yield
    purge_all()


async def test_the_server_answers_an_async_ping(infra_settings: Any) -> None:
    del infra_settings
    client = RedisClient()
    try:
        assert await client.aping() is True
    finally:
        await client.close_all()


def test_the_server_answers_a_sync_ping(infra_settings: Any) -> None:
    del infra_settings
    client = RedisClient()
    try:
        assert client.ping() is True
    finally:
        client.close()


async def test_both_clients_see_the_same_data(infra_settings: Any, unique: str) -> None:
    del infra_settings
    client = RedisClient()
    key = f"test-client-{unique}"
    try:
        client.sync_client.set(key, "written-sync")
        assert await client.async_client.get(key) == "written-sync"
    finally:
        await client.async_client.delete(key)
        await client.close_all()


async def test_a_ping_to_a_closed_port_is_false(make_settings: Any) -> None:
    """A health probe reports down rather than raising the client's error."""
    from base.services.configuration.mixins import RedisSettingsMixin

    make_settings(RedisSettingsMixin, redis_url="redis://localhost:6399/0")
    client = RedisClient()
    try:
        assert await client.aping() is False
        assert client.ping() is False
    finally:
        await client.close_all()


async def test_closing_drops_the_pools(infra_settings: Any) -> None:
    del infra_settings
    client = RedisClient()
    await client.aping()
    client.ping()
    await client.close_all()
    assert client._async_pool is None
    assert client._sync_pool is None


async def test_a_closed_client_reconnects_on_the_next_use(infra_settings: Any) -> None:
    del infra_settings
    client = RedisClient()
    await client.aping()
    await client.close_all()
    try:
        assert await client.aping() is True
    finally:
        await client.close_all()
