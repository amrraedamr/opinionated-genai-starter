"""Fixtures for the Redis Streams broker against the real server.

Every test gets its own topic and consumer group, named from `unique`, so a
rerun never reads a previous run's backlog. The harness remembers what it
named and deletes the streams afterwards, groups included: these run against
a shared database.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest

from base.services.common.meta import Singleton, purge_all
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    RedisSettingsMixin,
)

REDIS_URL = "redis://localhost:6379/1"


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    purge_all()
    yield
    purge_all()


async def await_until(predicate: Callable[[], bool], *, timeout: float = 10.0) -> bool:
    """Poll until the predicate holds, so no test waits a fixed time."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


class StreamsHarness:
    """Builds brokers, names streams, and cleans up what it named."""

    def __init__(self, unique: str, make_settings: Any) -> None:
        self._unique = unique
        self._make_settings = make_settings
        self._configured = False
        self.brokers: list[Any] = []
        self.topics: set[str] = set()

    def configure(
        self,
        *,
        max_attempts: int = 5,
        claim_idle_ms: int = 60_000,
        retry_backoff: float = 0.01,
    ) -> None:
        """Register the delivery policy the brokers built next will read."""
        self._make_settings(
            RedisSettingsMixin,
            BrokerSettingsMixin,
            redis_url=REDIS_URL,
            broker="redis",
            broker_max_attempts=max_attempts,
            broker_claim_idle_ms=claim_idle_ms,
            broker_retry_backoff_seconds=retry_backoff,
        )
        self._configured = True

    def topic(self, name: str = "events") -> str:
        composed = f"test.{name}.{self._unique}"
        self.topics.add(composed)
        return composed

    def track(self, topic: str) -> str:
        """Name a stream the harness did not compose, so it is cleaned up too."""
        self.topics.add(topic)
        return topic

    def group(self, name: str = "workers") -> str:
        return f"test-group-{name}-{self._unique}"

    def new(self) -> Any:
        """A broker that is not the one an earlier call cached."""
        from base.services.broker.redis_streams import RedisStreamsBroker

        if not self._configured:
            self.configure()
        Singleton._instances.pop(RedisStreamsBroker, None)
        broker = RedisStreamsBroker()
        self.brokers.append(broker)
        return broker


@pytest.fixture
async def streams(make_settings: Any, unique: str) -> AsyncIterator[StreamsHarness]:
    harness = StreamsHarness(unique, make_settings)
    try:
        yield harness
    finally:
        for broker in harness.brokers:
            await broker.astop()
        from redis.asyncio import Redis

        cleanup = Redis.from_url(REDIS_URL, decode_responses=True)
        names = {name for topic in harness.topics for name in (topic, f"{topic}.dlq")}
        # A test may name a stream the harness never saw, so sweep by token too.
        names.update([key async for key in cleanup.scan_iter(match=f"*{unique}*")])
        if names:
            await cleanup.delete(*names)
        await cleanup.aclose()
