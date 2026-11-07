"""What the Redis Streams broker decides before it connects.

Construction reads settings, names a consumer and records subscriptions
without opening a socket, so all of that is pinned here. Delivery itself needs
a server and lives in `tests/integration/broker`.
"""

from typing import Any

import pytest

from base.services.common.meta import Singleton
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    RedisSettingsMixin,
)
from contracts.events import Event


@pytest.fixture
def broker_settings(make_settings: Any) -> Any:
    return make_settings(
        BrokerSettingsMixin,
        RedisSettingsMixin,
        broker="redis",
        redis_url="redis://localhost:6379/15",
        broker_max_attempts=7,
        broker_claim_idle_ms=1234,
        broker_retry_backoff_seconds=0.5,
    )


def build() -> Any:
    """A broker that is not the one an earlier call cached."""
    from base.services.broker.redis_streams import RedisStreamsBroker

    Singleton._instances.pop(RedisStreamsBroker, None)
    return RedisStreamsBroker()


class UnreachableRedis:
    """A client whose every call fails, as a down server's would."""

    def __init__(self) -> None:
        self.pings = 0

    async def ping(self) -> bool:
        self.pings += 1
        raise ConnectionError("connection refused")


def test_the_delivery_policy_comes_from_settings(broker_settings: Any) -> None:
    del broker_settings
    broker = build()
    assert broker._max_attempts == 7
    assert broker._claim_idle_ms == 1234
    assert broker._retry_backoff == 0.5


def test_a_config_passed_in_is_used_instead_of_the_registered_settings(
    broker_settings: Any,
) -> None:
    del broker_settings
    from base.services.broker.redis_streams import RedisStreamsBroker

    Singleton._instances.pop(RedisStreamsBroker, None)
    override = BrokerSettingsMixin(broker="redis", broker_max_attempts=2)
    assert RedisStreamsBroker(override)._max_attempts == 2  # type: ignore[arg-type]


def test_the_consumer_name_starts_with_the_service_name(broker_settings: Any) -> None:
    del broker_settings
    assert build()._consumer.startswith("test-service-")


def test_two_brokers_in_one_service_take_different_consumer_names(
    broker_settings: Any,
) -> None:
    """Two replicas sharing a name would split a group's messages wrongly."""
    del broker_settings
    first = build()
    second = build()
    assert first._consumer != second._consumer


def test_construction_opens_no_connection(broker_settings: Any) -> None:
    del broker_settings
    assert build()._redis is None


async def test_subscribing_opens_no_connection(broker_settings: Any) -> None:
    del broker_settings
    broker = build()

    async def handler(event: Event) -> None:
        return None

    await broker.asubscribe("t", "g", handler)
    assert broker._redis is None
    assert broker._subscriptions == [("t", "g", handler)]


async def test_subscriptions_are_kept_in_the_order_they_were_made(
    broker_settings: Any,
) -> None:
    del broker_settings
    broker = build()

    async def handler(event: Event) -> None:
        return None

    await broker.asubscribe("first", "g", handler)
    await broker.asubscribe("second", "g", handler)
    assert [topic for topic, _, _ in broker._subscriptions] == ["first", "second"]


async def test_starting_with_no_subscriptions_starts_no_tasks(
    broker_settings: Any,
) -> None:
    del broker_settings
    broker = build()
    await broker.astart()
    assert broker._tasks == []
    await broker.astop()


async def test_stopping_without_starting_is_accepted(broker_settings: Any) -> None:
    del broker_settings
    broker = build()
    await broker.astop()
    assert broker._tasks == []


async def test_an_unreachable_server_reports_unhealthy(broker_settings: Any) -> None:
    del broker_settings
    broker = build()
    client = UnreachableRedis()
    broker._redis = client  # type: ignore[assignment]
    assert await broker.ahealth_check() is False
    assert client.pings == 1


async def test_a_health_check_is_not_retried(broker_settings: Any) -> None:
    """One probe per call: a readiness endpoint must answer promptly."""
    del broker_settings
    broker = build()
    client = UnreachableRedis()
    broker._redis = client  # type: ignore[assignment]
    await broker.ahealth_check()
    await broker.ahealth_check()
    assert client.pings == 2


def test_the_url_comes_from_the_redis_settings(broker_settings: Any) -> None:
    del broker_settings
    assert build()._config.redis_url == "redis://localhost:6379/15"
