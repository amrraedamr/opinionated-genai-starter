"""Which transport a service gets, and what happens when it asks for one that
does not exist.

The comment on `build_broker` is explicit that dispatch is exhaustive rather
than defaulting, because silently falling back to the in-process transport
starts cleanly, passes readiness and delivers nothing between containers.
What is pinned here is that it does fail — and with which error.
"""

import pytest
from pydantic import ValidationError

from base.models.definitions.broker import BrokerBackend
from base.services.broker.factory import build_broker
from base.services.broker.memory import InMemoryBroker
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    RedisSettingsMixin,
)
from tests.conftest import MakeSettings


class UnknownTransportConfig:
    """A settings object naming a transport the enum does not carry."""

    broker = "kafka"
    broker_max_attempts = 5
    broker_claim_idle_ms = 30_000
    broker_retry_backoff_seconds = 0.25


def test_the_memory_setting_selects_the_in_process_broker(
    make_settings: MakeSettings,
) -> None:
    make_settings(BrokerSettingsMixin, broker="memory")
    assert isinstance(build_broker(), InMemoryBroker)


def test_the_redis_setting_selects_the_streams_broker(
    make_settings: MakeSettings,
) -> None:
    from base.services.broker.redis_streams import RedisStreamsBroker

    make_settings(
        BrokerSettingsMixin,
        RedisSettingsMixin,
        broker="redis",
        redis_url="redis://localhost:6379/15",
    )
    assert isinstance(build_broker(), RedisStreamsBroker)


def test_the_default_setting_selects_the_in_process_broker(
    make_settings: MakeSettings,
) -> None:
    make_settings(BrokerSettingsMixin)
    assert isinstance(build_broker(), InMemoryBroker)


def test_a_config_passed_in_is_used_instead_of_the_registered_settings(
    make_settings: MakeSettings,
) -> None:
    make_settings(BrokerSettingsMixin, broker="memory")
    assert isinstance(
        build_broker(BrokerSettingsMixin(broker="memory")), InMemoryBroker
    )


def test_an_unknown_transport_names_the_transports_this_build_supports() -> None:
    with pytest.raises(
        ValueError, match="not a transport this library knows"
    ) as caught:
        build_broker(UnknownTransportConfig())  # type: ignore[arg-type]
    assert "This build supports" in str(caught.value)
    assert not isinstance(caught.value, ImportError)


def test_settings_refuse_an_unknown_transport_before_the_factory_sees_it(
    make_settings: MakeSettings,
) -> None:
    with pytest.raises(ValidationError):
        make_settings(BrokerSettingsMixin, broker="kafka")


def test_every_enum_member_has_a_transport(make_settings: MakeSettings) -> None:
    """No member may be added without a branch, which would fall through."""
    for backend in BrokerBackend:
        make_settings(
            BrokerSettingsMixin,
            RedisSettingsMixin,
            broker=backend,
            redis_url="redis://localhost:6379/15",
        )
        assert build_broker() is not None


def test_the_factory_hands_back_one_shared_instance(
    make_settings: MakeSettings,
) -> None:
    make_settings(BrokerSettingsMixin, broker="memory")
    assert build_broker() is build_broker()
