"""How `settings_for` decides a service composed what a subsystem reads.

Pins the structural (not inherited) satisfaction that makes mixins work, the
error raised when a contract goes unmet, and the per-(settings, contract)
cache that both `configure` and `reset` have to clear or a stale answer
outlives the settings it was about.
"""

from typing import Protocol, runtime_checkable

import pytest

from base.services.configuration import (
    _checked,
    configure,
    reset,
    settings_for,
)
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    RedisSettingsMixin,
    SQLSettingsMixin,
)
from base.services.configuration.protocols import (
    SupportsBrokerConfig,
    SupportsRedisBrokerConfig,
    SupportsRedisConfig,
    SupportsSQLConfig,
)
from base.tools.errors import MissingSettingsError
from tests.conftest import MakeSettings, settings_class


@runtime_checkable
class SupportsNothingAnyoneDeclares(Protocol):
    """A contract naming a field no mixin in this repository provides."""

    field_no_mixin_declares: str


def test_settings_satisfy_a_contract_structurally_not_by_inheritance(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    assert isinstance(settings, SupportsRedisConfig)
    assert SupportsRedisConfig not in type(settings).__mro__


def test_a_data_protocol_cannot_be_checked_with_issubclass(
    make_settings: MakeSettings,
) -> None:
    """Why the seam uses `isinstance`: `issubclass` is unavailable here."""
    settings = make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    with pytest.raises(TypeError, match="non-method members"):
        issubclass(type(settings), SupportsRedisConfig)


def test_a_contract_naming_a_field_the_settings_lack_is_refused(
    make_settings: MakeSettings,
) -> None:
    make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    with pytest.raises(MissingSettingsError):
        settings_for(SupportsNothingAnyoneDeclares)


def test_the_error_names_both_the_settings_class_and_the_contract(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    with pytest.raises(MissingSettingsError) as caught:
        settings_for(SupportsSQLConfig)
    assert caught.value.settings_class == type(settings).__name__
    assert caught.value.contract == "SupportsSQLConfig"
    assert "Compose the settings mixin" in str(caught.value)


def test_one_settings_object_answers_two_contracts_independently(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(
        RedisSettingsMixin,
        SQLSettingsMixin,
        redis_url="redis://x:6379/0",
        sql_backend="postgresql",
    )
    assert settings_for(SupportsRedisConfig) is settings
    assert settings_for(SupportsSQLConfig) is settings


def test_a_composed_contract_needs_every_mixin_it_unions(
    make_settings: MakeSettings,
) -> None:
    """`SupportsRedisBrokerConfig` is the connection plus the delivery policy."""
    make_settings(BrokerSettingsMixin, broker="memory")
    with pytest.raises(MissingSettingsError) as caught:
        settings_for(SupportsRedisBrokerConfig)
    assert caught.value.contract == "SupportsRedisBrokerConfig"

    reset()
    both = configure(
        settings_class(RedisSettingsMixin, BrokerSettingsMixin)(
            service_name="svc", redis_url="redis://x:6379/0", broker="redis"
        )
    )
    assert settings_for(SupportsRedisBrokerConfig) is both


def test_a_contract_that_is_not_a_class_is_passed_through_unchecked(
    make_settings: MakeSettings,
) -> None:
    """The `isinstance(contract, type)` guard: a union is not checkable."""
    settings = make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    assert settings_for(str | int) is settings


def test_reading_before_anything_is_configured_says_so() -> None:
    with pytest.raises(RuntimeError, match="Settings not configured"):
        settings_for(SupportsRedisConfig)


def test_the_check_is_cached_per_settings_and_contract(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(
        RedisSettingsMixin,
        SQLSettingsMixin,
        redis_url="redis://x:6379/0",
    )
    assert _checked == set()

    settings_for(SupportsRedisConfig)
    assert _checked == {(id(settings), id(SupportsRedisConfig))}

    settings_for(SupportsRedisConfig)
    assert len(_checked) == 1

    settings_for(SupportsSQLConfig)
    assert _checked == {
        (id(settings), id(SupportsRedisConfig)),
        (id(settings), id(SupportsSQLConfig)),
    }


def test_a_failed_check_is_not_cached(make_settings: MakeSettings) -> None:
    make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    for _ in range(2):
        with pytest.raises(MissingSettingsError):
            settings_for(SupportsSQLConfig)
    assert _checked == set()


def test_configure_clears_the_cache(make_settings: MakeSettings) -> None:
    settings = make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    settings_for(SupportsRedisConfig)
    assert _checked

    configure(settings)
    assert _checked == set()


def test_reset_clears_the_cache(make_settings: MakeSettings) -> None:
    make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    settings_for(SupportsRedisConfig)
    assert _checked

    reset()
    assert _checked == set()


def test_a_contract_unmet_by_the_newly_configured_settings_is_rechecked(
    make_settings: MakeSettings,
) -> None:
    """A cached pass must not survive the settings it was about."""
    make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    settings_for(SupportsRedisConfig)

    configure(settings_class(SQLSettingsMixin)(service_name="other"))
    with pytest.raises(MissingSettingsError):
        settings_for(SupportsRedisConfig)


def test_a_contract_reads_the_fields_it_declares(make_settings: MakeSettings) -> None:
    make_settings(
        BrokerSettingsMixin,
        broker="redis",
        broker_max_attempts=9,
        broker_claim_idle_ms=1234,
        broker_retry_backoff_seconds=0.5,
    )
    config = settings_for(SupportsBrokerConfig)
    assert config.broker == "redis"
    assert config.broker_max_attempts == 9
    assert config.broker_claim_idle_ms == 1234
    assert config.broker_retry_backoff_seconds == 0.5
