"""`settings_for` is the seam every subsystem reads its configuration through."""

import pytest

from base.services.configuration import settings_for
from base.services.configuration.mixins import RedisSettingsMixin, SQLSettingsMixin
from base.services.configuration.protocols import (
    SupportsRedisConfig,
    SupportsSQLConfig,
)
from base.tools.errors import MissingSettingsError


def test_reads_a_contract_the_service_composed(make_settings):
    make_settings(RedisSettingsMixin, redis_url="redis://x:1/0")
    assert settings_for(SupportsRedisConfig).redis_url == "redis://x:1/0"


def test_names_the_contract_a_service_failed_to_compose(make_settings):
    make_settings(RedisSettingsMixin, redis_url="redis://x:1/0")
    with pytest.raises(MissingSettingsError) as caught:
        settings_for(SupportsSQLConfig).sql_backend
    assert "SupportsSQLConfig" in str(caught.value)


def test_unconfigured_settings_say_so():
    from base.services.configuration import reset

    reset()
    with pytest.raises(RuntimeError, match="Settings not configured"):
        settings_for(SupportsRedisConfig)


def test_one_settings_object_serves_several_contracts(make_settings):
    make_settings(
        RedisSettingsMixin,
        SQLSettingsMixin,
        redis_url="redis://x:1/0",
        sql_backend="sqlite",
    )
    assert settings_for(SupportsRedisConfig).redis_url
    assert settings_for(SupportsSQLConfig).sql_backend == "sqlite"
