"""How the Redis client is configured, before anything connects.

Pool construction is lazy, so everything here is decided from settings without
a server: a service that forgot `REDIS_URL` should hear about it at startup
rather than on its first cache read.
"""

import pytest

from base.clients.redis_client import RedisClient
from base.services.configuration.mixins import RedisSettingsMixin
from tests.conftest import MakeSettings


def test_a_missing_redis_url_is_refused_at_construction(
    make_settings: MakeSettings,
) -> None:
    make_settings(RedisSettingsMixin, redis_url="")
    with pytest.raises(ValueError, match="redis_url must be configured"):
        RedisClient()


def test_settings_are_read_once_at_construction(make_settings: MakeSettings) -> None:
    make_settings(
        RedisSettingsMixin,
        redis_url="redis://localhost:6379/15",
        redis_max_connections=7,
        redis_socket_timeout=1.5,
        redis_decode_responses=False,
    )
    client = RedisClient()
    assert client._url == "redis://localhost:6379/15"
    assert client._max_connections == 7
    assert client._get_common_kwargs() == {
        "socket_timeout": 1.5,
        "socket_connect_timeout": 5.0,
        "retry_on_timeout": True,
        "health_check_interval": 30,
        "decode_responses": False,
    }


def test_a_blank_cert_requirement_is_left_out_of_the_connection(
    make_settings: MakeSettings,
) -> None:
    """redis-py rejects `ssl_cert_reqs=""`, so the key must be absent."""
    make_settings(
        RedisSettingsMixin,
        redis_url="redis://localhost:6379/15",
        redis_ssl_cert_reqs="",
    )
    assert "ssl_cert_reqs" not in RedisClient()._get_common_kwargs()


def test_a_cert_requirement_is_passed_through(make_settings: MakeSettings) -> None:
    make_settings(
        RedisSettingsMixin,
        redis_url="rediss://localhost:6379/15",
        redis_ssl_cert_reqs="required",
    )
    assert RedisClient()._get_common_kwargs()["ssl_cert_reqs"] == "required"


def test_construction_opens_no_pools(make_settings: MakeSettings) -> None:
    make_settings(RedisSettingsMixin, redis_url="redis://localhost:6379/15")
    client = RedisClient()
    assert client._sync_pool is None
    assert client._async_pool is None


def test_one_client_is_shared_across_the_process(make_settings: MakeSettings) -> None:
    make_settings(RedisSettingsMixin, redis_url="redis://localhost:6379/15")
    assert RedisClient() is RedisClient()


def test_closing_an_unopened_client_is_accepted(make_settings: MakeSettings) -> None:
    make_settings(RedisSettingsMixin, redis_url="redis://localhost:6379/15")
    client = RedisClient()
    client.close()
    assert client._sync_client is None
