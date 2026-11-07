"""Which cache implementation the factory hands back.

The choice is made from `redis_url` alone, and getting it wrong is quiet:
pub/sub on an in-process cache reaches only the replica that published.
"""

from typing import Any

from base.services.cache.factory import get_cache_service
from base.services.cache.memory import InMemoryCacheService
from base.services.configuration.mixins import RedisSettingsMixin


def test_an_empty_redis_url_selects_the_in_process_cache(make_settings: Any) -> None:
    make_settings(RedisSettingsMixin, redis_url="")
    assert isinstance(get_cache_service(), InMemoryCacheService)


def test_settings_without_a_redis_url_at_all_select_the_in_process_cache(
    make_settings: Any,
) -> None:
    make_settings()
    assert isinstance(get_cache_service(), InMemoryCacheService)


def test_a_configured_redis_url_selects_the_redis_cache(make_settings: Any) -> None:
    from base.services.cache.redis import RedisCacheService

    make_settings(RedisSettingsMixin, redis_url="redis://localhost:6379/15")
    assert isinstance(get_cache_service(), RedisCacheService)


def test_the_factory_hands_back_one_shared_instance(make_settings: Any) -> None:
    make_settings(RedisSettingsMixin, redis_url="")
    assert get_cache_service() is get_cache_service()
