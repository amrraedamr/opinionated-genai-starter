from base.services.cache import AbstractCacheService
from base.services.cache.memory import InMemoryCacheService
from base.services.configuration import get_settings


def get_cache_service() -> AbstractCacheService:
    """Redis when a URL is configured, otherwise in-process.

    Anything relying on pub/sub to reach other replicas must resolve its cache
    through here: an in-process channel only ever reaches itself.
    """
    if not getattr(get_settings(), "redis_url", ""):
        return InMemoryCacheService()

    from base.services.cache.redis import RedisCacheService

    return RedisCacheService()
