"""Fixtures shared by the in-process cache tests.

Every cache service in `base` is a singleton, so one built in an earlier test
would answer in this one with the previous test's store and a connection pool
bound to a closed event loop. Each test here gets a fresh one.
"""

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest

from base.services.cache.memory import InMemoryCacheService
from base.services.common.meta import purge_all
from tests.conftest import MakeSettings


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    """Drop every singleton around each test."""
    purge_all()
    yield
    purge_all()


@pytest.fixture
def memory_cache(make_settings: MakeSettings) -> InMemoryCacheService:
    """An in-process cache with no Redis configured."""
    from base.services.configuration.mixins import RedisSettingsMixin

    make_settings(RedisSettingsMixin, redis_url="")
    return InMemoryCacheService()


@pytest.fixture
def namespace() -> UUID:
    """A namespace no other test shares."""
    return uuid4()
