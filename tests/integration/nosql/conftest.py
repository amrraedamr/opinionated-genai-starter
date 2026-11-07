"""A real MongoDB, the one the repository's compose file starts.

That server runs with authentication off, which is exactly the configuration
the connection-string bug used to break, so these tests are also the proof
that the unauthenticated path reaches a server at all. Every test names its
collection uniquely and drops it afterwards, so reruns never collide.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from base.services.common.meta import Singleton
from base.services.nosql.mongodb import MongoDBService


@pytest.fixture(autouse=True)
def _purge_singletons() -> Iterator[None]:
    """Drop every cached singleton around each test."""
    Singleton.purge()
    yield
    Singleton.purge()


@pytest.fixture
async def mongo(infra_settings: Any) -> AsyncIterator[MongoDBService]:
    """A service pointed at the local container, closed afterwards."""
    service = MongoDBService()
    yield service
    await service.adispose()
    service.dispose()


@pytest.fixture
async def collection(mongo: MongoDBService, unique: str) -> AsyncIterator[str]:
    """A collection name nothing else uses, dropped when the test ends."""
    name = f"t_{unique}"
    yield name
    await mongo._aget_database()[name].drop()
