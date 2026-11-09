"""A real Qdrant, the one the repository's compose file starts.

The client prefers gRPC, which is stricter about point ids than the HTTP API,
so these tests run against the transport the repository actually uses. Each
test names its collection uniquely and deletes it afterwards, so reruns never
collide.
"""

import uuid
from collections.abc import AsyncIterator, Iterator

import pytest

from base.services.common.meta import Singleton
from base.services.vector_search.qdrant import QdrantVectorSearchService
from tests.conftest import CoreSettings


@pytest.fixture(autouse=True)
def _purge_singletons() -> Iterator[None]:
    """Drop every cached singleton around each test."""
    Singleton.purge()
    yield
    Singleton.purge()


@pytest.fixture
async def qdrant(
    infra_settings: CoreSettings,
) -> AsyncIterator[QdrantVectorSearchService]:
    """A service pointed at the local container, closed afterwards."""
    service = QdrantVectorSearchService()
    yield service
    await service.adispose()
    service.dispose()


@pytest.fixture
async def collection(
    qdrant: QdrantVectorSearchService, unique: str
) -> AsyncIterator[str]:
    """An empty three-dimensional collection, deleted when the test ends."""
    name = f"t_{unique}"
    await qdrant.acreate_collection(name, vector_size=3)
    yield name
    if await qdrant.acollection_exists(name):
        await qdrant.adelete_collection(name)


@pytest.fixture
def point_id() -> str:
    """A point id Qdrant accepts: a UUID."""
    return str(uuid.uuid4())
