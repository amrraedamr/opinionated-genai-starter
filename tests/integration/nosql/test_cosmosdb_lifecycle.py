"""The client's own surface against the emulator: accessors, health, closing.

`CosmosDBClient` is the layer with no error handling on it, so what it raises
is the Azure SDK's own exception and what it hands back is a proxy that has
not checked anything exists yet. Both are pinned here, along with the part
that is easy to get silently wrong: the sync and async clients are separate
resources, and closing one must leave the other alone.
"""

from collections.abc import AsyncIterator

import pytest
from azure.cosmos import PartitionKey, exceptions

from base.clients.nosql.cosmosdb import CosmosDBClient
from base.services.configuration.mixins import CosmosSettingsMixin
from base.services.nosql.cosmosdb import CosmosDBService
from tests.conftest import MakeSettings

from . import test_cosmosdb_crud

database = test_cosmosdb_crud.database
cosmos = test_cosmosdb_crud.cosmos
container = test_cosmosdb_crud.container

pytestmark = pytest.mark.integration


@pytest.fixture
def client(cosmos: CosmosDBService) -> CosmosDBClient:
    """The one client the service built, closed by the service fixture."""
    return CosmosDBClient()


@pytest.fixture
async def unreachable_client(
    make_settings: MakeSettings,
) -> AsyncIterator[CosmosDBClient]:
    """A client pointed at a port nothing is listening on."""
    make_settings(
        CosmosSettingsMixin,
        azure_cosmos_account_uri="http://127.0.0.1:9/",
        azure_cosmos_account_key=test_cosmosdb_crud.ACCOUNT_KEY,
        azure_cosmos_database_name="unreachable",
    )
    instance = CosmosDBClient()
    yield instance
    await instance.close_all()


# Credentials


def test_an_account_key_is_enough_to_reach_the_account(
    client: CosmosDBClient,
) -> None:
    """No identity is built, which is the point of the key branch."""
    assert client.ping() is True
    assert client._credential is None


# Databases and containers


def test_a_database_created_through_the_client_appears_in_its_listing(
    client: CosmosDBClient, unique: str
) -> None:
    name = f"db_{unique}"
    client.sync_client.create_database(name)
    try:
        assert name in [d["id"] for d in client.sync_client.list_databases()]
    finally:
        client.sync_client.delete_database(name)


def test_creating_a_database_twice_raises_the_sdks_own_exception(
    client: CosmosDBClient, unique: str
) -> None:
    """Nothing wraps the client layer, so a caller here sees Azure's error."""
    name = f"db_{unique}"
    client.sync_client.create_database(name)
    try:
        with pytest.raises(exceptions.CosmosResourceExistsError):
            client.sync_client.create_database(name)
    finally:
        client.sync_client.delete_database(name)


def test_a_database_accessor_does_not_check_that_the_database_exists(
    client: CosmosDBClient,
) -> None:
    assert client.get_database("never_created") is not None


def test_a_container_accessor_does_not_check_that_the_container_exists(
    client: CosmosDBClient, database: str
) -> None:
    assert client.get_container(database, "never_created") is not None


def test_a_container_created_through_the_client_can_be_written_to(
    client: CosmosDBClient, database: str, unique: str
) -> None:
    name = f"made_{unique}"
    proxy = client.get_database(database)
    proxy.create_container(name, PartitionKey(path="/id"))
    try:
        client.get_container(database, name).upsert_item({"id": "a", "n": 1})
        read = client.get_container(database, name).read_item("a", partition_key="a")
        assert read["n"] == 1
    finally:
        proxy.delete_container(name)


async def test_the_async_accessors_reach_what_the_sync_ones_wrote(
    client: CosmosDBClient, database: str, container: str
) -> None:
    client.get_container(database, container).upsert_item({"id": "a", "n": 1})
    read = await client.aget_container(database, container).read_item(
        "a", partition_key="a"
    )
    assert read["n"] == 1


async def test_deleting_a_database_takes_its_containers_with_it(
    client: CosmosDBClient, unique: str
) -> None:
    name = f"db_{unique}"
    proxy = client.sync_client.create_database(name)
    proxy.create_container("c", PartitionKey(path="/id"))
    client.sync_client.delete_database(name)
    with pytest.raises(exceptions.CosmosResourceNotFoundError):
        client.get_container(name, "c").read_item("a", partition_key="a")


# Health


def test_ping_is_true_against_a_running_emulator(client: CosmosDBClient) -> None:
    assert client.ping() is True


async def test_aping_is_true_against_a_running_emulator(
    client: CosmosDBClient,
) -> None:
    assert await client.aping() is True


def test_ping_is_false_when_nothing_is_listening(
    unreachable_client: CosmosDBClient,
) -> None:
    assert unreachable_client.ping() is False


async def test_aping_is_false_when_nothing_is_listening(
    unreachable_client: CosmosDBClient,
) -> None:
    assert await unreachable_client.aping() is False


def test_ping_reports_rather_than_raises(unreachable_client: CosmosDBClient) -> None:
    """A health probe answers a question; it does not become the failure."""
    assert unreachable_client.ping() is False
    assert unreachable_client.ping() is False


# Closing


def test_closing_the_sync_client_releases_its_transport(
    client: CosmosDBClient, database: str, container: str
) -> None:
    """A dropped reference is not a closed session: the proxy keeps the pool."""
    proxy = client.get_container(database, container)
    proxy.upsert_item({"id": "a"})
    client.close()
    with pytest.raises(ValueError, match="transport has already been closed"):
        proxy.upsert_item({"id": "b"})


async def test_closing_the_async_client_releases_its_transport(
    client: CosmosDBClient, database: str, container: str
) -> None:
    proxy = client.aget_container(database, container)
    await proxy.upsert_item({"id": "a"})
    await client.aclose()
    with pytest.raises(ValueError, match="transport has already been closed"):
        await proxy.upsert_item({"id": "b"})


def test_the_next_use_after_closing_builds_a_client_that_works(
    client: CosmosDBClient,
) -> None:
    client.close()
    assert client.ping() is True


async def test_closing_the_sync_client_leaves_the_async_one_working(
    client: CosmosDBClient,
) -> None:
    assert client.ping() is True
    assert await client.aping() is True
    client.close()
    assert await client.aping() is True


async def test_closing_the_async_client_leaves_the_sync_one_working(
    client: CosmosDBClient, database: str, container: str
) -> None:
    proxy = client.get_container(database, container)
    assert await client.aping() is True
    await client.aclose()
    proxy.upsert_item({"id": "a"})
    assert client.ping() is True


async def test_close_all_closes_both(
    client: CosmosDBClient, database: str, container: str
) -> None:
    sync_proxy = client.get_container(database, container)
    async_proxy = client.aget_container(database, container)
    sync_proxy.upsert_item({"id": "a"})
    await async_proxy.upsert_item({"id": "b"})
    await client.close_all()
    with pytest.raises(ValueError):
        sync_proxy.upsert_item({"id": "c"})
    with pytest.raises(ValueError):
        await async_proxy.upsert_item({"id": "d"})


def test_closing_a_client_that_was_never_used_is_harmless(
    client: CosmosDBClient,
) -> None:
    client.close()
    client.close()


async def test_closing_twice_is_harmless(client: CosmosDBClient) -> None:
    assert await client.aping() is True
    await client.aclose()
    await client.aclose()


def test_disposing_the_service_closes_the_sync_client(
    cosmos: CosmosDBService, client: CosmosDBClient, database: str, container: str
) -> None:
    proxy = client.get_container(database, container)
    proxy.upsert_item({"id": "a"})
    cosmos.dispose()
    with pytest.raises(ValueError, match="transport has already been closed"):
        proxy.upsert_item({"id": "b"})


async def test_adisposing_the_service_closes_the_async_client(
    cosmos: CosmosDBService, client: CosmosDBClient, database: str, container: str
) -> None:
    proxy = client.aget_container(database, container)
    await proxy.upsert_item({"id": "a"})
    await cosmos.adispose()
    with pytest.raises(ValueError, match="transport has already been closed"):
        await proxy.upsert_item({"id": "b"})


async def test_a_disposed_service_works_again_on_the_next_call(
    cosmos: CosmosDBService, container: str
) -> None:
    """The clients are rebuilt lazily, so disposing is not the end of it."""
    await cosmos.ainsert_one(container, {"id": "a"})
    await cosmos.adispose()
    cosmos.dispose()
    assert await cosmos.acount(container) == 1


def test_the_client_is_one_instance_for_the_process(client: CosmosDBClient) -> None:
    assert CosmosDBClient() is client
