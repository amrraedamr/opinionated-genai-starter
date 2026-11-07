"""The Atlas vector index definition, and the three paths that install it.

The width of the vector is the caller's argument, not a setting: it belongs to
whichever embedding model produced the vectors. These tests read the index
definition the service hands Atlas, and walk create, update-in-place, and the
drop-then-recreate fallback that an incompatible reshape forces.
"""

from typing import Any

import pytest
from pymongo.errors import OperationFailure

from base.models.view_models.exceptions.nosql import NoSqlBadRequestError
from base.services.configuration.mixins import MongoSettingsMixin
from base.services.nosql.mongodb import MongoDBService
from tests.unit.nosql.mongo_doubles import (
    AsyncFakeCollection,
    AsyncFakeDatabase,
    FakeCollection,
    FakeDatabase,
)


@pytest.fixture
def service(make_settings: Any) -> MongoDBService:
    make_settings(MongoSettingsMixin, mongodb_host="db:27017", mongodb_database="app")
    return MongoDBService()


@pytest.fixture
def database(service: MongoDBService, monkeypatch: pytest.MonkeyPatch) -> FakeDatabase:
    fake = FakeDatabase()
    monkeypatch.setattr(service, "_get_database", lambda: fake)
    return fake


@pytest.fixture
def adatabase(
    service: MongoDBService, monkeypatch: pytest.MonkeyPatch
) -> AsyncFakeDatabase:
    fake = AsyncFakeDatabase()
    monkeypatch.setattr(service, "_aget_database", lambda: fake)
    return fake


def created_definition(collection: FakeCollection) -> dict[str, Any]:
    """The definition handed to `create_search_index`."""
    (model,), _ = collection.call("create_search_index")
    return dict(model.document["definition"])


def test_the_definition_carries_the_width_the_caller_named(
    service: MongoDBService, database: FakeDatabase
) -> None:
    service.create_collection_vector_index("chunks", dimensions=1536)
    assert created_definition(database["chunks"])["fields"][0]["numDimensions"] == 1536


def test_a_different_width_produces_a_different_definition(
    service: MongoDBService, database: FakeDatabase
) -> None:
    service.create_collection_vector_index("chunks", dimensions=384)
    assert created_definition(database["chunks"])["fields"][0]["numDimensions"] == 384


def test_the_vector_field_is_declared_as_cosine_over_the_embedding_path(
    service: MongoDBService, database: FakeDatabase
) -> None:
    service.create_collection_vector_index("chunks", dimensions=8)
    field = created_definition(database["chunks"])["fields"][0]
    assert field["type"] == "vector"
    assert field["path"] == "embedding"
    assert field["similarity"] == "cosine"


def test_the_index_is_named_after_its_collection(
    service: MongoDBService, database: FakeDatabase
) -> None:
    service.create_collection_vector_index("chunks", dimensions=8)
    (model,), _ = database["chunks"].call("create_search_index")
    assert model.document["name"] == "chunks_vector_index"
    assert model.document["type"] == "vectorSearch"


def test_filter_fields_are_appended_after_the_vector_field(
    service: MongoDBService, database: FakeDatabase
) -> None:
    service.create_collection_vector_index(
        "chunks",
        dimensions=8,
        filter_fields=[
            {"type": "filter", "path": "tenant_id"},
            {"type": "filter", "path": "document_id"},
        ],
    )
    fields = created_definition(database["chunks"])["fields"]
    assert [f.get("path") for f in fields] == ["embedding", "tenant_id", "document_id"]


def test_no_filter_fields_leaves_the_vector_field_alone(
    service: MongoDBService, database: FakeDatabase
) -> None:
    service.create_collection_vector_index("chunks", dimensions=8)
    assert len(created_definition(database["chunks"])["fields"]) == 1


def test_an_existing_index_is_updated_in_place(
    service: MongoDBService, database: FakeDatabase
) -> None:
    collection = database["chunks"]
    collection.search_indexes.append("chunks_vector_index")
    service.create_collection_vector_index("chunks", dimensions=8)
    assert collection.called("update_search_index") == 1
    assert collection.called("create_search_index") == 0


def test_the_update_carries_the_new_width(
    service: MongoDBService, database: FakeDatabase
) -> None:
    collection = database["chunks"]
    collection.search_indexes.append("chunks_vector_index")
    service.create_collection_vector_index("chunks", dimensions=3072)
    (name, definition), _ = collection.call("update_search_index")
    assert name == "chunks_vector_index"
    assert definition["fields"][0]["numDimensions"] == 3072


def test_an_incompatible_reshape_drops_the_index_and_recreates_it(
    service: MongoDBService, database: FakeDatabase
) -> None:
    collection = database["chunks"]
    collection.search_indexes.append("chunks_vector_index")
    collection.update_error = OperationFailure("cannot reshape in place")
    service.create_collection_vector_index("chunks", dimensions=8)
    assert [
        name for name, _, _ in collection.calls if name.endswith("search_index")
    ] == [
        "update_search_index",
        "drop_search_index",
        "create_search_index",
    ]


def test_a_drop_that_never_lands_is_reported_rather_than_waited_on_forever(
    service: MongoDBService,
    database: FakeDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MongoDBService, "INDEX_DROP_WAIT_INTERVAL_S", 0.0)
    monkeypatch.setattr(MongoDBService, "INDEX_DROP_WAIT_TIMEOUT_S", 0.05)
    collection = database["chunks"]
    collection.search_indexes.append("chunks_vector_index")
    collection.update_error = OperationFailure("cannot reshape in place")
    collection.drop_never_completes = True
    with pytest.raises(NoSqlBadRequestError, match="Timed out waiting"):
        service.create_collection_vector_index("chunks", dimensions=8)


def test_a_listing_the_server_refuses_is_read_as_no_index(
    service: MongoDBService, database: FakeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atlas answers `list_search_indexes` with a failure on a plain server."""
    collection = database["chunks"]

    def refuse(name: str | None = None, **kwargs: Any) -> list[Any]:
        raise OperationFailure("search indexes are not supported")

    monkeypatch.setattr(collection, "list_search_indexes", refuse)
    service.create_collection_vector_index("chunks", dimensions=8)
    assert collection.called("create_search_index") == 1


async def test_the_async_definition_carries_the_width_the_caller_named(
    service: MongoDBService, adatabase: AsyncFakeDatabase
) -> None:
    await service.acreate_collection_vector_index("chunks", dimensions=1536)
    assert created_definition(adatabase["chunks"])["fields"][0]["numDimensions"] == 1536


async def test_the_async_index_appends_filter_fields_too(
    service: MongoDBService, adatabase: AsyncFakeDatabase
) -> None:
    await service.acreate_collection_vector_index(
        "chunks", dimensions=8, filter_fields=[{"type": "filter", "path": "tenant_id"}]
    )
    fields = created_definition(adatabase["chunks"])["fields"]
    assert [f.get("path") for f in fields] == ["embedding", "tenant_id"]


async def test_the_async_path_updates_an_existing_index_in_place(
    service: MongoDBService, adatabase: AsyncFakeDatabase
) -> None:
    collection = adatabase["chunks"]
    assert isinstance(collection, AsyncFakeCollection)
    collection.search_indexes.append("chunks_vector_index")
    await service.acreate_collection_vector_index("chunks", dimensions=8)
    assert collection.called("update_search_index") == 1
    assert collection.called("create_search_index") == 0


async def test_the_async_path_drops_and_recreates_on_an_incompatible_reshape(
    service: MongoDBService, adatabase: AsyncFakeDatabase
) -> None:
    collection = adatabase["chunks"]
    collection.search_indexes.append("chunks_vector_index")
    collection.update_error = OperationFailure("cannot reshape in place")
    await service.acreate_collection_vector_index("chunks", dimensions=8)
    assert [
        name for name, _, _ in collection.calls if name.endswith("search_index")
    ] == [
        "update_search_index",
        "drop_search_index",
        "create_search_index",
    ]
