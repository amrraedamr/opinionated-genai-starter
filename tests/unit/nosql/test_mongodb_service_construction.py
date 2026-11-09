"""What `MongoDBService` needs before it will build, and what it no longer needs.

It used to read `azure_openai_embedding_dimensions` in `__init__` to size the
Atlas vector index, which made a document store refuse to start for anyone who
was not embedding with Azure. The width is now the caller's to state, so the
settings a service composes for Mongo are the Mongo settings and nothing else.
"""

import inspect
from typing import Any

import pytest

from base.services.configuration import get_settings
from base.services.configuration.mixins import (
    AzureOpenAIEmbeddingSettingsMixin,
    MongoSettingsMixin,
)
from base.services.nosql import AbstractNoSQLService
from base.services.nosql.mongodb import MongoDBService
from tests.conftest import MakeSettings


@pytest.fixture
def mongo_only(make_settings: MakeSettings) -> Any:
    """Settings composed of the Mongo mixin and nothing else."""
    return make_settings(
        MongoSettingsMixin, mongodb_host="db:27017", mongodb_database="app"
    )


def test_the_service_builds_with_only_the_mongo_mixin_composed(mongo_only: Any) -> None:
    assert MongoDBService().get_service_name() == "MongoDB"


def test_no_embedding_setting_exists_on_a_mongo_only_service(mongo_only: Any) -> None:
    """The guard for the test above: there is genuinely nothing to read."""
    MongoDBService()
    assert not hasattr(get_settings(), "azure_openai_embedding_dimensions")


def test_the_embedding_mixin_would_have_supplied_that_field(
    make_settings: MakeSettings,
) -> None:
    """Names the field the service used to require, so the pin cannot rot."""
    make_settings(
        AzureOpenAIEmbeddingSettingsMixin,
        MongoSettingsMixin,
        mongodb_host="db:27017",
        mongodb_database="app",
    )
    assert hasattr(get_settings(), "azure_openai_embedding_dimensions")


def test_the_service_reads_its_database_name_from_settings(mongo_only: Any) -> None:
    assert MongoDBService()._database_name == "app"


def test_the_service_is_one_instance_for_the_process(mongo_only: Any) -> None:
    assert MongoDBService() is MongoDBService()


def test_the_service_is_a_nosql_service(mongo_only: Any) -> None:
    assert isinstance(MongoDBService(), AbstractNoSQLService)


def test_the_abstract_service_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        AbstractNoSQLService()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "method",
    [
        "create_collection_vector_index",
        "acreate_collection_vector_index",
    ],
)
def test_the_vector_index_takes_its_width_as_an_explicit_argument(
    method: str,
) -> None:
    parameters = inspect.signature(getattr(MongoDBService, method)).parameters
    assert "dimensions" in parameters
    assert parameters["dimensions"].annotation is int
    assert parameters["dimensions"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "method",
    [
        "find_one",
        "find_many",
        "insert_one",
        "insert_many",
        "update_one",
        "update_many",
        "replace_one",
        "delete_one",
        "delete_many",
        "count",
        "distinct",
        "aggregate",
        "paginate",
        "health_check",
    ],
)
def test_every_sync_method_has_an_async_counterpart(method: str) -> None:
    assert callable(getattr(MongoDBService, f"a{method}", None))


def test_the_service_implements_every_abstract_method(mongo_only: Any) -> None:
    assert MongoDBService.__abstractmethods__ == frozenset()
