"""The index Azure AI Search is asked to create, field by field.

No Azure AI Search service is reachable from this repository and no key
exists, so the adapter is pinned entirely by reading the `SearchIndex` it
builds. The schema is a convention — an `id` key, a vector field, and
metadata flattened to a JSON string — and a drift in any of those silently
breaks every document already written.
"""

from typing import Any

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.identity.aio import (
    DefaultAzureCredential as AsyncDefaultAzureCredential,
)
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SimpleField,
    VectorSearchAlgorithmMetric,
)

from base.services.configuration.mixins import AzureSearchSettingsMixin
from base.services.vector_search.azure_ai_search import AzureAISearchService


@pytest.fixture
def service(make_settings: Any) -> AzureAISearchService:
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://account.search.windows.net",
        azure_search_api_key="the-key",
    )
    return AzureAISearchService()


def fields_by_name(index: Any) -> dict[str, Any]:
    return {field.name: field for field in index.fields}


def test_the_index_takes_the_collection_name(service: AzureAISearchService) -> None:
    assert service._build_index("chunks", 8, "cosine").name == "chunks"


def test_the_key_field_is_a_filterable_string(service: AzureAISearchService) -> None:
    key = fields_by_name(service._build_index("chunks", 8, "cosine"))["id"]
    assert key.type == SearchFieldDataType.String
    assert key.key is True
    assert key.filterable is True


def test_the_vector_field_is_a_collection_of_singles(
    service: AzureAISearchService,
) -> None:
    vector = fields_by_name(service._build_index("chunks", 8, "cosine"))["vector"]
    assert vector.type == SearchFieldDataType.Collection(SearchFieldDataType.Single)
    assert vector.searchable is True


def test_the_vector_field_carries_the_width_the_caller_named(
    service: AzureAISearchService,
) -> None:
    vector = fields_by_name(service._build_index("chunks", 1536, "cosine"))["vector"]
    assert vector.vector_search_dimensions == 1536


def test_the_vector_field_points_at_the_profile(service: AzureAISearchService) -> None:
    index = service._build_index("chunks", 8, "cosine")
    vector = fields_by_name(index)["vector"]
    assert vector.vector_search_profile_name == index.vector_search.profiles[0].name


def test_the_profile_points_at_the_algorithm(service: AzureAISearchService) -> None:
    search = service._build_index("chunks", 8, "cosine").vector_search
    assert search.profiles[0].algorithm_configuration_name == search.algorithms[0].name


def test_metadata_is_stored_as_an_unfilterable_string(
    service: AzureAISearchService,
) -> None:
    """It holds arbitrary JSON, so Azure cannot index it as a field."""
    metadata = fields_by_name(service._build_index("chunks", 8, "cosine"))["metadata"]
    assert metadata.type == SearchFieldDataType.String
    assert metadata.filterable is False


@pytest.mark.parametrize(
    ("named", "expected"),
    [
        ("cosine", VectorSearchAlgorithmMetric.COSINE),
        ("dot", VectorSearchAlgorithmMetric.DOT_PRODUCT),
        ("euclidean", VectorSearchAlgorithmMetric.EUCLIDEAN),
    ],
)
def test_each_distance_name_maps_to_an_azure_metric(
    service: AzureAISearchService, named: str, expected: VectorSearchAlgorithmMetric
) -> None:
    index = service._build_index("chunks", 8, named)
    assert index.vector_search.algorithms[0].parameters.metric == expected


def test_an_unknown_distance_falls_back_to_cosine(
    service: AzureAISearchService,
) -> None:
    index = service._build_index("chunks", 8, "manhattan")
    assert (
        index.vector_search.algorithms[0].parameters.metric
        == VectorSearchAlgorithmMetric.COSINE
    )


def test_the_schema_is_three_fields_without_extras(
    service: AzureAISearchService,
) -> None:
    assert [f.name for f in service._build_index("chunks", 8, "cosine").fields] == [
        "id",
        "vector",
        "metadata",
    ]


def test_extra_fields_are_appended_after_the_convention(
    service: AzureAISearchService,
) -> None:
    extra = SimpleField(
        name="tenant_id", type=SearchFieldDataType.String, filterable=True
    )
    index = service._build_index("chunks", 8, "cosine", extra_fields=[extra])
    assert [f.name for f in index.fields] == ["id", "vector", "metadata", "tenant_id"]


def test_several_extra_fields_keep_their_order(
    service: AzureAISearchService,
) -> None:
    extras: list[SearchField] = [
        SimpleField(name="a", type=SearchFieldDataType.String),
        SimpleField(name="b", type=SearchFieldDataType.Int32),
    ]
    index = service._build_index("chunks", 8, "cosine", extra_fields=extras)
    assert [f.name for f in index.fields][-2:] == ["a", "b"]


def test_the_field_names_follow_the_configured_settings(make_settings: Any) -> None:
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://account.search.windows.net",
        azure_search_api_key="the-key",
        azure_search_key_field="chunk_id",
        azure_search_vector_field="embedding",
    )
    index = AzureAISearchService()._build_index("chunks", 8, "cosine")
    assert [f.name for f in index.fields] == ["chunk_id", "embedding", "metadata"]


def test_an_unset_endpoint_is_refused_at_construction(make_settings: Any) -> None:
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="",
        azure_search_api_key="the-key",
    )
    with pytest.raises(ValueError, match="azure_search_endpoint"):
        AzureAISearchService()


def test_an_unset_api_key_falls_back_to_the_deployments_own_identity(
    make_settings: Any,
) -> None:
    """No key means a role assignment, as Cosmos has always allowed."""
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://account.search.windows.net",
        azure_search_api_key="",
    )
    client = AzureAISearchService()._client
    assert client._key_credential is None
    assert isinstance(client._credential, DefaultAzureCredential)


def test_a_configured_api_key_is_used_instead_of_an_identity(
    make_settings: Any,
) -> None:
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://account.search.windows.net",
        azure_search_api_key="the-key",
    )
    client = AzureAISearchService()._client
    assert isinstance(client._credential, AzureKeyCredential)
    assert client._managed_credential is None


def test_the_async_clients_take_an_async_identity(make_settings: Any) -> None:
    """A sync credential on an async client cannot be awaited closed."""
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://account.search.windows.net",
        azure_search_api_key="",
    )
    client = AzureAISearchService()._client
    assert isinstance(client._async_credential, AsyncDefaultAzureCredential)


def test_the_service_names_itself_azure_ai_search(
    service: AzureAISearchService,
) -> None:
    assert service.get_service_name() == "Azure AI Search"
