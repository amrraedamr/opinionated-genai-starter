"""The surface every vector backend must present, sync and async alike.

A caller picks a backend from settings and uses it through
`AbstractVectorSearchService`, so a method that exists on one backend and not
another is a runtime failure rather than a type error. These tests walk the
abstract class and check each concrete backend answers the whole of it, in
both call styles.
"""

import inspect
from typing import Any

import pytest

from base.services.vector_search import AbstractVectorSearchService
from base.services.vector_search.azure_ai_search import AzureAISearchService
from base.services.vector_search.pgvector import PgVectorSearchService
from base.services.vector_search.qdrant import QdrantVectorSearchService

BACKENDS: list[type[AbstractVectorSearchService]] = [
    QdrantVectorSearchService,
    AzureAISearchService,
    PgVectorSearchService,
]

REQUIRED = [
    "create_collection",
    "delete_collection",
    "collection_exists",
    "upsert",
    "search",
    "get",
    "delete",
    "count",
    "scroll",
    "health_check",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_the_abstract_service_declares_the_sync_method(name: str) -> None:
    assert name in AbstractVectorSearchService.__abstractmethods__


@pytest.mark.parametrize("name", REQUIRED)
def test_the_abstract_service_declares_the_async_method(name: str) -> None:
    assert f"a{name}" in AbstractVectorSearchService.__abstractmethods__


def test_the_abstract_service_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        AbstractVectorSearchService()  # type: ignore[abstract]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
def test_every_backend_implements_the_whole_surface(
    backend: type[AbstractVectorSearchService],
) -> None:
    assert backend.__abstractmethods__ == frozenset()


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("name", REQUIRED)
def test_every_backend_has_both_call_styles(
    backend: type[AbstractVectorSearchService], name: str
) -> None:
    assert callable(getattr(backend, name, None))
    assert callable(getattr(backend, f"a{name}", None))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("name", REQUIRED)
def test_the_async_half_of_every_pair_is_a_coroutine(
    backend: type[AbstractVectorSearchService], name: str
) -> None:
    assert inspect.iscoroutinefunction(getattr(backend, f"a{name}"))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
@pytest.mark.parametrize(
    "name", ["create_collection", "upsert", "search", "get", "delete", "scroll"]
)
def test_the_two_halves_of_a_pair_take_the_same_arguments(
    backend: type[AbstractVectorSearchService], name: str
) -> None:
    """A caller swapping `search` for `asearch` must not have to rewrite the call."""
    sync = inspect.signature(getattr(backend, name)).parameters
    asynchronous = inspect.signature(getattr(backend, f"a{name}")).parameters
    assert list(sync) == list(asynchronous)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
def test_every_backend_names_itself(
    backend: type[AbstractVectorSearchService],
) -> None:
    assert isinstance(backend.get_service_name(backend), str)  # type: ignore[arg-type]


def test_search_defaults_to_ten_results_everywhere() -> None:
    for backend in BACKENDS:
        parameters: Any = inspect.signature(backend.search).parameters
        assert parameters["limit"].default == 10


def test_create_collection_defaults_to_cosine_everywhere() -> None:
    for backend in BACKENDS:
        parameters: Any = inspect.signature(backend.create_collection).parameters
        assert parameters["distance"].default == "cosine"
