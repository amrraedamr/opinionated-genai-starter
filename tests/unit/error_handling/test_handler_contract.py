"""What every concrete handler must do, checked the same way for all of them.

The per-SDK files say how one client's exceptions are read. This one says what
is true of all fourteen handlers regardless: each failure kind becomes its own
class inside one family, a kind that retries becomes a class that can report
attempts and a kind that does not becomes one that cannot, and a real SDK
error driven through `retrying` is repeated exactly as often as its kind
allows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.blob import BlobError
from base.models.view_models.exceptions.cache import CacheError
from base.models.view_models.exceptions.embedding import EmbeddingError
from base.models.view_models.exceptions.keyvault import KeyVaultError
from base.models.view_models.exceptions.llm import LLMError
from base.models.view_models.exceptions.nosql import NoSqlError
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.models.view_models.exceptions.rerank import RerankError
from base.models.view_models.exceptions.sql import SQLError
from base.models.view_models.exceptions.vector_search import VectorSearchError
from base.services.error_handling import AbstractErrorHandler, aretrying, retrying
from tests.fakes import FlakyCall, no_sleep
from tests.unit.error_handling.support import AzureResponse, http_response


def _openai_errors() -> tuple[Exception, Exception]:
    from openai import InternalServerError, NotFoundError

    return (
        InternalServerError("down", response=http_response(500), body=None),
        NotFoundError("gone", response=http_response(404), body=None),
    )


def _anthropic_errors() -> tuple[Exception, Exception]:
    from anthropic import NotFoundError, OverloadedError

    return (
        OverloadedError("busy", response=http_response(529), body=None),
        NotFoundError("gone", response=http_response(404), body=None),
    )


def _gemini_errors() -> tuple[Exception, Exception]:
    from google.genai import errors as genai_errors

    return (
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        genai_errors.ClientError(404, {"error": {"message": "no model"}}),
    )


def _mongo_errors() -> tuple[Exception, Exception]:
    from pymongo.errors import AutoReconnect, DuplicateKeyError

    return AutoReconnect("socket closed"), DuplicateKeyError("E11000", code=11000)


def _redis_errors() -> tuple[Exception, Exception]:
    from redis.exceptions import BusyLoadingError, ResponseError

    return BusyLoadingError("loading"), ResponseError("WRONGTYPE")


def _cohere_errors() -> tuple[Exception, Exception]:
    from cohere.errors import NotFoundError, ServiceUnavailableError

    return ServiceUnavailableError("busy"), NotFoundError("gone")


def _qdrant_errors() -> tuple[Exception, Exception]:
    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse

    def response(status: int) -> UnexpectedResponse:
        return UnexpectedResponse(
            status_code=status,
            reason_phrase="Test",
            content=b"{}",
            headers=httpx.Headers({}),
        )

    return response(503), response(404)


def _sqlalchemy_errors() -> tuple[Exception, Exception]:
    import asyncpg
    from sqlalchemy.exc import IntegrityError, OperationalError

    return (
        OperationalError(
            "UPDATE t", {}, asyncpg.exceptions.DeadlockDetectedError("deadlock")
        ),
        IntegrityError(
            "INSERT t", {}, asyncpg.exceptions.UniqueViolationError("duplicate")
        ),
    )


def _azure_errors() -> tuple[Exception, Exception]:
    from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

    return (
        HttpResponseError(message="throttled", response=AzureResponse(429)),
        ResourceNotFoundError(message="gone", response=AzureResponse(404)),
    )


def _handler(path: str, name: str) -> Callable[[], AbstractErrorHandler[Any]]:
    """Import a handler lazily so a collection error names the module."""

    def build() -> AbstractErrorHandler[Any]:
        import importlib

        return getattr(importlib.import_module(path), name)(retries=2, base_wait=0.0)

    return build


HANDLERS: tuple[
    tuple[str, Callable[[], Any], type[Exception], Callable[[], Any]], ...
] = (
    (
        "openai-llm",
        _handler("base.services.error_handling.llm.openai", "OpenAIErrorHandler"),
        LLMError,
        _openai_errors,
    ),
    (
        "openai-embedding",
        _handler(
            "base.services.error_handling.embedding.openai",
            "OpenAIEmbeddingErrorHandler",
        ),
        EmbeddingError,
        _openai_errors,
    ),
    (
        "anthropic",
        _handler("base.services.error_handling.llm.anthropic", "AnthropicErrorHandler"),
        LLMError,
        _anthropic_errors,
    ),
    (
        "gemini",
        _handler("base.services.error_handling.llm.gemini", "GeminiErrorHandler"),
        LLMError,
        _gemini_errors,
    ),
    (
        "mongodb",
        _handler("base.services.error_handling.nosql.mongodb", "MongoErrorHandler"),
        NoSqlError,
        _mongo_errors,
    ),
    (
        "redis",
        _handler("base.services.error_handling.cache.redis", "RedisErrorHandler"),
        CacheError,
        _redis_errors,
    ),
    (
        "cohere",
        _handler("base.services.error_handling.rerank.cohere", "CohereErrorHandler"),
        RerankError,
        _cohere_errors,
    ),
    (
        "qdrant",
        _handler(
            "base.services.error_handling.vector_search.qdrant", "QdrantErrorHandler"
        ),
        VectorSearchError,
        _qdrant_errors,
    ),
    (
        "sql",
        _handler("base.services.error_handling.sql", "SQLErrorHandler"),
        SQLError,
        _sqlalchemy_errors,
    ),
    (
        "pgvector",
        _handler(
            "base.services.error_handling.vector_search.pgvector",
            "PgVectorErrorHandler",
        ),
        VectorSearchError,
        _sqlalchemy_errors,
    ),
    (
        "blob",
        _handler("base.services.error_handling.blob.azure", "AzureBlobErrorHandler"),
        BlobError,
        _azure_errors,
    ),
    (
        "keyvault",
        _handler(
            "base.services.error_handling.keyvault.azure", "AzureKeyVaultErrorHandler"
        ),
        KeyVaultError,
        _azure_errors,
    ),
    (
        "cosmos",
        _handler("base.services.error_handling.nosql.cosmos", "CosmosErrorHandler"),
        NoSqlError,
        _azure_errors,
    ),
    (
        "azure-search",
        _handler(
            "base.services.error_handling.vector_search.azure",
            "AzureSearchErrorHandler",
        ),
        VectorSearchError,
        _azure_errors,
    ),
)

CASES = [pytest.param(*case[1:], id=case[0]) for case in HANDLERS]


def domain_classes(handler: AbstractErrorHandler[Any]) -> dict[FailureKind, type]:
    """Every kind's domain class, however this handler reaches it."""
    from tests.unit.error_handling.support import domain_class_for

    return {kind: domain_class_for(handler, kind) for kind in FailureKind}


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_every_kind_has_a_class_inside_one_family(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del errors
    classes = domain_classes(build())
    assert set(classes) == set(FailureKind)
    assert all(issubclass(cls, family) for cls in classes.values())


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_no_two_kinds_share_a_class(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del family, errors
    classes = domain_classes(build())
    assert len(set(classes.values())) == len(FailureKind)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_retryable_kind_becomes_a_class_that_can_report_attempts(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del family, errors
    classes = domain_classes(build())
    for kind in RETRYABLE_FAILURES:
        assert issubclass(classes[kind], ProviderRetryExhausted)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_terminal_kind_becomes_a_class_that_cannot_report_attempts(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del family, errors
    classes = domain_classes(build())
    for kind in set(FailureKind) - RETRYABLE_FAILURES:
        assert not issubclass(classes[kind], ProviderRetryExhausted)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_an_unknown_failure_becomes_the_familys_unexpected_class(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del errors
    handler = build()
    unexpected = handler.wrap_unexpected(ValueError("nothing anticipated"))
    assert isinstance(unexpected, family)
    assert type(unexpected).__name__.endswith("UnexpectedError")
    assert type(unexpected) is domain_classes(handler)[FailureKind.UNKNOWN]


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_handler_reads_a_kind_off_every_error_it_claims(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del family
    handler = build()
    for error in errors():
        assert isinstance(error, handler.provider_error_type)
        assert isinstance(handler.failure_kind(error), FailureKind)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_transient_sdk_error_is_repeated_until_the_retries_run_out(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], Any],
    family: type[Exception],
    errors: Callable[[], Any],
) -> None:
    no_sleep(monkeypatch)
    handler = build()
    transient, _ = errors()
    call = FlakyCall(failures=99, error=transient)
    with pytest.raises(family) as caught:
        handler.run(call)
    assert call.attempts == 3
    assert isinstance(caught.value, ProviderRetryExhausted)
    assert caught.value.attempts == 3
    assert caught.value.provider_error is transient


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_terminal_sdk_error_is_answered_once(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], Any],
    family: type[Exception],
    errors: Callable[[], Any],
) -> None:
    no_sleep(monkeypatch)
    handler = build()
    _, terminal = errors()
    call = FlakyCall(failures=99, error=terminal)
    with pytest.raises(family) as caught:
        handler.run(call)
    assert call.attempts == 1
    assert not isinstance(caught.value, ProviderRetryExhausted)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
async def test_a_transient_sdk_error_is_repeated_when_awaited(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], Any],
    family: type[Exception],
    errors: Callable[[], Any],
) -> None:
    no_sleep(monkeypatch)
    handler = build()
    transient, _ = errors()
    call = FlakyCall(failures=2, error=transient)
    assert await handler.arun(call.acall) == "ok"
    assert call.attempts == 3
    assert issubclass(family, ProviderError)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_decorated_method_retries_exactly_what_its_handler_allows(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], Any],
    family: type[Exception],
    errors: Callable[[], Any],
) -> None:
    no_sleep(monkeypatch)
    transient, terminal = errors()

    class Service:
        def __init__(self) -> None:
            self._error_handler = build()
            self.attempts = 0
            self.error: Exception = transient

        @retrying
        def call(self) -> str:
            self.attempts += 1
            raise self.error

    service = Service()
    with pytest.raises(family):
        service.call()
    assert service.attempts == 3

    service.attempts = 0
    service.error = terminal
    with pytest.raises(family):
        service.call()
    assert service.attempts == 1


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
async def test_a_decorated_async_method_retries_exactly_what_its_handler_allows(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], Any],
    family: type[Exception],
    errors: Callable[[], Any],
) -> None:
    no_sleep(monkeypatch)
    transient, _ = errors()

    class Service:
        def __init__(self) -> None:
            self._error_handler = build()
            self.attempts = 0

        @aretrying
        async def call(self) -> str:
            self.attempts += 1
            raise transient

    service = Service()
    with pytest.raises(family):
        await service.call()
    assert service.attempts == 3


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_an_unhandled_failure_becomes_the_same_family_without_running_anything(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    handler = build()
    transient, terminal = errors()
    assert isinstance(handler.as_domain_error(transient), ProviderRetryExhausted)
    assert isinstance(handler.as_domain_error(terminal), family)
    assert isinstance(handler.as_domain_error(ValueError("foreign")), family)


@pytest.mark.parametrize(("build", "family", "errors"), CASES)
def test_a_handler_exposes_the_whole_classification_surface(
    build: Callable[[], Any], family: type[Exception], errors: Callable[[], Any]
) -> None:
    del family, errors
    handler = build()
    for name in ("failure_kind", "classify", "wrap", "wrap_exhausted", "retry_after"):
        assert callable(getattr(handler, name))
    assert issubclass(handler.provider_error_type, Exception)


def test_every_handler_in_the_package_is_covered_here() -> None:
    """A handler added to the package without a case here fails this."""
    import importlib
    import pkgutil

    import base.services.error_handling as package

    found: set[str] = set()
    for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        imported = importlib.import_module(module.name)
        for name, obj in vars(imported).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, AbstractErrorHandler)
                and obj.__module__ == module.name
                and not getattr(obj, "__abstractmethods__", None)
            ):
                found.add(name)

    covered = {
        build().__class__.__name__
        for _, build, _, _ in HANDLERS  # noqa: SLF001
    }
    assert found == covered
