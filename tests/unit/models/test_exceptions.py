"""The domain error families: one per capability over one shared shape.

Catching a model failure must never catch a store failure, and every class a
retry can end at must carry the attempt count. Both are properties of the whole
tree rather than of any one class, so they are checked by walking it.
"""

import inspect

import pytest

from base.models.view_models import exceptions as family
from base.models.view_models.exceptions.blob import BlobError, BlobNotFoundError
from base.models.view_models.exceptions.cache import CacheError
from base.models.view_models.exceptions.embedding import EmbeddingError
from base.models.view_models.exceptions.keyvault import KeyVaultError
from base.models.view_models.exceptions.llm import (
    LLMContentFilterError,
    LLMError,
    LLMIncompleteError,
    LLMStreamError,
)
from base.models.view_models.exceptions.mcp import McpError
from base.models.view_models.exceptions.nosql import NoSqlError
from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)
from base.models.view_models.exceptions.rerank import RerankError
from base.models.view_models.exceptions.sql import SQLError
from base.models.view_models.exceptions.vector_search import VectorSearchError

ROOTS: list[type[ProviderError]] = [
    BlobError,
    CacheError,
    EmbeddingError,
    KeyVaultError,
    LLMError,
    McpError,
    NoSqlError,
    RerankError,
    SQLError,
    VectorSearchError,
]


SHARED: tuple[type[ProviderError], ...] = (
    ProviderError,
    ProviderConfigError,
    ProviderRetryExhausted,
)


def exported_errors() -> list[type[ProviderError]]:
    """Every error class the package re-exports."""
    return [
        obj
        for name in family.__all__
        if inspect.isclass(obj := getattr(family, name))
        and issubclass(obj, ProviderError)
    ]


def domain_errors() -> list[type[ProviderError]]:
    """Every exported class that a capability owns."""
    return [cls for cls in exported_errors() if cls not in SHARED]


def test_every_exported_name_is_a_provider_error() -> None:
    exported = exported_errors()
    assert len(exported) == len(family.__all__)
    assert all(issubclass(cls, Exception) for cls in exported)


@pytest.mark.parametrize("root", ROOTS, ids=lambda cls: cls.__name__)
def test_each_family_descends_from_the_one_shared_shape(
    root: type[ProviderError],
) -> None:
    assert issubclass(root, ProviderError)
    assert root is not ProviderError


@pytest.mark.parametrize("root", ROOTS, ids=lambda cls: cls.__name__)
def test_catching_one_family_never_catches_another(root: type[ProviderError]) -> None:
    others = [other for other in ROOTS if other is not root]
    assert not any(issubclass(other, root) for other in others)


@pytest.mark.parametrize("cls", domain_errors(), ids=lambda cls: cls.__name__)
def test_every_error_belongs_to_exactly_one_family(cls: type[ProviderError]) -> None:
    assert sum(issubclass(cls, root) for root in ROOTS) == 1


def test_the_shared_shape_belongs_to_no_family() -> None:
    for cls in SHARED:
        assert not any(issubclass(cls, root) for root in ROOTS)


RETRYABLE_SUFFIXES = (
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ConnectionError",
)


def retryable_errors() -> list[type[ProviderError]]:
    """Every class a handler reaches only after its retries ran out."""
    return [cls for cls in domain_errors() if cls.__name__.endswith(RETRYABLE_SUFFIXES)]


@pytest.mark.parametrize("cls", retryable_errors(), ids=lambda cls: cls.__name__)
def test_a_retryable_class_is_always_a_retry_exhausted_class(
    cls: type[ProviderError],
) -> None:
    """A handler asserts this before wrapping, so the tree must already hold it."""
    assert issubclass(cls, ProviderRetryExhausted)


RETRYABLE_CLASSES: dict[type[ProviderError], int] = {McpError: 2}
"""Families the SDK lets split less finely; MCP folds a rate limit and a server error into one."""


def test_every_family_contributes_its_retryable_classes() -> None:
    for root in ROOTS:
        found = [cls for cls in retryable_errors() if issubclass(cls, root)]
        assert len(found) == RETRYABLE_CLASSES.get(root, 4), root.__name__


@pytest.mark.parametrize("root", ROOTS, ids=lambda cls: cls.__name__)
def test_each_family_has_its_own_retry_exhausted_class(
    root: type[ProviderError],
) -> None:
    exhausted = [
        cls
        for cls in exported_errors()
        if issubclass(cls, root) and cls.__name__.endswith("RetryExhausted")
    ]
    assert len(exhausted) == 1
    assert issubclass(exhausted[0], ProviderRetryExhausted)


@pytest.mark.parametrize("root", ROOTS, ids=lambda cls: cls.__name__)
def test_each_family_marks_its_configuration_error_as_unretryable(
    root: type[ProviderError],
) -> None:
    config = next(
        cls
        for cls in exported_errors()
        if issubclass(cls, root) and cls.__name__.endswith("ConfigError")
    )
    assert issubclass(config, ProviderConfigError)
    assert not issubclass(config, ProviderRetryExhausted)


def test_a_provider_error_defaults_every_detail_to_unknown() -> None:
    error = ProviderError("boom")
    assert error.status_code is None
    assert error.request_id is None
    assert error.provider_error is None
    assert str(error) == "boom"


def test_a_provider_error_reports_the_status_and_request_it_was_given() -> None:
    error = ProviderError("boom", status_code=503, request_id="req-1")
    assert str(error) == "boom | status=503 | request_id=req-1"


def test_a_provider_error_reports_a_status_without_a_request_id() -> None:
    assert str(ProviderError("boom", status_code=404)) == "boom | status=404"


def test_from_provider_keeps_the_sdk_exception_for_inspection() -> None:
    cause = RuntimeError("socket closed")
    error = BlobNotFoundError.from_provider(cause, status_code=404, request_id="r")
    assert isinstance(error, BlobNotFoundError)
    assert error.provider_error is cause
    assert error.status_code == 404
    assert error.request_id == "r"


def test_from_provider_falls_back_to_the_sdk_message() -> None:
    assert "socket closed" in str(
        BlobNotFoundError.from_provider(RuntimeError("socket closed"))
    )


def test_from_provider_prefers_an_explicit_message() -> None:
    error = BlobNotFoundError.from_provider(RuntimeError("raw"), "blob is gone")
    assert str(error) == "blob is gone"


def test_from_provider_returns_the_subclass_it_was_called_on() -> None:
    assert type(KeyVaultError.from_provider(RuntimeError("x"))) is KeyVaultError


def test_a_retry_exhausted_error_reports_how_many_tries_it_took() -> None:
    error = ProviderRetryExhausted(
        "gave up", attempts=4, status_code=429, provider_error=RuntimeError("429")
    )
    assert error.attempts == 4
    assert str(error) == "gave up | status=429 | attempts=4"


def test_a_retry_exhausted_error_requires_the_attempt_count() -> None:
    with pytest.raises(TypeError):
        ProviderRetryExhausted("gave up")  # type: ignore[call-arg]


def test_the_llm_family_adds_failures_no_other_capability_has() -> None:
    for cls in (LLMContentFilterError, LLMIncompleteError, LLMStreamError):
        assert issubclass(cls, LLMError)
        assert not any(issubclass(cls, root) for root in ROOTS if root is not LLMError)
