"""The rules every cache backend shares, pinned on the abstraction itself.

`_check_ttl`, `_encode` and `_refuse_type_clash` live on
`AbstractCacheService` so both backends answer a caller the same way. They are
pure, so they are pinned here rather than through a backend; each one is also
exercised against the real server in `tests/integration/cache`.
"""

import pytest

from base.models.view_models.exceptions.cache import (
    CacheBadRequestError,
    CacheUnexpectedError,
)
from base.services.cache import AbstractCacheService

check_ttl = AbstractCacheService._check_ttl
encode = AbstractCacheService._encode


def test_no_expiry_at_all_is_allowed() -> None:
    assert check_ttl(None) is None


def test_an_expiry_in_the_future_is_allowed() -> None:
    assert check_ttl(1) is None


@pytest.mark.parametrize("ttl", [0, -1, -3600])
def test_an_expiry_that_is_not_in_the_future_is_refused(ttl: int) -> None:
    with pytest.raises(CacheBadRequestError, match="invalid expire time"):
        check_ttl(ttl)


def test_a_value_is_encoded_as_json() -> None:
    assert encode({"a": 1}) == '{"a": 1}'


def test_a_string_is_encoded_as_a_quoted_string() -> None:
    assert encode("hello") == '"hello"'


def test_a_value_the_encoder_cannot_reach_becomes_a_cache_error() -> None:
    with pytest.raises(CacheUnexpectedError, match="not JSON serializable") as caught:
        encode({1, 2})
    assert isinstance(caught.value.provider_error, TypeError)


def test_a_type_clash_names_the_answer_redis_gives() -> None:
    with pytest.raises(CacheBadRequestError, match="WRONGTYPE"):
        AbstractCacheService._refuse_type_clash()


def test_both_backends_inherit_the_guards_rather_than_owning_them() -> None:
    from base.services.cache.memory import InMemoryCacheService

    for name in ("_check_ttl", "_encode", "_refuse_type_clash"):
        assert name not in vars(InMemoryCacheService)
        assert name in vars(AbstractCacheService)
