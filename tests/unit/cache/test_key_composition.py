"""How a cache key is composed from a key and a namespace.

`_get_key` is the whole of the cache's isolation story: two tenants sharing a
Redis rely on it to keep their entries apart. It is pure, so it is pinned here
rather than through a backend, and the cases that do *not* isolate are pinned
as loudly as the ones that do.
"""

from uuid import UUID, uuid5

from base.models.definitions.cache import CacheKeys
from base.services.cache import AbstractCacheService

get_key = AbstractCacheService._get_key


def test_a_key_without_a_namespace_hashes_under_the_nil_uuid() -> None:
    assert get_key("alpha") == str(uuid5(UUID(int=0), "alpha"))


def test_the_same_key_in_two_namespaces_does_not_collide() -> None:
    assert get_key("alpha", UUID(int=1)) != get_key("alpha", UUID(int=2))


def test_two_keys_in_one_namespace_do_not_collide() -> None:
    namespace = UUID(int=3)
    assert get_key("alpha", namespace) != get_key("beta", namespace)


def test_key_composition_is_deterministic() -> None:
    namespace = UUID(int=4)
    assert get_key("alpha", namespace) == get_key("alpha", namespace)


def test_a_uuid_namespace_and_its_string_form_compose_the_same_key() -> None:
    namespace = UUID("11111111-2222-3333-4444-555555555555")
    assert get_key("alpha", namespace) == get_key("alpha", str(namespace))


def test_an_enum_key_composes_as_its_value() -> None:
    assert get_key(CacheKeys.TEST) == get_key(CacheKeys.TEST.value)


def test_a_namespace_that_is_not_a_uuid_discards_the_key() -> None:
    """Pins a bug: every key under a non-UUID namespace is the same entry.

    `_get_key` falls back to returning the namespace itself when `UUID(...)`
    rejects it, so `get("a", "tenant")` and `get("b", "tenant")` read and write
    one shared value. `base/services/cache/__init__.py:236`.
    """
    assert get_key("alpha", "tenant") == "tenant"
    assert get_key("beta", "tenant") == "tenant"


def test_two_non_uuid_namespaces_still_differ_from_each_other() -> None:
    assert get_key("alpha", "tenant-a") != get_key("alpha", "tenant-b")


def test_a_non_string_key_is_composed_through_its_text() -> None:
    assert get_key(7) == get_key("7")  # type: ignore[arg-type]
