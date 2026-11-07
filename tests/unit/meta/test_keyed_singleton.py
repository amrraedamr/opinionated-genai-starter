"""`KeyedSingleton` and `KeyedSingletonABC`: one instance per (class, key).

Pins what counts as the key — an explicit `_key`, else a hash of the
constructor arguments — because that is what decides whether two callers share
a client or each open their own.
"""

from abc import abstractmethod
from collections.abc import Iterator

import pytest

from base.services.common.meta import KeyedSingleton, KeyedSingletonABC, purge_all
from base.tools.hashing import hash_args


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """Keyed instances are process-wide, so one test must not seed the next."""
    purge_all()
    yield
    purge_all()


class Pool(metaclass=KeyedSingleton):
    """A class that notices how often it was really constructed."""

    builds = 0

    def __init__(self, dsn: str = "memory://", pool_size: int = 1) -> None:
        Pool.builds += 1
        self.dsn = dsn
        self.pool_size = pool_size


class Registry(metaclass=KeyedSingleton):
    def __init__(self, dsn: str = "memory://") -> None:
        self.dsn = dsn


def test_the_same_explicit_key_returns_the_identical_object() -> None:
    assert Pool(_key="primary") is Pool(_key="primary")


def test_different_explicit_keys_return_different_objects() -> None:
    assert Pool(_key="primary") is not Pool(_key="replica")


def test_the_explicit_key_wins_over_the_constructor_arguments() -> None:
    first = Pool("postgres://a", _key="shared")
    second = Pool("postgres://b", _key="shared")
    assert first is second
    assert second.dsn == "postgres://a"


def test_equal_constructor_arguments_derive_the_same_key() -> None:
    Pool.builds = 0
    assert Pool("postgres://a", 5) is Pool("postgres://a", 5)
    assert Pool.builds == 1


def test_different_constructor_arguments_derive_different_keys() -> None:
    assert Pool("postgres://a") is not Pool("postgres://b")


def test_the_derived_key_is_the_hash_of_the_arguments() -> None:
    instance = Pool("postgres://a", 5)
    assert Pool.get(hash_args("postgres://a", 5)) is instance


def test_passing_an_argument_by_keyword_derives_a_different_key_than_by_position() -> (
    None
):
    """Pinned, not endorsed: the hash is taken over args and kwargs separately."""
    assert Pool("postgres://a") is not Pool(dsn="postgres://a")


def test_two_keyed_classes_sharing_a_key_do_not_share_an_instance() -> None:
    assert Pool(_key="same") is not Registry(_key="same")


def test_get_answers_with_the_instance_for_a_known_key() -> None:
    created = Pool(_key="primary")
    assert Pool.get("primary") is created


def test_get_answers_none_for_a_key_that_was_never_built() -> None:
    assert Pool.get("never-built") is None


def test_replace_swaps_the_instance_behind_one_key_only() -> None:
    kept = Pool(_key="replica")
    stand_in = object.__new__(Pool)
    assert Pool.replace("primary", stand_in) is stand_in
    assert Pool(_key="primary") is stand_in
    assert Pool(_key="replica") is kept


def test_purge_forgets_every_key_of_every_keyed_class() -> None:
    first = Pool(_key="primary")
    KeyedSingleton.purge()
    assert Pool.get("primary") is None
    assert Pool(_key="primary") is not first


class Store(metaclass=KeyedSingletonABC):
    """An abstract base that is also keyed."""

    @abstractmethod
    def read(self) -> str: ...


class FileStore(Store):
    def __init__(self, root: str = "/data") -> None:
        self.root = root

    def read(self) -> str:
        return self.root


def test_keyed_singleton_abc_refuses_a_class_with_an_unimplemented_method() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Store(_key="any")


def test_keyed_singleton_abc_keeps_one_instance_per_key() -> None:
    assert FileStore(_key="a") is FileStore(_key="a")
    assert FileStore(_key="a") is not FileStore(_key="b")


def test_an_empty_explicit_key_falls_back_to_the_derived_one() -> None:
    assert Pool("postgres://a", _key="") is Pool("postgres://a")
