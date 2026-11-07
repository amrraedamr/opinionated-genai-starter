"""`Singleton` and `SingletonABC`: one instance per class, process-wide.

Pins the identity guarantee every service in this repository relies on, and
that making a class a singleton does not stop ABC refusing an incomplete one.
"""

from abc import abstractmethod
from collections.abc import Iterator

import pytest

from base.services.common.meta import Singleton, SingletonABC, purge_all


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """Instances are process-wide, so one test must not seed the next."""
    purge_all()
    yield
    purge_all()


class Counter(metaclass=Singleton):
    """A class that notices how often it was really constructed."""

    builds = 0

    def __init__(self, label: str = "default") -> None:
        Counter.builds += 1
        self.label = label


class Other(metaclass=Singleton):
    def __init__(self) -> None:
        self.value = 1


def test_a_singleton_class_builds_its_instance_once() -> None:
    Counter.builds = 0
    first = Counter("a")
    second = Counter("b")
    assert first is second
    assert Counter.builds == 1


def test_later_constructor_arguments_are_ignored_once_an_instance_exists() -> None:
    assert Counter("first").label == "first"
    assert Counter("second").label == "first"


def test_two_singleton_classes_do_not_share_an_instance() -> None:
    assert Counter() is not Other()


def test_a_subclass_gets_its_own_instance() -> None:
    class Child(Counter):
        pass

    assert Child() is not Counter()
    assert Child() is Child()


def test_replace_swaps_the_cached_instance() -> None:
    original = Counter()
    stand_in = object.__new__(Counter)
    assert Counter.replace(stand_in) is stand_in
    assert Counter() is stand_in
    assert Counter() is not original


def test_purge_forgets_every_singleton_class_at_once() -> None:
    first = Counter()
    Other()
    Singleton.purge()
    assert Counter() is not first


def test_singleton_instances_live_in_one_shared_registry() -> None:
    Counter()
    Other()
    assert {Counter, Other} <= set(Singleton._instances)


class Capability(metaclass=SingletonABC):
    """An abstract base that is also a singleton."""

    @abstractmethod
    def run(self) -> str: ...


class Implemented(Capability):
    def run(self) -> str:
        return "ran"


def test_singleton_abc_still_refuses_a_class_with_an_unimplemented_method() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Capability()


def test_singleton_abc_builds_a_complete_subclass_once() -> None:
    assert Implemented() is Implemented()
    assert Implemented().run() == "ran"


def test_singleton_abc_subclasses_each_get_their_own_instance() -> None:
    class Alternative(Capability):
        def run(self) -> str:
            return "other"

    assert Implemented() is not Alternative()


def test_singleton_construction_is_not_thread_safe() -> None:
    """`__call__` holds no lock, so two threads can both pass the check.

    Pinned rather than wished away: callers that need exclusivity under
    concurrency have to arrange it themselves, or the metaclass has to grow a
    lock — and this test is what would notice if it did.
    """
    import threading
    import time

    class Slow(metaclass=Singleton):
        builds = 0

        def __init__(self) -> None:
            Slow.builds += 1
            time.sleep(0.05)

    barrier = threading.Barrier(2)

    def construct() -> None:
        barrier.wait()
        Slow()

    threads = [threading.Thread(target=construct) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert Slow.builds == 2
