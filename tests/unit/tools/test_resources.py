"""`resources.py`: closing clients that may never have been opened.

A service builds its SDK clients lazily, so shutdown runs over a mix of real
clients and `None`. These helpers exist so every `close` is not guarded at the
call site.
"""

import pytest

from base.tools.resources import (
    SupportsAsyncClose,
    SupportsClose,
    aclose_all,
    close_all,
)


class Client:
    """Something with a sync `close`, like an SDK client or a credential."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class AsyncClient:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class NotCloseable:
    pass


def test_anything_with_a_close_satisfies_the_sync_protocol() -> None:
    assert isinstance(Client(), SupportsClose)
    assert not isinstance(NotCloseable(), SupportsClose)


def test_anything_with_a_close_satisfies_the_async_protocol() -> None:
    """Runtime protocol checks see the name, not whether it is a coroutine."""
    assert isinstance(AsyncClient(), SupportsAsyncClose)
    assert not isinstance(NotCloseable(), SupportsAsyncClose)


def test_every_resource_given_is_closed() -> None:
    first, second = Client(), Client()
    close_all(first, second)
    assert (first.closed, second.closed) == (1, 1)


def test_a_resource_that_was_never_opened_is_skipped() -> None:
    opened = Client()
    close_all(None, opened, None)
    assert opened.closed == 1


def test_closing_nothing_at_all_is_allowed() -> None:
    close_all()
    close_all(None)


def test_a_failing_close_stops_the_rest() -> None:
    """Pinned as it stands: there is no per-resource guard."""

    class Stubborn:
        def close(self) -> None:
            raise RuntimeError("still busy")

    after = Client()
    with pytest.raises(RuntimeError, match="still busy"):
        close_all(Stubborn(), after)
    assert after.closed == 0


async def test_every_async_resource_given_is_awaited_closed() -> None:
    first, second = AsyncClient(), AsyncClient()
    await aclose_all(first, second)
    assert (first.closed, second.closed) == (1, 1)


async def test_an_async_resource_that_was_never_opened_is_skipped() -> None:
    opened = AsyncClient()
    await aclose_all(None, opened, None)
    assert opened.closed == 1


async def test_closing_no_async_resources_is_allowed() -> None:
    await aclose_all()
    await aclose_all(None)


async def test_resources_are_closed_in_the_order_they_were_given() -> None:
    order: list[str] = []

    class Named:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            order.append(self.name)

    await aclose_all(Named("client"), Named("credential"))
    assert order == ["client", "credential"]
