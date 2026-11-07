"""Fixtures for the messaging tests that need no database.

`OutboxDispatcher` and `InboxProcessor` both reach a `SQLService` singleton,
so anything built here is dropped around every test rather than carried into
the next one with a closed event loop behind it.
"""

from collections.abc import Iterator

import pytest

from base.services.common.meta import purge_all


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    purge_all()
    yield
    purge_all()
