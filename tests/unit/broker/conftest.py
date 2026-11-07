"""Fixtures for the in-process broker tests.

Both brokers are singletons holding an `asyncio.Queue` and an `asyncio.Event`,
which bind to the loop that built them. pytest gives every async test its own
loop, so one kept between tests would hand the next a dead queue.
"""

from collections.abc import Iterator

import pytest

from base.services.common.meta import purge_all


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    """Drop every singleton around each test."""
    purge_all()
    yield
    purge_all()
