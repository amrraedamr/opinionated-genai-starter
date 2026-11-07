"""Singleton hygiene for the document-store unit tests.

Every client and service in this area is a process-wide singleton, so one
built against one test's settings would answer in the next and quietly
invalidate it. The purge is local to this directory on purpose.
"""

from collections.abc import Iterator

import pytest

from base.services.common.meta import Singleton


@pytest.fixture(autouse=True)
def _purge_singletons() -> Iterator[None]:
    """Drop every cached singleton around each test."""
    Singleton.purge()
    yield
    Singleton.purge()
