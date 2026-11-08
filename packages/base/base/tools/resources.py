"""Closing clients and credentials that may never have been initialized."""

import logging
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsClose(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class SupportsAsyncClose(Protocol):
    async def close(self) -> None: ...


def close_all(*resources: SupportsClose | None) -> None:
    """Close every resource that was initialized, skipping the ones that were not.

    One resource that will not close does not strand the rest; the first
    failure is re-raised once every other resource has been given its turn.
    """
    first: Exception | None = None
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as exc:
            logging.exception("Failed to close %s", type(resource).__name__)
            first = first or exc
    if first is not None:
        raise first


async def aclose_all(*resources: SupportsAsyncClose | None) -> None:
    """Close every resource that was initialized, skipping the ones that were not.

    One resource that will not close does not strand the rest; the first
    failure is re-raised once every other resource has been given its turn.
    """
    first: Exception | None = None
    for resource in resources:
        if resource is None:
            continue
        try:
            await resource.close()
        except Exception as exc:
            logging.exception("Failed to close %s", type(resource).__name__)
            first = first or exc
    if first is not None:
        raise first
