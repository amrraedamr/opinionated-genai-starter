"""Closing clients and credentials that may never have been initialized."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsClose(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class SupportsAsyncClose(Protocol):
    async def close(self) -> None: ...


def close_all(*resources: SupportsClose | None) -> None:
    """Close every resource that was initialized, skipping the ones that were not."""
    for resource in resources:
        if resource is not None:
            resource.close()


async def aclose_all(*resources: SupportsAsyncClose | None) -> None:
    """Close every resource that was initialized, skipping the ones that were not."""
    for resource in resources:
        if resource is not None:
            await resource.close()
