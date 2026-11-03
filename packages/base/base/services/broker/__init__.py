from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import final

from contracts.events import Event

Handler = Callable[[Event], Awaitable[None]]


class AbstractBroker(ABC):
    @abstractmethod
    async def _apublish(self, event: Event) -> None: ...

    @final
    async def apublish(self, event: Event) -> None:
        await self._apublish(event)

    @abstractmethod
    async def _asubscribe(self, topic: str, group: str, handler: Handler) -> None: ...

    @final
    async def asubscribe(self, topic: str, group: str, handler: Handler) -> None:
        await self._asubscribe(topic, group, handler)

    @abstractmethod
    async def _astart(self) -> None: ...

    @final
    async def astart(self) -> None:
        await self._astart()

    @abstractmethod
    async def _astop(self) -> None: ...

    @final
    async def astop(self) -> None:
        await self._astop()

    @abstractmethod
    async def _areplay_dlq(self, topic: str, *, limit: int = 100) -> int: ...

    @final
    async def areplay_dlq(self, topic: str, *, limit: int = 100) -> int:
        return await self._areplay_dlq(topic, limit=limit)

    @abstractmethod
    async def _ahealth_check(self) -> bool:
        """Whether the transport is reachable."""

    @final
    async def ahealth_check(self) -> bool:
        return await self._ahealth_check()
