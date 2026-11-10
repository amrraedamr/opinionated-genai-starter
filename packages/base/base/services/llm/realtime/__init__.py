"""A model session held open, rather than a call made and answered."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from base.models.view_models.realtime_session import RealtimeInput, RealtimeOutput


class AbstractRealtimeSession(ABC):
    """One open conversation with a model."""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def input_sample_rate(self) -> int:
        """What this provider listens at, in hertz; a wrong rate raises nothing."""

    @abstractmethod
    def get_service_name(self) -> str: ...

    @abstractmethod
    async def aconnect(self) -> None:
        """Open the session and negotiate it. Everything else assumes this ran."""

    @abstractmethod
    async def asend(self, value: RealtimeInput) -> None:
        """Pass input to the model. Audio is a fragment; order is the meaning."""

    @abstractmethod
    def areceive(self) -> AsyncIterator[RealtimeOutput]:
        """Events as the model produces them, until the session closes."""

    @abstractmethod
    async def asend_tool_result(self, call_id: str, result: str) -> None:
        """Answer a tool the model asked for, mid-turn."""

    @abstractmethod
    async def ainterrupt(self) -> None:
        """Stop the model's current turn because the user started talking."""

    @abstractmethod
    async def aclose(self) -> None:
        """End the session. Idempotent: a closed session closes quietly."""


__all__ = ["AbstractRealtimeSession"]
