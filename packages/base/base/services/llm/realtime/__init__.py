"""A model session held open, rather than a call made and answered.

Deliberately not part of `AbstractLLMService`. That seam promises one request
produces one result, and every method on it is built on that: retries replay
the call, the token budget charges once, the span closes when the answer
arrives. None of it survives contact with a session that stays open for
minutes, bills per turn, and can be interrupted halfway through a sentence.

Bending one seam to cover both would cost the request/response path its
guarantees and give the realtime path none of its own.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from base.models.view_models.realtime_session import RealtimeInput, RealtimeOutput


class AbstractRealtimeSession(ABC):
    """One open conversation with a model.

    Async only. There is no sensible synchronous reading of a stream that
    arrives when the model feels like it, and offering one would mean
    blocking a thread for the length of a conversation.
    """

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def input_sample_rate(self) -> int:
        """What this provider listens at, in hertz.

        Asked rather than assumed, because the two disagree: OpenAI wants
        24 kHz and Gemini 16 kHz. Sending the wrong one raises nothing — the
        model simply hears a voice at the wrong pitch and speed, and answers
        a question nobody asked.
        """

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
        """Answer a tool the model asked for, mid-turn.

        Identified by the call id rather than position: a model may ask for
        several at once, and nothing guarantees they are answered in order.
        """

    @abstractmethod
    async def ainterrupt(self) -> None:
        """Stop the model's current turn because the user started talking.

        A person talking over the model expects it to stop, and audio already
        buffered on the client keeps playing unless the turn is cancelled
        rather than merely ignored.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """End the session. Idempotent: a closed session closes quietly."""


__all__ = ["AbstractRealtimeSession"]
