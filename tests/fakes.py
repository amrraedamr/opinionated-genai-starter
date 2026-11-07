"""Stand-ins for the things a unit test must not really call.

A fake here answers like the SDK it replaces rather than like a mock: it
returns the SDK's own response objects where it can, so a test exercises the
real parsing code instead of agreeing with itself. Where an SDK object cannot
be built without a network, the smallest faithful shape stands in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RecordingCall:
    """Captures the request an adapter built, and answers with a canned reply.

    What a provider is *sent* is most of what these adapters do, so the
    request is kept rather than discarded: a test asserts on the body that
    would have gone out.
    """

    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "nothing was sent"
        return self.calls[-1]

    @property
    def count(self) -> int:
        return len(self.calls)


class FlakyCall:
    """Fails a set number of times, then succeeds.

    The shape every retry test needs: it proves an attempt was repeated, and
    counts how many times, without waiting on a real backoff.
    """

    def __init__(self, failures: int, error: Exception, reply: Any = "ok") -> None:
        self.remaining = failures
        self.error = error
        self.reply = reply
        self.attempts = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        return self.reply

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)


def no_sleep(monkeypatch: Any) -> None:
    """Make every backoff instant, so a retry test costs nothing."""
    import asyncio
    import time

    async def _asleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _asleep)
    monkeypatch.setattr(time, "sleep", lambda _: None)


def build_tool(name: str = "get_weather") -> Any:
    """A real `Tool`, since adapters read its name, description and schema."""
    from base.models.view_models import BaseModel
    from base.services.tool_calling import Tool

    class Args(BaseModel):
        city: str
        units: str = "c"

    async def handler(args: Args) -> str:
        return f"21 in {args.city}"

    return Tool(
        name=name,
        description="Current weather for a city.",
        parameters=Args,
        handler=handler,
    )


def build_prompt(
    instruction: str = "You are terse.", question: str = "Capital of France?"
) -> Any:
    """A rendered prompt with both an instruction and a message."""
    from base.models.definitions.message import MessageRole
    from base.models.view_models.prompt import (
        RenderedPrompt,
        RenderedPromptInstruction,
        RenderedPromptMessage,
    )

    return RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content=instruction)
        ],
        messages=[RenderedPromptMessage(role=MessageRole.USER, content=question)],
    )


def stored_message(role: Any = None, content: str = "hello", sequence: int = 1) -> Any:
    """One stored conversation message, with a stable conversation id."""
    import uuid

    from base.models.definitions.message import MessageRole
    from base.models.view_models.chat_history import StoredMessage

    return StoredMessage(
        conversation_id=uuid.UUID(int=1),
        sequence=sequence,
        role=role or MessageRole.USER,
        content=content,
    )


def subclasses_of(root: type) -> list[type]:
    """Every concrete subclass of an exception root, depth first.

    Error-handling tests walk an SDK's whole exception tree rather than naming
    a few classes, so a class the SDK adds later is covered the day it lands.
    """
    seen: list[type] = []

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.append(sub)
                walk(sub)

    walk(root)
    return seen


def try_build(cls: type) -> Exception | None:
    """Best-effort construction of an SDK exception.

    Their constructors disagree wildly, so each shape is tried in turn and a
    class that resists all of them is reported as unbuildable rather than
    failing the test that needed it.
    """
    attempts: list[Callable[[], Exception]] = [
        lambda: cls("boom"),
        lambda: cls(),
        lambda: cls("boom", None, None),
        lambda: cls("boom", 1, {}),
        lambda: cls("boom", {}),
    ]
    for attempt in attempts:
        try:
            built = attempt()
        except Exception:
            continue
        if isinstance(built, BaseException):
            return built  # type: ignore[return-value]
    return None
