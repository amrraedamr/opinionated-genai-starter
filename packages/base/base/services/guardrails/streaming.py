"""Guarding a streamed reply: chunks as they go, the whole at the end."""

from collections.abc import Iterator

from base.models.definitions.guardrails import (
    RETRACTION_CODE,
    RETRACTION_TEXT,
    STREAM_CUT_ATTEMPTS,
    STREAM_HELD_LIMIT,
    GuardrailStage,
)
from base.models.view_models.guardrails import GuardrailReport
from base.models.view_models.streams import (
    AnyStreamMessage,
    StreamEnd,
    StreamError,
    StreamMessage,
)
from base.services.guardrails.pipeline import GuardrailPipeline
from base.tools.stream import StreamIterator


class GuardedStream:
    """Chunks checked at `OUTPUT_DELTA` with a held tail; the whole at `OUTPUT`."""

    def __init__(self, guardrails: GuardrailPipeline, tail: int) -> None:
        self._guardrails = guardrails
        self._delta = guardrails.applies_to(GuardrailStage.OUTPUT_DELTA)
        self._whole = guardrails.applies_to(GuardrailStage.OUTPUT)
        self._tail = max(tail, 0)
        self._held = ""
        self._assembled: list[str] = []
        self._blocked_by: GuardrailReport | None = None
        self._judgment: GuardrailReport | None = None

    @property
    def assembled(self) -> str:
        """Everything emitted so far, as the caller received it."""
        return "".join(self._assembled)

    @property
    def active(self) -> bool:
        """Whether any guardrail checks a reply, so a caller can skip the work."""
        return self._delta or self._whole

    @property
    def blocked(self) -> bool:
        return self._blocked_by is not None

    @property
    def content(self) -> str:
        """The reply after the whole-text check, or as emitted before it runs."""
        return self._judgment.content if self._judgment is not None else self.assembled

    def push(self, chunk: str) -> str:
        """The checked text that may go out now; the tail waits for more."""
        if self.blocked or not chunk:
            return ""
        if not self._delta:
            self._assembled.append(chunk)
            return chunk
        pending = self._held + chunk
        whole = self._guardrails.run(pending, GuardrailStage.OUTPUT_DELTA)
        if whole.blocked or self._tail == 0:
            self._held = ""
            return self._emitted(whole)
        for cut in self._cuts(pending):
            head = self._guardrails.run(pending[:cut], GuardrailStage.OUTPUT_DELTA)
            rest = self._guardrails.run(pending[cut:], GuardrailStage.OUTPUT_DELTA)
            if self._agrees(head, rest, whole):
                self._held = pending[cut:]
                return self._emitted(head)
        return self._overflow(pending, whole)

    async def apush(self, chunk: str) -> str:
        """The checked text that may go out now; the tail waits for more, asynchronously."""
        if self.blocked or not chunk:
            return ""
        if not self._delta:
            self._assembled.append(chunk)
            return chunk
        pending = self._held + chunk
        whole = await self._guardrails.arun(pending, GuardrailStage.OUTPUT_DELTA)
        if whole.blocked or self._tail == 0:
            self._held = ""
            return self._emitted(whole)
        for cut in self._cuts(pending):
            head = await self._guardrails.arun(
                pending[:cut], GuardrailStage.OUTPUT_DELTA
            )
            rest = await self._guardrails.arun(
                pending[cut:], GuardrailStage.OUTPUT_DELTA
            )
            if self._agrees(head, rest, whole):
                self._held = pending[cut:]
                return self._emitted(head)
        return self._overflow(pending, whole)

    def flush(self) -> str:
        """The held tail, checked, now that nothing more is coming."""
        remainder, self._held = self._held, ""
        if self.blocked or not remainder:
            return ""
        return self._emitted(
            self._guardrails.run(remainder, GuardrailStage.OUTPUT_DELTA)
        )

    async def aflush(self) -> str:
        """The held tail, checked, now that nothing more is coming, asynchronously."""
        remainder, self._held = self._held, ""
        if self.blocked or not remainder:
            return ""
        report = await self._guardrails.arun(remainder, GuardrailStage.OUTPUT_DELTA)
        return self._emitted(report)

    def verdict(self) -> GuardrailReport | None:
        """The block that ended the reply; judges the whole once, after `flush`."""
        if self._judgment is None and not self.blocked and self._whole:
            self._judgment = self._guardrails.run(self.assembled, GuardrailStage.OUTPUT)
            if self._judgment.blocked:
                self._blocked_by = self._judgment
        return self._blocked_by

    async def averdict(self) -> GuardrailReport | None:
        """The block that ended the reply; judges the whole once, after `aflush`, asynchronously."""
        if self._judgment is None and not self.blocked and self._whole:
            self._judgment = await self._guardrails.arun(
                self.assembled, GuardrailStage.OUTPUT
            )
            if self._judgment.blocked:
                self._blocked_by = self._judgment
        return self._blocked_by

    @staticmethod
    def retraction() -> StreamError:
        """Says the reply is withdrawn, without saying which rule caught it."""
        return StreamError(error_code=RETRACTION_CODE, text=RETRACTION_TEXT)

    def _emitted(self, report: GuardrailReport) -> str:
        if report.blocked:
            self._blocked_by = report
            self._held = ""
            return ""
        self._assembled.append(report.content)
        return report.content

    def _cuts(self, pending: str) -> Iterator[int]:
        """Positions after whitespace at least `tail` from the end, latest first."""
        limit = len(pending) - self._tail
        tried = 0
        index = limit
        while index > 0 and tried < STREAM_CUT_ATTEMPTS:
            index = max(pending.rfind(" ", 0, index), pending.rfind("\n", 0, index))
            if index < 0:
                return
            tried += 1
            yield index + 1

    @staticmethod
    def _agrees(
        head: GuardrailReport, rest: GuardrailReport, whole: GuardrailReport
    ) -> bool:
        """A cut is clean when the halves check the way the whole does."""
        if head.blocked or rest.blocked:
            return False
        return head.content + rest.content == whole.content

    def _overflow(self, pending: str, whole: GuardrailReport) -> str:
        """Hold on, unless no clean cut has turned up in far too much text."""
        if len(pending) <= STREAM_HELD_LIMIT:
            self._held = pending
            return ""
        self._held = ""
        return self._emitted(whole)


class GuardedStreamIterator(StreamIterator):
    """Passes a provider's stream to `inner` through a guard; a block retracts it."""

    def __init__(self, inner: StreamIterator, guard: GuardedStream) -> None:
        super().__init__()
        self._inner = inner
        self._guard = guard
        self._finished = False
        self._retracted = False

    @property
    def guard(self) -> GuardedStream:
        return self._guard

    async def __anext__(self) -> dict[str, str | int] | str:
        return await self._inner.__anext__()

    def send(self, value: AnyStreamMessage) -> None:
        if isinstance(value, StreamMessage) and isinstance(value.data, str):
            self._forward(value, self._guard.push(value.data))
        elif isinstance(value, StreamEnd):
            self.finish(value)
        else:
            self._finished = self._finished or isinstance(value, StreamError)
            self._inner.send(value)

    async def asend(self, value: AnyStreamMessage) -> None:
        if isinstance(value, StreamMessage) and isinstance(value.data, str):
            await self._aforward(value, await self._guard.apush(value.data))
        elif isinstance(value, StreamEnd):
            await self.afinish(value)
        else:
            self._finished = self._finished or isinstance(value, StreamError)
            await self._inner.asend(value)

    def end(self, end_stream_model: StreamEnd | None = None) -> None:
        self.finish(end_stream_model)

    async def aend(self, end_stream_model: StreamEnd | None = None) -> None:
        await self.afinish(end_stream_model)

    def error(self, error_stream_model: StreamError | None = None) -> None:
        self._finished = True
        self._inner.error(error_stream_model)

    async def aerror(self, error_stream_model: StreamError | None = None) -> None:
        self._finished = True
        await self._inner.aerror(error_stream_model)

    def finish(self, end: StreamEnd | None = None) -> GuardrailReport | None:
        """Send the held tail, then `end` or a retraction; returns the verdict."""
        if not self._finished:
            self._finished = True
            rest = self._guard.flush()
            if rest:
                self._inner.send(StreamMessage(data=rest))
            if self._guard.verdict() is not None:
                self._retract()
            elif end is not None:
                self._inner.send(end)
        return self._guard.verdict()

    async def afinish(self, end: StreamEnd | None = None) -> GuardrailReport | None:
        """Send the held tail, then `end` or a retraction; returns the verdict, asynchronously."""
        if not self._finished:
            self._finished = True
            rest = await self._guard.aflush()
            if rest:
                await self._inner.asend(StreamMessage(data=rest))
            if await self._guard.averdict() is not None:
                await self._aretract()
            elif end is not None:
                await self._inner.asend(end)
        return await self._guard.averdict()

    def _forward(self, message: StreamMessage[str], text: str) -> None:
        if text:
            self._inner.send(message.model_copy(update={"data": text}))
        elif self._guard.blocked:
            self._retract()

    async def _aforward(self, message: StreamMessage[str], text: str) -> None:
        if text:
            await self._inner.asend(message.model_copy(update={"data": text}))
        elif self._guard.blocked:
            await self._aretract()

    def _retract(self) -> None:
        if not self._retracted:
            self._retracted = True
            self._inner.send(GuardedStream.retraction())

    async def _aretract(self) -> None:
        if not self._retracted:
            self._retracted = True
            await self._inner.asend(GuardedStream.retraction())


__all__ = ["GuardedStream", "GuardedStreamIterator"]
