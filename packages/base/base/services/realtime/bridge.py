"""A user on one socket, a model on another, and this in between.

Two conversations run at once and neither waits for the other. A person talks
over the model; the model asks for a tool mid-sentence; a browser disappears
without a close frame. Piping one into the other and reading them in turn
would serialise all of it and deadlock on the first overlap, so each
direction gets its own pump and either one ending ends the pair.

What passes through is checked. Text from a user is checked before the model
sees it, text from the model before the user does, and a tool's output before
the model is told what it returned. Audio is not: it is samples, not
sentences, and a guardrail reading it as text would either refuse everything
or pass everything. The transcript of that audio is checked, which is the
nearest honest thing and is what a reviewer reads later anyway.
"""

import asyncio
import base64
import time
from collections.abc import Sequence

from pydantic import ValidationError

from base.models.definitions.guardrails import RETRACTION_TEXT, GuardrailStage
from base.models.definitions.realtime import (
    BUDGET_EXHAUSTED_REASON,
    RealtimeCloseCode,
    RealtimeMessageType,
)
from base.models.definitions.realtime_session import REALTIME_USAGE_LABEL
from base.models.view_models.realtime import RealtimeEnvelope
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeError,
    RealtimeOutput,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.services.guardrails.factory import build_guarded_stream
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.guardrails.streaming import GuardedStream
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.logging.console import ConsoleLoggerService
from base.services.rate_limiting.llm_budget import LLMBudgetExhausted, TokenBudget
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.manager import ConnectionManager
from base.services.tool_calling.toolbox import Toolbox
from base.services.usage import UsageTracker
from base.tools.tracing import OptionalTelemetry


class RealtimeBridge:
    """Pipes one client socket to one provider session, both ways at once."""

    def __init__(
        self,
        connection: RealtimeConnection,
        session: AbstractRealtimeSession,
        toolbox: Toolbox | None = None,
        guardrails: GuardrailPipeline | None = None,
        manager: ConnectionManager | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        self._connection = connection
        self._session = session
        self._toolbox = toolbox
        self._guardrails = guardrails
        self._manager = manager
        self._budget = budget or TokenBudget()
        self._logger = ConsoleLoggerService()
        self._speaking = False
        self._turn_started = 0.0
        self._reply: GuardedStream | None = None
        self._tools: set[asyncio.Task[None]] = set()

    @property
    def speaking(self) -> bool:
        """Whether the model is mid-turn, which is what makes input a barge-in."""
        return self._speaking

    async def arun(self) -> None:
        """Carry the conversation until either side stops.

        One span covers the session rather than one per turn. A conversation
        is the unit someone investigating a complaint actually has: they know
        when a user called, not which of forty turns went wrong.
        """
        with OptionalTelemetry.continued_span(
            "realtime.session",
            None,
            kind="server",
            attributes={
                "realtime.model": self._session.model,
                "realtime.provider": self._session.get_service_name(),
            },
        ):
            try:
                await self._session.aconnect()
                async with asyncio.TaskGroup() as group:
                    group.create_task(self._apump_client())
                    group.create_task(self._apump_provider())
            except* Exception as errors:
                self._log_failure(errors.exceptions)
            finally:
                await self._aclose()

    async def _apump_client(self) -> None:
        """Client to model; delivery frames go to the manager instead."""
        while not self._connection.closed:
            frame = await self._connection.areceive()
            if frame is None:
                continue
            if self._manager is not None:
                frame = await self._manager.ahandle(self._connection, frame)
            if frame is None or frame.type != RealtimeMessageType.MESSAGE:
                continue
            await self._aforward(frame)

    async def _aforward(self, frame: RealtimeEnvelope) -> None:
        """Pass one client frame on, unless something refuses it."""
        if not await self._ahas_budget():
            return
        audio = frame.payload.get("audio")
        if isinstance(audio, str):
            try:
                chunk = RealtimeAudioInput(audio=audio)
            except ValidationError:
                await self._connection.asend(RealtimeEnvelope.error("Malformed audio"))
                return
            await self._session.asend(chunk)
            return

        text = frame.payload.get("text")
        if not isinstance(text, str) or not text:
            return

        allowed = await self._achecked(text, GuardrailStage.INPUT)
        if allowed is None:
            return
        if self._speaking:
            await self._session.ainterrupt()
            self._speaking = False
        await self._session.asend(RealtimeTextInput(text=allowed))

    async def _apump_provider(self) -> None:
        """Model to client, translating each event into a frame."""
        async for event in self._session.areceive():
            if self._connection.closed:
                return
            await self._adeliver(event)

    async def _adeliver(self, event: RealtimeOutput) -> None:
        match event:
            case RealtimeTextDelta():
                await self._asay(event.text)
            case RealtimeAudioDelta():
                self._begin_turn()
                await self._connection.asend(
                    RealtimeEnvelope(
                        type=RealtimeMessageType.MESSAGE,
                        payload={"audio": base64.b64encode(event.audio).decode()},
                    )
                )
            case RealtimeTranscriptDelta():
                await self._aheard(event.text)
            case RealtimeToolCall():
                self._run_tool(event)
            case RealtimeTurnEnd():
                self._end_turn(event)
                if event.usage is not None:
                    await self._budget.aspend(event.usage.total_tokens)
                await self._afinish_reply()
                await self._connection.asend(
                    RealtimeEnvelope(
                        type=RealtimeMessageType.MESSAGE,
                        payload={"turnComplete": True},
                    )
                )
            case RealtimeError():
                await self._connection.asend(RealtimeEnvelope.error(event.message))
                if not event.recoverable:
                    await self._connection.aclose(
                        RealtimeCloseCode.INTERNAL_ERROR, event.message
                    )

    async def _asay(self, text: str) -> None:
        """Send the model's words through this turn's guard; a block cuts the turn."""
        reply = self._turn_reply()
        if reply.blocked:
            return
        self._begin_turn()
        allowed = await reply.apush(text)
        if reply.blocked:
            await self._session.ainterrupt()
            self._speaking = False
            await self._connection.asend(RealtimeEnvelope.error(RETRACTION_TEXT))
            return
        await self._asend_text(allowed)

    async def _afinish_reply(self) -> None:
        """Send the held tail; retract the turn if the whole of it is refused."""
        reply, self._reply = self._reply, None
        if reply is None or reply.blocked:
            return
        await self._asend_text(await reply.aflush())
        if await reply.averdict() is not None:
            await self._connection.asend(RealtimeEnvelope.error(RETRACTION_TEXT))

    def _turn_reply(self) -> GuardedStream:
        if self._reply is None:
            self._reply = build_guarded_stream(self._guardrails or GuardrailPipeline())
        return self._reply

    async def _asend_text(self, text: str) -> None:
        if text:
            await self._connection.asend(
                RealtimeEnvelope(
                    type=RealtimeMessageType.MESSAGE, payload={"text": text}
                )
            )

    async def _aheard(self, text: str) -> None:
        """Report what the model heard, checked as the user's own words.

        The audio itself cannot be checked; this is the only readable form it
        ever takes, so refusing here is the one place a spoken secret can be
        caught before it is written into a transcript.
        """
        allowed = await self._achecked(text, GuardrailStage.INPUT)
        if allowed is None:
            return
        await self._connection.asend(
            RealtimeEnvelope(
                type=RealtimeMessageType.MESSAGE, payload={"transcript": allowed}
            )
        )

    async def _achecked(self, text: str, stage: GuardrailStage) -> str | None:
        """What survives the guardrails, or None when they refused it.

        A refusal tells the client and leaves the session open. Closing the
        socket over one blocked sentence would end a conversation that was
        otherwise fine, and the user would have no idea why.
        """
        if self._guardrails is None or not self._guardrails.applies_to(stage):
            return text
        report = await self._guardrails.arun(text, stage)
        if report.blocked:
            verdict = report.blocking_verdict
            await self._connection.asend(
                RealtimeEnvelope.error(
                    verdict.reason if verdict and verdict.reason else "Blocked"
                )
            )
            return None
        return report.content

    def _run_tool(self, call: RealtimeToolCall) -> None:
        """Run a tool without stopping the model's stream.

        Awaiting it here would hold up the only reader of the provider socket,
        so audio already generated for this turn would arrive late or not at
        all. The model is identified by call id and does not care in which
        order answers come back.
        """
        task = asyncio.create_task(self._ainvoke(call))
        self._tools.add(task)
        task.add_done_callback(self._tools.discard)

    async def _ainvoke(self, call: RealtimeToolCall) -> None:
        if self._toolbox is None:
            await self._session.asend_tool_result(
                call.call.id, "Error: this session has no tools"
            )
            return
        result = await self._toolbox.ainvoke(call.call)
        allowed = await self._achecked(result, GuardrailStage.TOOL_RESULT)
        await self._session.asend_tool_result(
            call.call.id, allowed if allowed is not None else "Error: blocked"
        )

    async def _ahas_budget(self) -> bool:
        """Whether the caller has tokens left; when not, say so and keep the socket open."""
        try:
            await self._budget.acheck()
        except LLMBudgetExhausted:
            await self._connection.asend(
                RealtimeEnvelope.error(BUDGET_EXHAUSTED_REASON)
            )
            return False
        return True

    def _begin_turn(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._turn_started = time.monotonic()

    def _end_turn(self, event: RealtimeTurnEnd) -> None:
        """Bill the turn as it ends.

        Per turn rather than per session: a conversation can stay open for an
        hour, and nothing that bills for it can wait that long to find out
        what it cost.
        """
        self._speaking = False
        if event.usage is None:
            return
        UsageTracker().record(
            model=self._session.model,
            label=REALTIME_USAGE_LABEL,
            input_tokens=event.usage.input_tokens,
            output_tokens=event.usage.output_tokens,
            duration_ms=int((time.monotonic() - self._turn_started) * 1000),
            cached_input_tokens=event.usage.cached_input_tokens,
            cache_write_tokens=event.usage.cache_write_tokens,
        )

    async def _aclose(self) -> None:
        """Let outstanding tools finish, then close both sides."""
        for task in list(self._tools):
            task.cancel()
        if self._tools:
            await asyncio.gather(*self._tools, return_exceptions=True)
        await self._session.aclose()
        await self._connection.aclose(RealtimeCloseCode.NORMAL, "Session ended")

    def _log_failure(self, errors: Sequence[Exception]) -> None:
        for error in errors:
            self._logger.warning(
                "Realtime bridge stopped: %s: %s", type(error).__name__, error
            )


__all__ = ["RealtimeBridge"]
