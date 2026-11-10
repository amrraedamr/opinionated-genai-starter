"""Two conversations at once, and what happens where they collide.

The bridge is almost entirely about overlap: a person talking over the model,
a tool running while audio is still arriving, one side vanishing while the
other is mid-sentence. These drive both pumps against doubles and assert what
came out the far end.
"""

import asyncio
import base64
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from base.models.definitions.guardrails import (
    GuardrailOutcome,
    GuardrailStage,
)
from base.models.definitions.realtime import RealtimeMessageType
from base.models.view_models import BaseModel, JSONObject
from base.models.view_models.chat_history import (
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.guardrails import GuardrailVerdict
from base.models.view_models.rate_limiting import RateDecision
from base.models.view_models.realtime import ConnectionTicket, RealtimeEnvelope
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeError,
    RealtimeInput,
    RealtimeOutput,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.models.view_models.usage import ProviderUsage
from base.services.configuration.mixins import RateLimitSettingsMixin
from base.services.guardrails import AbstractGuardrail
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.rate_limiting.llm_budget import LLMBudgetExhausted, TokenBudget
from base.services.rate_limiting.memory import InMemoryRateLimiter
from base.services.realtime.bridge import RealtimeBridge
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.context import RealtimeContext
from base.services.tool_calling.toolbox import Tool, Toolbox
from base.services.usage import UsageTracker
from tests.conftest import MakeSettings


class FakeSocket:
    """A client that says what it was told to and records what it was sent."""

    def __init__(
        self, script: list[str] | None = None, after: asyncio.Event | None = None
    ) -> None:
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None
        self._script = list(script or [])
        self._after = after
        """Holds the client back until the model is actually talking.

        Barging in is about order. Without this the client frame can arrive
        first and there is nothing to interrupt, so the test would pass or
        fail on scheduling rather than on behaviour.
        """

    async def receive_text(self) -> str:
        if self._after is not None:
            await self._after.wait()
        if self._script:
            return self._script.pop(0)
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    def frames(self) -> list[RealtimeEnvelope]:
        return [RealtimeEnvelope.model_validate_json(raw) for raw in self.sent]

    def payloads(self, key: str) -> list[object]:
        return [f.payload[key] for f in self.frames() if key in f.payload]


class FakeSession(AbstractRealtimeSession):
    """A provider that emits a script and records what it was sent."""

    def __init__(
        self,
        script: list[RealtimeOutput] | None = None,
        speaking: asyncio.Event | None = None,
    ) -> None:
        self.speaking = speaking
        self.sent: list[RealtimeInput] = []
        self.tool_results: list[tuple[str, str]] = []
        self.interrupts = 0
        self.closed = False
        self.connected = False
        self._script = list(script or [])
        self._gate = asyncio.Event()

    @property
    def model(self) -> str:
        return "fake-realtime"

    @property
    def input_sample_rate(self) -> int:
        return 24_000

    def get_service_name(self) -> str:
        return "fake"

    async def aconnect(self) -> None:
        self.connected = True

    async def asend(self, value: RealtimeInput) -> None:
        self.sent.append(value)

    async def areceive(self) -> AsyncIterator[RealtimeOutput]:
        for event in self._script:
            yield event
            if self.speaking is not None:
                self.speaking.set()
            await asyncio.sleep(0)
        self._gate.set()
        await asyncio.sleep(3600)
        yield RealtimeTurnEnd()

    async def asend_tool_result(self, call_id: str, result: str) -> None:
        self.tool_results.append((call_id, result))

    async def ainterrupt(self) -> None:
        self.interrupts += 1

    async def aclose(self) -> None:
        self.closed = True


class Refuses(AbstractGuardrail):
    """Blocks anything containing a word, at the stages it was given."""

    def __init__(self, word: str, *stages: GuardrailStage) -> None:
        self._word = word
        self._at = frozenset(stages)

    def _name(self) -> str:
        return f"refuses-{self._word}"

    def _stages(self) -> frozenset[GuardrailStage]:
        return self._at

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        if self._word in content:
            return GuardrailVerdict(
                outcome=GuardrailOutcome.BLOCK, reason=f"no {self._word}"
            )
        return GuardrailVerdict(outcome=GuardrailOutcome.ALLOW, content=content)

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return self._check(content, stage)


class Redacts(AbstractGuardrail):
    """Replaces a word rather than refusing the whole message."""

    def _name(self) -> str:
        return "redacts"

    def _stages(self) -> frozenset[GuardrailStage]:
        return frozenset(
            {
                GuardrailStage.OUTPUT,
                GuardrailStage.OUTPUT_DELTA,
                GuardrailStage.TOOL_RESULT,
            }
        )

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return GuardrailVerdict(
            outcome=GuardrailOutcome.REDACT, content=content.replace("secret", "[gone]")
        )

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return self._check(content, stage)


class Args(BaseModel):
    pass


def ticket() -> ConnectionTicket:
    return ConnectionTicket(
        ticket="t",
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        subject="user-1",
        user_id=uuid4(),
        device_id="laptop",
    )


def says(**payload: object) -> str:
    return RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload=payload).jsonify()


@pytest.fixture
def settings(make_settings: MakeSettings) -> None:
    make_settings(RateLimitSettingsMixin)


def bridge(
    client: list[str] | None = None,
    provider: list[RealtimeOutput] | None = None,
    toolbox: Toolbox | None = None,
    guardrails: GuardrailPipeline | None = None,
    client_waits_for_model: bool = False,
) -> tuple[RealtimeBridge, FakeSocket, FakeSession]:
    speaking = asyncio.Event() if client_waits_for_model else None
    socket = FakeSocket(client, after=speaking)
    session = FakeSession(provider, speaking=speaking)
    connection = RealtimeConnection(
        socket,  # type: ignore[arg-type]
        ticket(),
        limiter=InMemoryRateLimiter(),
    )
    return RealtimeBridge(connection, session, toolbox, guardrails), socket, session


async def run(live: RealtimeBridge, seconds: float = 0.2) -> None:
    """Let both pumps run, then stop. Neither ends on its own."""
    task = asyncio.create_task(live.arun())
    await asyncio.sleep(seconds)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# Client to model


async def test_what_a_user_types_reaches_the_model(settings: None) -> None:
    live, _, session = bridge(client=[says(text="hello")])
    await run(live)
    assert [s.text for s in session.sent if isinstance(s, RealtimeTextInput)] == [
        "hello"
    ]


async def test_what_a_user_says_reaches_the_model_as_audio(settings: None) -> None:
    pcm = b"\x01\x02\x03"
    live, _, session = bridge(client=[says(audio=base64.b64encode(pcm).decode())])
    await run(live)
    assert [s.audio for s in session.sent if isinstance(s, RealtimeAudioInput)] == [pcm]


async def test_an_empty_frame_is_not_forwarded(settings: None) -> None:
    live, _, session = bridge(client=[says(text="")])
    await run(live)
    assert session.sent == []


# Model to client


async def test_the_models_words_reach_the_user(settings: None) -> None:
    live, socket, _ = bridge(provider=[RealtimeTextDelta(text="hi")])
    await run(live)
    assert socket.payloads("text") == ["hi"]


async def test_the_models_voice_reaches_the_user_as_base64(settings: None) -> None:
    """The frame carries text, so samples have to be encoded to survive it."""
    pcm = bytes(range(256))
    live, socket, _ = bridge(provider=[RealtimeAudioDelta(audio=pcm)])
    await run(live)
    assert socket.payloads("audio") == [base64.b64encode(pcm).decode()]


async def test_what_the_model_heard_is_reported_back(settings: None) -> None:
    live, socket, _ = bridge(provider=[RealtimeTranscriptDelta(text="I said this")])
    await run(live)
    assert socket.payloads("transcript") == ["I said this"]


async def test_the_end_of_a_turn_is_announced(settings: None) -> None:
    """A client needs to know it can speak, which silence does not tell it."""
    live, socket, _ = bridge(provider=[RealtimeTurnEnd()])
    await run(live)
    assert socket.payloads("turnComplete") == [True]


# Barging in


async def test_speaking_over_the_model_cuts_it_off(settings: None) -> None:
    """The point of the bridge being two pumps rather than one loop."""
    live, _, session = bridge(
        client=[says(text="actually, wait")],
        provider=[RealtimeTextDelta(text="let me explain at length")],
        client_waits_for_model=True,
    )
    await run(live)
    assert session.interrupts == 1


async def test_microphone_audio_does_not_cut_the_model_off(settings: None) -> None:
    """Audio streams continuously; the provider's own VAD decides a barge-in."""
    live, _, session = bridge(
        client=[says(audio=base64.b64encode(b"\x00\x01").decode())],
        provider=[RealtimeTextDelta(text="let me explain at length")],
        client_waits_for_model=True,
    )
    await run(live)
    assert session.interrupts == 0


async def test_malformed_audio_is_refused_without_ending_the_session(
    settings: None,
) -> None:
    live, socket, session = bridge(client=[says(audio="not base64!!"), says(text="hi")])
    await run(live)

    assert any(f.type == RealtimeMessageType.ERROR for f in socket.frames())
    assert [s.text for s in session.sent if isinstance(s, RealtimeTextInput)] == ["hi"]


async def test_the_provider_session_is_connected_before_it_is_read(
    settings: None,
) -> None:
    live, _, session = bridge(provider=[RealtimeTextDelta(text="hi")])
    await run(live)
    assert session.connected


async def test_input_before_the_model_speaks_interrupts_nothing(
    settings: None,
) -> None:
    """Interrupting a model that is not talking would cancel the next turn."""
    live, _, session = bridge(client=[says(text="hello")])
    await run(live)
    assert session.interrupts == 0


async def test_input_after_a_turn_ends_interrupts_nothing(settings: None) -> None:
    live, _, session = bridge(
        client=[says(text="next question")],
        provider=[RealtimeTextDelta(text="done"), RealtimeTurnEnd()],
    )
    await asyncio.sleep(0)
    await run(live)
    assert session.interrupts == 0


# Guardrails


async def test_a_refused_question_never_reaches_the_model(settings: None) -> None:
    live, socket, session = bridge(
        client=[says(text="my password is hunter2")],
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.INPUT)),
    )
    await run(live)
    assert session.sent == []
    assert any(f.type == RealtimeMessageType.ERROR for f in socket.frames())


async def test_a_refusal_leaves_the_conversation_open(settings: None) -> None:
    """Closing over one blocked sentence ends a conversation that was fine."""
    live, socket, _ = bridge(
        client=[says(text="my password is hunter2")],
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.INPUT)),
    )
    await run(live)
    assert socket.closed is None or socket.closed[0] == 1000


async def test_a_refused_answer_never_reaches_the_user(settings: None) -> None:
    live, socket, session = bridge(
        provider=[RealtimeTextDelta(text="the password is hunter2")],
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.OUTPUT_DELTA)),
    )
    await run(live)
    assert socket.payloads("text") == []
    assert session.interrupts == 1, "a refused answer should stop the rest of it"


async def test_a_word_split_across_deltas_is_still_refused(settings: None) -> None:
    live, socket, session = bridge(
        provider=[
            RealtimeTextDelta(text="the pass"),
            RealtimeTextDelta(text="word is hunter2"),
            RealtimeTurnEnd(),
        ],
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.OUTPUT_DELTA)),
    )
    await run(live)
    assert "password" not in "".join(str(text) for text in socket.payloads("text"))
    assert session.interrupts == 1


async def test_a_reply_refused_as_a_whole_is_retracted_at_turn_end(
    settings: None,
) -> None:
    """A check that can only judge the complete reply runs once it is complete."""
    live, socket, _ = bridge(
        provider=[RealtimeTextDelta(text="the password is hunter2"), RealtimeTurnEnd()],
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.OUTPUT)),
    )
    await run(live)
    errors = [f for f in socket.frames() if f.type == RealtimeMessageType.ERROR]
    assert errors, "the reply should be retracted"


async def test_a_redacted_answer_reaches_the_user_redacted(settings: None) -> None:
    live, socket, _ = bridge(
        provider=[RealtimeTextDelta(text="the secret is here"), RealtimeTurnEnd()],
        guardrails=GuardrailPipeline(Redacts()),
    )
    await run(live)
    assert socket.payloads("text") == ["the [gone] is here"]


async def test_spoken_words_are_checked_through_their_transcript(
    settings: None,
) -> None:
    """Audio cannot be read, and this is the only readable form it takes."""
    live, socket, _ = bridge(
        provider=[RealtimeTranscriptDelta(text="my password is hunter2")],
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.INPUT)),
    )
    await run(live)
    assert socket.payloads("transcript") == []


# Tools


def toolbox_of(answer: str = "sunny", delay: float = 0.0) -> Toolbox:
    async def handler(_: JSONObject) -> str:
        if delay:
            await asyncio.sleep(delay)
        return answer

    return Toolbox(
        Tool.from_json_schema(
            name="weather",
            description="the weather",
            schema={"type": "object", "properties": {}},
            handler=handler,
        )
    )


def tool_call(name: str = "weather") -> RealtimeToolCall:
    return RealtimeToolCall(
        call=StoredToolCall(
            id="call-1",
            function=StoredToolCallFunction(name=name, arguments="{}"),
        )
    )


async def test_a_tool_the_model_asked_for_is_run_and_answered(
    settings: None,
) -> None:
    live, _, session = bridge(provider=[tool_call()], toolbox=toolbox_of())
    await run(live)
    assert session.tool_results == [("call-1", "sunny")]


async def test_a_slow_tool_does_not_hold_up_the_models_voice(
    settings: None,
) -> None:
    """Awaiting it inline would stop the only reader of the provider stream,
    so audio already generated for this turn would arrive late or not at all."""
    live, socket, _ = bridge(
        provider=[tool_call(), RealtimeAudioDelta(audio=b"\x01")],
        toolbox=toolbox_of(delay=0.15),
    )
    await run(live, seconds=0.05)
    assert socket.payloads("audio"), "audio waited behind the tool"


async def test_a_tool_result_is_checked_before_the_model_sees_it(
    settings: None,
) -> None:
    live, _, session = bridge(
        provider=[tool_call()],
        toolbox=toolbox_of(answer="the secret is hunter2"),
        guardrails=GuardrailPipeline(Redacts()),
    )
    await run(live)
    assert session.tool_results == [("call-1", "the [gone] is hunter2")]


async def test_a_blocked_tool_result_is_answered_rather_than_dropped(
    settings: None,
) -> None:
    """Silence would leave the model waiting for an answer that never comes."""
    live, _, session = bridge(
        provider=[tool_call()],
        toolbox=toolbox_of(answer="password: hunter2"),
        guardrails=GuardrailPipeline(Refuses("password", GuardrailStage.TOOL_RESULT)),
    )
    await run(live)
    assert len(session.tool_results) == 1
    assert "blocked" in session.tool_results[0][1].lower()


async def test_a_session_without_tools_says_so_rather_than_hanging(
    settings: None,
) -> None:
    live, _, session = bridge(provider=[tool_call()])
    await run(live)
    assert len(session.tool_results) == 1
    assert "no tools" in session.tool_results[0][1]


async def test_an_unknown_tool_is_reported_to_the_model(settings: None) -> None:
    live, _, session = bridge(provider=[tool_call("nope")], toolbox=toolbox_of())
    await run(live)
    assert "no tool named" in session.tool_results[0][1]


# Errors and shutdown


async def test_a_recoverable_error_is_passed_on_without_closing(
    settings: None,
) -> None:
    live, socket, _ = bridge(
        provider=[RealtimeError(message="rate limited", recoverable=True)]
    )
    await run(live)
    assert any(f.type == RealtimeMessageType.ERROR for f in socket.frames())
    assert socket.closed is None or socket.closed[0] == 1000


async def test_an_unrecoverable_error_closes_the_socket(settings: None) -> None:
    live, socket, _ = bridge(
        provider=[RealtimeError(message="session died", recoverable=False)]
    )
    await run(live)
    assert socket.closed is not None
    assert socket.closed[0] != 1000


async def test_the_provider_session_is_closed_when_the_bridge_stops(
    settings: None,
) -> None:
    """A forgotten provider socket bills for as long as it stays open."""
    live, _, session = bridge()
    await run(live, seconds=0.05)
    assert session.closed


# Usage


async def test_a_turn_is_billed_as_it_ends(settings: None) -> None:
    """Per turn, not per session: a conversation can last an hour.

    Run inside a connection scope, as a route does. `UsageTracker` is stored
    per scope and a task only inherits a copy of the context, so a bridge
    started without one would bill into a scope nobody ever reads.
    """
    live, _, _ = bridge(
        provider=[
            RealtimeTextDelta(text="hi"),
            RealtimeTurnEnd(usage=ProviderUsage(input_tokens=11, output_tokens=5)),
        ]
    )
    async with RealtimeContext.connection(ticket()):
        await run(live)
        records = [r for r in UsageTracker().records if r.label == "realtime"]

    assert records
    assert records[-1].input_tokens == 11


async def test_a_turn_without_usage_is_not_billed_as_zero(settings: None) -> None:
    """A zero-cost record would read as a free turn rather than an unknown one."""
    live, _, _ = bridge(provider=[RealtimeTurnEnd()])
    async with RealtimeContext.connection(ticket()):
        await run(live)
        billed = [r for r in UsageTracker().records if r.label == "realtime"]

    assert billed == []


class Budget(TokenBudget):
    """A token budget a test can exhaust, recording what was charged."""

    def __init__(self, exhausted: bool = False) -> None:
        super().__init__(InMemoryRateLimiter())
        self.exhausted = exhausted
        self.charged: list[int] = []

    async def acheck(self) -> None:
        if self.exhausted:
            raise LLMBudgetExhausted(
                RateDecision(allowed=False, remaining=0, retry_after_seconds=10)
            )

    async def aspend(self, tokens: int) -> None:
        self.charged.append(tokens)


def budgeted(
    budget: TokenBudget,
    client: list[str] | None = None,
    provider: list[RealtimeOutput] | None = None,
) -> tuple[RealtimeBridge, FakeSocket, FakeSession]:
    socket = FakeSocket(client)
    session = FakeSession(provider)
    connection = RealtimeConnection(
        socket,  # type: ignore[arg-type]
        ticket(),
        limiter=InMemoryRateLimiter(),
    )
    return RealtimeBridge(connection, session, budget=budget), socket, session


async def test_a_turns_tokens_are_charged_to_the_budget(settings: None) -> None:
    """A socket used to spend tokens without the budget ever hearing of it."""
    budget = Budget()
    live, _, _ = budgeted(
        budget,
        provider=[
            RealtimeTurnEnd(usage=ProviderUsage(input_tokens=30, output_tokens=12))
        ],
    )
    await run(live)
    assert budget.charged == [42]


async def test_an_exhausted_budget_drops_input_but_keeps_the_socket(
    settings: None,
) -> None:
    budget = Budget(exhausted=True)
    live, socket, session = budgeted(budget, client=[says(text="hello")])
    await run(live)
    assert session.sent == []
    assert any(f.type == RealtimeMessageType.ERROR for f in socket.frames())
    assert socket.closed is None or socket.closed[0] == 1000
