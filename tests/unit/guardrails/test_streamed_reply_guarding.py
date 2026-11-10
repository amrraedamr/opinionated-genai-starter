"""A streamed reply is guarded on its way to the caller, not only once it has finished."""

import pytest

from base.models.definitions.guardrails import RETRACTION_CODE, GuardrailStage
from base.models.view_models.guardrails import GuardrailVerdict
from base.models.view_models.llm_result import LLMResult
from base.models.view_models.streams import StreamEnd, StreamError, StreamMessage
from base.services.agents import AbstractAgent
from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails import AbstractGuardrail, GuardrailBlocked
from base.services.guardrails.factory import build_guardrail_pipeline
from base.services.guardrails.pipeline import GuardrailPipeline
from base.tools.stream import StreamIterator
from tests.conftest import MakeSettings
from tests.unit.agents.fake_llm import RecordingLLM, build_agent

type Frame = StreamMessage[str] | StreamEnd | StreamError


class StreamingLLM(RecordingLLM):
    """Streams its chunks into the iterator it is handed, as a provider adapter does."""

    chunks: tuple[str, ...] = ()

    async def acomplete(self, **kwargs: object) -> LLMResult:
        iterator = kwargs["stream_iterator"]
        assert isinstance(iterator, StreamIterator)
        for chunk in self.chunks:
            await iterator.asend(StreamMessage(data=chunk))
        await iterator.asend(StreamEnd())
        return LLMResult(content="".join(self.chunks))


class RefusesWhole(AbstractGuardrail):
    """Judges only a complete reply, the way moderation does."""

    def _name(self) -> str:
        return "refuses-whole"

    def _stages(self) -> frozenset[GuardrailStage]:
        return frozenset({GuardrailStage.OUTPUT})

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        if "forbidden" in content:
            return GuardrailVerdict.block("refused", category="test")
        return GuardrailVerdict.allow()

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return self._check(content, stage)


def streaming(*chunks: str) -> tuple[AbstractAgent, StreamIterator]:
    agent = build_agent(LLM_SERVICE=type("Script", (StreamingLLM,), {"chunks": chunks}))
    iterator = StreamIterator()
    agent._execution_context.streaming = True
    agent._execution_context.stream_iterator = iterator
    return agent, iterator


def received(iterator: StreamIterator) -> list[Frame]:
    """Every frame the caller would read, in order."""
    queue = iterator._queue
    assert queue is not None
    frames: list[Frame] = []
    while not queue.empty():
        frame = queue.get_nowait()
        assert isinstance(frame, StreamMessage | StreamEnd | StreamError)
        frames.append(frame)
    return frames


def text(frames: list[Frame]) -> str:
    return "".join(f.data for f in frames if isinstance(f, StreamMessage))


@pytest.fixture
def guarded(make_settings: MakeSettings) -> None:
    make_settings(GuardrailSettingsMixin, guardrails_enabled=True)


async def test_a_card_split_across_chunks_never_reaches_the_caller(
    guarded: None,
) -> None:
    agent, iterator = streaming("your card is 4111 1111 ", "1111 1111, thanks")
    result = await agent._acomplete()

    frames = received(iterator)
    assert text(frames) == "your card is [redacted:card], thanks"
    assert isinstance(frames[-1], StreamEnd)
    assert result.content == "your card is [redacted:card], thanks"


async def test_a_reply_refused_as_a_whole_ends_in_a_retraction(
    make_settings: MakeSettings,
) -> None:
    make_settings(GuardrailSettingsMixin, guardrails_enabled=True)
    agent, iterator = streaming("this is ", "forbidden")
    agent._guardrail_pipeline = GuardrailPipeline(RefusesWhole())

    with pytest.raises(GuardrailBlocked):
        await agent._acomplete()

    frames = received(iterator)
    assert isinstance(frames[-1], StreamError)
    assert frames[-1].error_code == RETRACTION_CODE
    assert not any(isinstance(f, StreamEnd) for f in frames)


async def test_nothing_changes_while_guardrails_are_off(
    make_settings: MakeSettings,
) -> None:
    make_settings(GuardrailSettingsMixin)
    agent, iterator = streaming("card 4111 1111 ", "1111 1111")
    assert not build_guardrail_pipeline().applies_to(GuardrailStage.OUTPUT_DELTA)

    await agent._acomplete()

    assert text(received(iterator)) == "card 4111 1111 1111 1111"
