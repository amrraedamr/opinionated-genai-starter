"""Checking a reply that is already going out, chunk by chunk.

The case worth the machinery is a value split across a chunk boundary: each
chunk is harmless on its own and the pair is not. Holding a tail is what
catches it, and holding rather than emitting-then-correcting is the point —
a caller that has already rendered a card number cannot un-see it.
"""

import pytest

from base.models.definitions.guardrails import RETRACTION_CODE, GuardrailStage
from base.models.view_models.guardrails import GuardrailVerdict
from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails import AbstractGuardrail
from base.services.guardrails.pii import PatternPiiGuardrail
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.guardrails.streaming import GuardedStream
from tests.conftest import MakeSettings

CARD = "4111111111111111"


class Blocker(AbstractGuardrail):
    """Refuses a word; at `OUTPUT` alone it can only judge the whole reply."""

    def __init__(self, word: str, *stages: GuardrailStage) -> None:
        self.word = word
        self._at = frozenset(stages or {GuardrailStage.OUTPUT})

    def _name(self) -> str:
        return "blocker"

    def _stages(self) -> frozenset[GuardrailStage]:
        return self._at

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        if self.word in content:
            return GuardrailVerdict.block("refused", category="test")
        return GuardrailVerdict.allow()

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return self._check(content, stage)


@pytest.fixture
def pii(make_settings: MakeSettings) -> GuardrailPipeline:
    make_settings(GuardrailSettingsMixin)
    return GuardrailPipeline(PatternPiiGuardrail())


async def drain(stream: GuardedStream, chunks: list[str]) -> str:
    """Everything a caller would have received, in order."""
    out = [await stream.apush(chunk) for chunk in chunks]
    out.append(await stream.aflush())
    return "".join(out)


async def test_a_card_split_across_chunks_is_still_redacted(
    pii: GuardrailPipeline,
) -> None:
    """Neither half is a card. Together they are, and the tail is what sees it."""
    stream = GuardedStream(pii, tail=32)
    emitted = await drain(stream, ["your card is 41111111", "11111111 ok"])
    assert CARD not in emitted
    assert "[redacted:card]" in emitted


async def test_a_card_split_across_three_chunks_is_still_redacted(
    pii: GuardrailPipeline,
) -> None:
    stream = GuardedStream(pii, tail=32)
    emitted = await drain(stream, ["card 4111", "1111", "11111111 done"])
    assert CARD not in emitted


async def test_an_unbroken_card_is_redacted(pii: GuardrailPipeline) -> None:
    stream = GuardedStream(pii, tail=8)
    emitted = await drain(stream, [f"card {CARD} ok"])
    assert CARD not in emitted


async def test_ordinary_text_passes_through_unchanged(
    pii: GuardrailPipeline,
) -> None:
    stream = GuardedStream(pii, tail=8)
    chunks = ["the weather ", "in Cairo is ", "31 degrees"]
    assert await drain(stream, chunks) == "".join(chunks)


async def test_nothing_is_lost_when_the_stream_ends(
    pii: GuardrailPipeline,
) -> None:
    """The held tail must be flushed, or the last characters vanish."""
    stream = GuardedStream(pii, tail=32)
    assert await drain(stream, ["short"]) == "short"


async def test_the_tail_is_held_back_rather_than_emitted_early(
    pii: GuardrailPipeline,
) -> None:
    """Cut at whitespace before the tail, so no word is split across chunks."""
    stream = GuardedStream(pii, tail=10)
    first = await stream.apush("alpha bravo charlie delta")
    assert first == "alpha bravo "
    assert await stream.aflush() == "charlie delta"


async def test_a_token_longer_than_the_window_is_held_whole(
    pii: GuardrailPipeline,
) -> None:
    stream = GuardedStream(pii, tail=10)
    assert await stream.apush("abcdefghijklmnopqrst") == ""
    assert await stream.aflush() == "abcdefghijklmnopqrst"


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7, 13])
async def test_a_value_split_across_chunks_is_still_redacted(
    pii: GuardrailPipeline, size: int
) -> None:
    """The split the hold-back exists for; small chunks are what streams send."""
    text = "card 4111111111111111 and mail jane.doe@example.com done."
    stream = GuardedStream(pii, tail=32)
    out = [await stream.apush(text[i : i + size]) for i in range(0, len(text), size)]
    out.append(await stream.aflush())
    joined = "".join(out)

    assert "4111" not in joined
    assert "jane.doe" not in joined
    assert joined.count("[redacted:") == 2


async def test_a_block_mid_stream_stops_everything_after_it() -> None:
    blocker = Blocker("forbidden", GuardrailStage.OUTPUT_DELTA)
    stream = GuardedStream(GuardrailPipeline(blocker), tail=0)
    assert await stream.apush("the forbidden word is here, ") == ""
    assert await stream.apush("and the rest is fine text") == ""
    assert await stream.averdict() is not None


async def test_no_tail_emits_each_chunk_immediately(pii: GuardrailPipeline) -> None:
    stream = GuardedStream(pii, tail=0)
    assert await stream.apush("hello ") == "hello "


async def test_the_assembled_reply_is_what_the_caller_received(
    pii: GuardrailPipeline,
) -> None:
    stream = GuardedStream(pii, tail=4)
    emitted = await drain(stream, ["hello ", "there"])
    assert stream.assembled == emitted


async def test_a_block_at_the_end_produces_a_retraction() -> None:
    """Some things can only be judged on the whole text."""
    stream = GuardedStream(GuardrailPipeline(Blocker("forbidden")), tail=4)
    await drain(stream, ["this is ", "forbidden"])

    verdict = await stream.averdict()
    assert verdict is not None
    assert verdict.blocked

    retraction = GuardedStream.retraction()
    assert retraction.error_code == RETRACTION_CODE
    assert retraction.text


async def test_a_retraction_names_no_guardrail() -> None:
    """Saying which rule caught it tells someone how to probe the policy."""
    assert "blocker" not in (GuardedStream.retraction().text or "")


async def test_a_clean_reply_produces_no_retraction(
    pii: GuardrailPipeline,
) -> None:
    stream = GuardedStream(pii, tail=4)
    await drain(stream, ["all fine here"])
    assert await stream.averdict() is None


async def test_a_redacted_reply_is_not_retracted(pii: GuardrailPipeline) -> None:
    """It was already cleaned on the way out; retracting it would be wrong."""
    stream = GuardedStream(pii, tail=32)
    await drain(stream, [f"card {CARD} ok"])
    assert await stream.averdict() is None


async def test_an_empty_pipeline_does_nothing(make_settings: MakeSettings) -> None:
    """Not configured must mean not in the way."""
    make_settings(GuardrailSettingsMixin)
    stream = GuardedStream(GuardrailPipeline(), tail=32)
    assert not stream.active
    chunks = [f"card {CARD}", " and more"]
    assert await drain(stream, chunks) == "".join(chunks)
    assert await stream.averdict() is None


async def test_an_empty_chunk_is_harmless(pii: GuardrailPipeline) -> None:
    stream = GuardedStream(pii, tail=4)
    assert await stream.apush("") == ""
    assert await drain(stream, ["ok"]) == "ok"


async def test_a_zero_tail_still_redacts_what_it_can_see(
    pii: GuardrailPipeline,
) -> None:
    """Switching the buffer off must not switch redaction off with it."""
    stream = GuardedStream(pii, tail=0)
    emitted = await drain(stream, [f"card {CARD} ok"])
    assert CARD not in emitted


async def test_a_check_on_the_whole_reply_never_sees_a_fragment() -> None:
    """Moderation bills per call and cannot judge half a sentence."""
    stream = GuardedStream(GuardrailPipeline(Blocker("forbidden")), tail=0)
    assert await stream.apush("the forbidden word") == "the forbidden word"
    assert await stream.averdict() is not None


async def test_a_value_split_anywhere_is_redacted_as_the_whole_would_be(
    pii: GuardrailPipeline,
) -> None:
    """Spaced IBANs and grouped cards have whitespace inside, where cuts land."""
    text = "pay GB82 WEST 1234 5698 7654 32 or card 4111 1111 1111 1111 today"
    expected = pii.run(text, GuardrailStage.OUTPUT_DELTA).content
    for size in (1, 2, 3, 5, 7, 11):
        stream = GuardedStream(pii, tail=48)
        chunks = [text[i : i + size] for i in range(0, len(text), size)]
        assert await drain(stream, chunks) == expected, f"chunks of {size}"


def test_the_sync_counterpart_redacts_the_same_way(pii: GuardrailPipeline) -> None:
    stream = GuardedStream(pii, tail=32)
    out = [stream.push(chunk) for chunk in ["card 4111 1111 ", "1111 1111 ok"]]
    out.append(stream.flush())
    assert "".join(out) == "card [redacted:card] ok"
    assert stream.verdict() is None
