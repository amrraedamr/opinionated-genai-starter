"""Gemini Live against a real endpoint, where there is one.

Skipped by default, and the reason is not an oversight. Gemini Live connects
to `/ws/google.cloud.aiplatform.../BidiGenerateContent`, a path the shared
gateway refuses with a 403 while proxying OpenAI's `/realtime` happily. The
credential it holds is the gateway's, not Google's, so there is nothing to
point these at here.

Set `GEMINI_LIVE_MODEL` with a Live-capable deployment to run them. Without
one the adapter rests on unit tests built from the SDK's own types, which
catch a shape change but not a gateway that will not carry the protocol.
"""

import asyncio
import os

import pytest

from base.models.definitions.model import ProviderType
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.realtime_session import (
    RealtimeOutput,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeTurnEnd,
)
from base.services.llm.realtime.gemini import GeminiLiveSession

LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_MODEL,
        reason="set GEMINI_LIVE_MODEL; the shared gateway 403s Gemini's Live path",
    ),
]


@pytest.mark.live
async def test_a_typed_question_is_answered() -> None:
    """The whole translation, end to end: config, send, receive, turn end."""
    session = GeminiLiveSession(
        RealtimeSessionConfig(model=LIVE_MODEL, instructions="Answer in one word."),
        ModelConfigurations(
            api_host=os.environ.get("GEMINI_LLM_ENDPOINT") or None,
            api_key=os.environ["GEMINI_API_KEY"],
            model_name=LIVE_MODEL,
            api_type=ProviderType.GEMINI,
        ),
    )
    await asyncio.wait_for(session.aconnect(), timeout=45)
    try:
        await session.asend(RealtimeTextInput(text="Name the capital of France."))

        events: list[RealtimeOutput] = []

        async def drain() -> None:
            async for event in session.areceive():
                events.append(event)
                if isinstance(event, RealtimeTurnEnd):
                    return

        await asyncio.wait_for(drain(), timeout=90)

        said = "".join(e.text for e in events if isinstance(e, RealtimeTextDelta))
        assert "Paris" in said
    finally:
        await session.aclose()
