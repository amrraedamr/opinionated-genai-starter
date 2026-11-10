"""The moderation guardrail against the real endpoint.

The unit suite proves the threshold and category logic against a double. This
proves the two things a double cannot: that the request shape the SDK builds is
one the service accepts, and that benign text really does score below the line
while plainly violent text really does score above it.
"""

from typing import Any

import pytest

from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.models.definitions.model import ProviderType
from base.models.view_models.model import ModelConfigurations
from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails.llm.openai import OpenAIModerationGuardrail
from tests.conftest import MakeSettings, credential

pytestmark = pytest.mark.live

MODEL = "omni-moderation-latest"


def build(make_settings: MakeSettings, **settings: Any) -> OpenAIModerationGuardrail:
    make_settings(GuardrailSettingsMixin, moderation_model=MODEL, **settings)
    return OpenAIModerationGuardrail(
        model_config=ModelConfigurations(
            api_host=credential("OPENAI_LLM_ENDPOINT"),
            api_key=credential("OPENAI_LLM_API_KEY"),
            api_type=ProviderType.OPENAI,
        )
    )


async def test_benign_text_is_allowed(make_settings: MakeSettings) -> None:
    guardrail = build(make_settings)
    verdict = await guardrail.acheck(
        "What time does the library open on Saturday?", GuardrailStage.INPUT
    )
    assert verdict.outcome == GuardrailOutcome.ALLOW


async def test_the_endpoint_accepts_what_the_guardrail_sends(
    make_settings: MakeSettings,
) -> None:
    """A parameter the SDK types and the service rejects is the usual bug."""
    guardrail = build(make_settings)
    assert guardrail._scores(
        await guardrail._async_client.moderations.create(
            model=MODEL, input="hello there"
        )
    )


async def test_a_score_comes_back_for_every_category(
    make_settings: MakeSettings,
) -> None:
    """Thresholds are meaningless if the reply carries no scores."""
    guardrail = build(make_settings)
    response = await guardrail._async_client.moderations.create(
        model=MODEL, input="hello there"
    )
    scores = guardrail._scores(response)
    assert len(scores) > 3
    assert all(0.0 <= score <= 1.0 for score in scores.values())


async def test_plainly_violent_text_is_refused(make_settings: MakeSettings) -> None:
    """The one case worth paying for: the policy actually fires."""
    guardrail = build(make_settings, moderation_threshold=0.5)
    verdict = await guardrail.acheck(
        "I am going to find you and kill you with my bare hands.",
        GuardrailStage.INPUT,
    )
    assert verdict.outcome == GuardrailOutcome.BLOCK
    assert verdict.category


async def test_a_threshold_of_one_lets_almost_everything_through(
    make_settings: MakeSettings,
) -> None:
    """Proves the decision is the threshold's, not the endpoint's `flagged`."""
    guardrail = build(make_settings, moderation_threshold=1.01)
    verdict = await guardrail.acheck(
        "I am going to find you and kill you with my bare hands.",
        GuardrailStage.INPUT,
    )
    assert verdict.outcome == GuardrailOutcome.ALLOW


def test_both_counterparts_reach_the_endpoint(make_settings: MakeSettings) -> None:
    """The sync client is a separate client and is separately broken-able."""
    guardrail = build(make_settings)
    verdict = guardrail.check("Good morning.", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.ALLOW


async def test_scores_carry_the_api_category_names(make_settings: MakeSettings) -> None:
    """What a deployment writes in `moderation_categories` must match a key."""
    guardrail = build(make_settings)
    response = await guardrail._async_client.moderations.create(
        model=MODEL, input="hello there"
    )
    assert "self-harm" in guardrail._scores(response)
