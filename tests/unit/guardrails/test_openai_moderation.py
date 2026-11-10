"""The moderation guardrail, without spending money to find out what it does.

The endpoint is replaced with a double that records what it was asked and
answers with scores the test chooses, so threshold and category behaviour can
be pinned exactly. `tests/live` calls the real endpoint.
"""

from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError
from openai.types.moderation import CategoryScores
from openai.types.moderation_create_response import ModerationCreateResponse

from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.models.definitions.model import ProviderType
from base.models.view_models.model import ModelConfigurations
from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails.llm.openai import OpenAIModerationGuardrail
from tests.conftest import MakeSettings

REQUEST = httpx.Request("POST", "https://moderation.invalid/moderations")
CATEGORIES = [
    field.alias or name for name, field in CategoryScores.model_fields.items()
]


def scores(**over: float) -> ModerationCreateResponse:
    """A real SDK reply, scored by API category name (`self-harm`, not `self_harm`)."""
    return ModerationCreateResponse.model_validate(
        {
            "id": "modr-test",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": False,
                    "categories": dict.fromkeys(CATEGORIES, False),
                    "category_scores": dict.fromkeys(CATEGORIES, 0.0) | over,
                    "category_applied_input_types": {c: ["text"] for c in CATEGORIES},
                }
            ],
        }
    )


class Endpoint:
    """Records every call, so a test can assert what was actually sent."""

    def __init__(self, response: ModerationCreateResponse) -> None:
        self.response = response
        self.inputs: list[str] = []
        self.models: list[str] = []

    def create(self, *, model: str, input: str) -> ModerationCreateResponse:
        self.models.append(model)
        self.inputs.append(input)
        return self.response


class AsyncEndpoint(Endpoint):
    """The same recorder, awaitable, for the async client."""

    async def create(self, *, model: str, input: str) -> ModerationCreateResponse:  # type: ignore[override]
        return super().create(model=model, input=input)


def attach(guardrail: OpenAIModerationGuardrail, endpoint: Endpoint) -> Endpoint:
    """Point the sync client at the double."""
    guardrail._client.moderations = endpoint  # type: ignore[assignment]
    return endpoint


def attach_async(
    guardrail: OpenAIModerationGuardrail, endpoint: AsyncEndpoint
) -> AsyncEndpoint:
    """Point the async client at the awaitable double."""
    guardrail._async_client.moderations = endpoint  # type: ignore[assignment]
    return endpoint


def build(make_settings: MakeSettings, **settings: Any) -> OpenAIModerationGuardrail:
    make_settings(GuardrailSettingsMixin, **settings)
    return OpenAIModerationGuardrail(
        model_config=ModelConfigurations(
            api_host="https://moderation.invalid/",
            api_key="unused",
            api_type=ProviderType.OPENAI,
        )
    )


def test_clean_content_is_allowed(make_settings: MakeSettings) -> None:
    guardrail = build(make_settings)
    attach(guardrail, Endpoint(scores()))
    verdict = guardrail.check("hello there", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.ALLOW


def test_a_score_over_the_threshold_blocks(make_settings: MakeSettings) -> None:
    guardrail = build(make_settings, moderation_threshold=0.5)
    attach(guardrail, Endpoint(scores(violence=0.9)))
    verdict = guardrail.check("something", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.BLOCK
    assert verdict.category == "violence"


def test_a_score_under_the_threshold_is_allowed(make_settings: MakeSettings) -> None:
    guardrail = build(make_settings, moderation_threshold=0.9)
    attach(guardrail, Endpoint(scores(violence=0.6)))
    assert (
        guardrail.check("something", GuardrailStage.INPUT).outcome
        == GuardrailOutcome.ALLOW
    )


def test_the_threshold_is_inclusive(make_settings: MakeSettings) -> None:
    """A score exactly at the configured line counts as crossing it."""
    guardrail = build(make_settings, moderation_threshold=0.5)
    attach(guardrail, Endpoint(scores(hate=0.5)))
    assert guardrail.check("x", GuardrailStage.INPUT).outcome == GuardrailOutcome.BLOCK


def test_the_worst_category_is_the_one_reported(make_settings: MakeSettings) -> None:
    """With several over the line, the report names the strongest signal."""
    guardrail = build(make_settings, moderation_threshold=0.3)
    attach(guardrail, Endpoint(scores(hate=0.4, violence=0.95, sexual=0.35)))
    assert guardrail.check("x", GuardrailStage.INPUT).category == "violence"


def test_only_watched_categories_are_enforced(make_settings: MakeSettings) -> None:
    """A deployment that only cares about one category says so."""
    guardrail = build(
        make_settings, moderation_threshold=0.5, moderation_categories=["hate"]
    )
    attach(guardrail, Endpoint(scores(violence=0.99)))
    assert guardrail.check("x", GuardrailStage.INPUT).outcome == GuardrailOutcome.ALLOW


def test_a_watched_category_still_blocks(make_settings: MakeSettings) -> None:
    guardrail = build(
        make_settings, moderation_threshold=0.5, moderation_categories=["hate"]
    )
    attach(guardrail, Endpoint(scores(hate=0.99, violence=0.99)))
    assert guardrail.check("x", GuardrailStage.INPUT).category == "hate"


def test_the_configured_model_is_the_one_called(make_settings: MakeSettings) -> None:
    guardrail = build(make_settings, moderation_model="omni-moderation-latest")
    endpoint = attach(guardrail, Endpoint(scores()))
    guardrail.check("hello", GuardrailStage.INPUT)
    assert endpoint.models == ["omni-moderation-latest"]
    assert endpoint.inputs == ["hello"]


def test_empty_content_never_reaches_the_endpoint(make_settings: MakeSettings) -> None:
    """An empty string is not worth a billable call."""
    guardrail = build(make_settings)
    endpoint = attach(guardrail, Endpoint(scores()))
    assert guardrail.check("", GuardrailStage.INPUT).outcome == GuardrailOutcome.ALLOW
    assert endpoint.inputs == []


def test_tool_results_are_not_moderated(make_settings: MakeSettings) -> None:
    """A tool result is checked for injection, which is a different question."""
    guardrail = build(make_settings)
    assert guardrail.stages == frozenset({GuardrailStage.INPUT, GuardrailStage.OUTPUT})


def test_an_unreachable_endpoint_allows_the_content(
    make_settings: MakeSettings,
) -> None:
    """A moderation outage must not take the service down with it."""

    class Dead:
        def create(self, **_: Any) -> ModerationCreateResponse:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]

    guardrail = build(make_settings)
    guardrail._client.moderations = Dead()  # type: ignore[assignment]
    assert (
        guardrail.check("anything", GuardrailStage.INPUT).outcome
        == GuardrailOutcome.ALLOW
    )


def test_a_reply_with_no_results_is_allowed(make_settings: MakeSettings) -> None:
    """A malformed reply is not evidence of a violation."""

    empty = scores().model_copy(update={"results": []})
    guardrail = build(make_settings)
    assert guardrail._verdict(empty).outcome == GuardrailOutcome.ALLOW


def test_the_guardrail_names_itself(make_settings: MakeSettings) -> None:
    assert build(make_settings).name == "openai-moderation"


async def test_the_async_path_blocks_the_same_way(make_settings: MakeSettings) -> None:
    """The async client is a second client, so it gets its own proof."""
    guardrail = build(make_settings, moderation_threshold=0.5)
    endpoint = attach_async(guardrail, AsyncEndpoint(scores(hate=0.8)))
    verdict = await guardrail.acheck("something", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.BLOCK
    assert verdict.category == "hate"
    assert endpoint.inputs == ["something"]


async def test_an_unreachable_endpoint_allows_the_content_async(
    make_settings: MakeSettings,
) -> None:
    class Dead:
        async def create(self, **_: Any) -> ModerationCreateResponse:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]

    guardrail = build(make_settings)
    guardrail._async_client.moderations = Dead()  # type: ignore[assignment]
    verdict = await guardrail.acheck("anything", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.ALLOW


async def test_empty_content_never_reaches_the_endpoint_async(
    make_settings: MakeSettings,
) -> None:
    guardrail = build(make_settings)
    endpoint = attach_async(guardrail, AsyncEndpoint(scores()))
    await guardrail.acheck("", GuardrailStage.INPUT)
    assert endpoint.inputs == []


def rejects(status: int) -> Any:
    """An endpoint answering with one HTTP status, as the service would."""

    class Refusing:
        def create(self, **_: Any) -> ModerationCreateResponse:
            raise APIStatusError(
                "refused", response=httpx.Response(status, request=REQUEST), body=None
            )

    return Refusing()


class RecordingLogger:
    """Captures which level the guardrail chose, which is the contract here."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str, *args: Any) -> None:
        self.errors.append(message % args if args else message)

    def warning(self, message: str, *args: Any) -> None:
        self.warnings.append(message % args if args else message)


def test_a_rejected_request_is_logged_as_a_misconfiguration(
    make_settings: MakeSettings,
) -> None:
    """A model the endpoint will not accept disables moderation entirely.

    Failing open means this looks healthy from the outside, so the error is
    the only thing standing between a typo and no moderation at all.
    """
    guardrail = build(make_settings)
    guardrail._client.moderations = rejects(400)  # type: ignore[assignment]
    logger = RecordingLogger()
    guardrail._logger = logger  # type: ignore[assignment]

    verdict = guardrail.check("anything", GuardrailStage.INPUT)

    assert verdict.outcome == GuardrailOutcome.ALLOW
    assert logger.warnings == []
    assert len(logger.errors) == 1
    assert "Nothing is being moderated" in logger.errors[0]


@pytest.mark.parametrize("status", [401, 404, 422])
def test_every_client_error_counts_as_a_misconfiguration(
    make_settings: MakeSettings, status: int
) -> None:
    guardrail = build(make_settings)
    error = APIStatusError(
        "refused", response=httpx.Response(status, request=REQUEST), body=None
    )
    assert guardrail._is_misconfigured(error)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_load_and_outages_are_not_misconfigurations(
    make_settings: MakeSettings, status: int
) -> None:
    """429 is load and 5xx is theirs; neither means the deployment is wrong."""
    guardrail = build(make_settings)
    error = APIStatusError(
        "refused", response=httpx.Response(status, request=REQUEST), body=None
    )
    assert not guardrail._is_misconfigured(error)


def test_an_unreachable_endpoint_is_only_a_warning(make_settings: MakeSettings) -> None:
    """A blip is not a misconfiguration and must not cry wolf as one."""
    guardrail = build(make_settings)

    class Dead:
        def create(self, **_: Any) -> ModerationCreateResponse:
            raise APIConnectionError(request=REQUEST)

    guardrail._client.moderations = Dead()  # type: ignore[assignment]
    logger = RecordingLogger()
    guardrail._logger = logger  # type: ignore[assignment]

    guardrail.check("anything", GuardrailStage.INPUT)

    assert logger.errors == []
    assert len(logger.warnings) == 1


def test_categories_are_watched_by_their_api_names(make_settings: MakeSettings) -> None:
    """The SDK's field is `self_harm`; the API and its docs say `self-harm`."""
    guardrail = build(make_settings, moderation_categories=["self-harm"])
    attach(guardrail, Endpoint(scores(**{"self-harm": 0.99})))
    verdict = guardrail.check("x", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.BLOCK
    assert verdict.category == "self-harm"


def test_one_client_serves_every_guardrail_with_the_same_settings(
    make_settings: MakeSettings,
) -> None:
    """A ToolRunner builds a pipeline each time; each must not open new connections."""
    first = build(make_settings)
    second = OpenAIModerationGuardrail(
        model_config=ModelConfigurations(
            api_host="https://moderation.invalid/",
            api_key="unused",
            api_type=ProviderType.OPENAI,
        )
    )
    assert second is first
