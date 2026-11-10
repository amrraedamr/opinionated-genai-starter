"""Asking a provider whether content crosses a policy line."""

from typing import TYPE_CHECKING, ClassVar

from base.models.definitions.guardrails import GuardrailStage
from base.models.definitions.model import ProviderType
from base.models.view_models.guardrails import GuardrailVerdict
from base.models.view_models.model import ModelConfigurations
from base.services.configuration import settings_for

if TYPE_CHECKING:
    from openai.types.moderation_create_response import ModerationCreateResponse
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.services.guardrails import AbstractGuardrail
from base.services.logging.console import ConsoleLoggerService
from base.tools.errors import MissingExtraError

try:
    from openai import OpenAIError
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "openai", "openai", feature="OpenAIModerationGuardrail"
    ) from exc
from base.tools.openai_clients import create_openai_clients


class OpenAIModerationGuardrail(AbstractGuardrail):
    """Moderation over the OpenAI-compatible `/moderations` endpoint.

    The API shape, not the vendor: anything serving that spec answers it —
    OpenAI, Azure OpenAI, LiteLLM, vLLM. A chat model is not valid input to it,
    so Gemini and Anthropic models are rejected; neither provider publishes an
    equivalent endpoint, and either would be a sibling module here.

    Blocks content scoring at or above a threshold. Scores are used rather than
    the endpoint's own `flagged`, because what is worth refusing differs per
    deployment and a boolean cannot be tuned.

    **This guardrail fails open.** If the endpoint cannot be reached the
    content is allowed, because a moderation outage should not take the service
    down with it. Where that trade is wrong, refuse at a layer that can.

    Failing open makes a misconfiguration dangerous rather than merely broken:
    a `moderation_model` the endpoint rejects would otherwise allow everything
    while looking healthy. A refused request is therefore logged as an error
    saying nothing is being moderated, not as a passing blip.
    """

    DEFAULT_TIMEOUT_TOTAL: ClassVar[float] = 10.0
    DEFAULT_TIMEOUT_CONNECT: ClassVar[float] = 5.0

    def __init__(
        self,
        config: SupportsGuardrailConfig | None = None,
        model_config: ModelConfigurations | None = None,
    ) -> None:
        self._config = config or settings_for(SupportsGuardrailConfig)
        self._model_config = (
            model_config
            or ModelConfigurations.get_default_for_provider(ProviderType.OPENAI)
        )
        self._logger = ConsoleLoggerService()
        self._client, self._async_client = create_openai_clients(
            api_key=self._model_config.api_key,
            api_host=self._model_config.api_host,
            api_type=ProviderType.OPENAI,
            timeout_total=self.DEFAULT_TIMEOUT_TOTAL,
            timeout_connect=self.DEFAULT_TIMEOUT_CONNECT,
        )

    def _name(self) -> str:
        return "openai-moderation"

    def _stages(self) -> frozenset[GuardrailStage]:
        return frozenset({GuardrailStage.INPUT, GuardrailStage.OUTPUT})

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        if not content:
            return GuardrailVerdict.allow()
        try:
            response = self._client.moderations.create(
                model=self._config.moderation_model, input=content
            )
        except OpenAIError as exc:
            return self._failed_open(exc)
        return self._verdict(response)

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        if not content:
            return GuardrailVerdict.allow()
        try:
            response = await self._async_client.moderations.create(
                model=self._config.moderation_model, input=content
            )
        except OpenAIError as exc:
            return self._failed_open(exc)
        return self._verdict(response)

    def _verdict(self, response: "ModerationCreateResponse") -> GuardrailVerdict:
        """Refuse on the highest-scoring category that crosses the threshold."""
        scores = self._scores(response)
        watched = self._config.moderation_categories
        threshold = self._config.moderation_threshold

        crossed = {
            category: score
            for category, score in scores.items()
            if score >= threshold and (not watched or category in watched)
        }
        if not crossed:
            return GuardrailVerdict.allow()

        worst = max(crossed, key=lambda category: crossed[category])
        return GuardrailVerdict.block(
            "Content was refused by the moderation policy.", category=worst
        )

    @staticmethod
    def _scores(response: "ModerationCreateResponse") -> dict[str, float]:
        """Scores keyed by API category name: `self-harm`, not `self_harm`."""
        if not response.results:
            return {}
        scores = response.results[0].category_scores.model_dump(by_alias=True)
        return {
            category: float(score)
            for category, score in scores.items()
            if score is not None
        }

    def _failed_open(self, exc: OpenAIError) -> GuardrailVerdict:
        """Allow the content, saying how loudly depending on what went wrong."""
        if self._is_misconfigured(exc):
            self._logger.error(
                "Moderation refused the request itself (%s). Nothing is being "
                "moderated: check moderation_model and the credentials.",
                exc,
            )
        else:
            self._logger.warning("Moderation unreachable, allowing content: %s", exc)
        return GuardrailVerdict.allow()

    @staticmethod
    def _is_misconfigured(exc: OpenAIError) -> bool:
        """A 4xx is this deployment being wrong, not the network blinking.

        Rate limiting is excluded: 429 is load, and it passes on its own.
        """
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and 400 <= status < 500 and status != 429


__all__ = ["OpenAIModerationGuardrail"]
