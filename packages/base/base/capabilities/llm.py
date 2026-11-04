"""Point every agent at an LLM client that authenticates with a credential."""

from typing import cast

from fastapi import FastAPI

from base.clients.credentials import OutboundAuth
from base.models.definitions.health import ProbeStatus
from base.models.definitions.model import ProviderType
from base.models.view_models.health import ProbeResult
from base.models.view_models.model import ModelConfigurations
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsCredentialsConfig
from base.services.llm import AbstractLLMService
from base.services.llm.openai import OpenAIService
from base.services.logging.console import ConsoleLoggerService
from base.tools.hashing import hash_args

CONFIGURED_SERVICE_KEY: str = "authenticated-llm"
"""Explicit key: the default hashes constructor args, and an `httpx.Auth` has
no stable serialization."""


class AuthenticatedLLM:
    """Installs an LLM service carrying `auth` where a bare one resolves.

    One consumer of :class:`~base.capabilities.credentials.OutboundCredentials`,
    for an LLM endpoint behind an expiring token rather than a static key.
    List it after the credential it reads::

        credentials = OutboundCredentials()
        capabilities = [credentials, AuthenticatedLLM(credentials.auth)]

    `service_type` selects the provider. It defaults to OpenAI; any
    `AbstractLLMService` whose constructor takes `(config, auth, verify)` works::

        AuthenticatedLLM(
            Httpx2BearerTokenAuth(credentials.auth.provider),
            provider_type=ProviderType.ANTHROPIC,
            service_type=AnthropicService,
        )

    The auth object must come from the SDK's own HTTP library — `httpx` for
    OpenAI, `httpx2` for Anthropic — because the two reject each other. Both
    bindings live in `base.clients.credentials` and read one `TokenProvider`,
    so the token is acquired once however many providers consume it.

    `verify` turns off TLS verification for an endpoint behind an inspecting
    proxy; it defaults to the `outbound_verify_tls` setting.
    """

    name = "llm"

    def __init__(
        self,
        auth: OutboundAuth,
        model_config: ModelConfigurations | None = None,
        provider_type: ProviderType = ProviderType.AZURE_OPENAI,
        verify: bool | None = None,
        service_type: type[AbstractLLMService] = OpenAIService,
    ) -> None:
        self._auth = auth
        self._model_config = model_config
        self._provider_type = provider_type
        self._verify = verify
        self._service_type = service_type
        self._logger = ConsoleLoggerService()

    def _verify_tls(self) -> bool:
        if self._verify is not None:
            return self._verify
        return cast(SupportsCredentialsConfig, get_settings()).outbound_verify_tls

    async def astartup(self, app: FastAPI) -> None:
        del app
        model_config = (
            self._model_config
            or ModelConfigurations.get_default_for_provider(self._provider_type)
        )
        configured = self._service_type(
            model_config,
            auth=self._auth,
            verify=self._verify_tls(),
            _key=CONFIGURED_SERVICE_KEY,
        )
        # Installed under the key a no-argument construction resolves to, so
        # agents need no configuration. Safe because capabilities start before
        # the first request and agents are built lazily inside routers — a
        # service constructing one at import time would lose the race.
        self._service_type.replace(hash_args(), configured)
        self._logger.info(
            "Authenticated %s installed as the default LLM client.",
            self._service_type.__name__,
        )

    async def ashutdown(self, app: FastAPI) -> None:
        del app

    async def aprobe(self) -> ProbeResult:
        # The credential it carries is probed by OutboundCredentials; calling
        # the model here would bill a token on every readiness check.
        configured = self._service_type.get(CONFIGURED_SERVICE_KEY)
        installed = (
            configured is not None and self._service_type.get(hash_args()) is configured
        )
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK if installed else ProbeStatus.FAILED,
            detail=None if installed else "authenticated client not installed",
        )
