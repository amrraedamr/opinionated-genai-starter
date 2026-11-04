"""Point every agent at an LLM client that authenticates with a credential."""

from typing import cast

import httpx
from fastapi import FastAPI

from base.models.definitions.health import ProbeStatus
from base.models.definitions.model import ProviderType
from base.models.view_models.health import ProbeResult
from base.models.view_models.model import ModelConfigurations
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsCredentialsConfig
from base.services.llm import OpenAIService
from base.services.logging.console import ConsoleLoggerService
from base.tools.hashing import hash_args

CONFIGURED_SERVICE_KEY: str = "authenticated-llm"
"""Explicit key: the default hashes constructor args, and an `httpx.Auth` has
no stable serialization."""


class AuthenticatedLLM:
    """Installs an `OpenAIService` carrying `auth` where a bare one resolves.

    One consumer of :class:`~base.capabilities.credentials.OutboundCredentials`,
    for an LLM endpoint behind an expiring token rather than a static key.
    List it after the credential it reads::

        credentials = OutboundCredentials()
        capabilities = [credentials, AuthenticatedLLM(credentials.auth)]

    `verify` turns off TLS verification for an endpoint behind an inspecting
    proxy; it defaults to the `outbound_verify_tls` setting.
    """

    name = "llm"

    def __init__(
        self,
        auth: httpx.Auth,
        model_config: ModelConfigurations | None = None,
        provider_type: ProviderType = ProviderType.AZURE_OPENAI,
        verify: bool | None = None,
    ) -> None:
        self._auth = auth
        self._model_config = model_config
        self._provider_type = provider_type
        self._verify = verify
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
        configured = OpenAIService(
            model_config,
            auth=self._auth,
            verify=self._verify_tls(),
            _key=CONFIGURED_SERVICE_KEY,
        )
        # Installed under the key a no-argument OpenAIService() resolves to, so
        # agents need no configuration. Safe because capabilities start before
        # the first request and agents are built lazily inside routers — a
        # service constructing one at import time would lose the race.
        OpenAIService.replace(hash_args(), configured)
        self._logger.info("Authenticated LLM client installed as the default.")

    async def ashutdown(self, app: FastAPI) -> None:
        del app

    async def aprobe(self) -> ProbeResult:
        # The credential it carries is probed by OutboundCredentials; calling
        # the model here would bill a token on every readiness check.
        configured = OpenAIService.get(CONFIGURED_SERVICE_KEY)
        installed = (
            configured is not None and OpenAIService.get(hash_args()) is configured
        )
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK if installed else ProbeStatus.FAILED,
            detail=None if installed else "authenticated client not installed",
        )
