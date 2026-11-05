"""Acquire and refresh a token this service presents to a protected dependency."""

from typing import cast

from fastapi import FastAPI

from base.clients.credentials import BearerTokenAuth, TokenProvider
from base.clients.credentials.oauth2 import (
    CredentialPlacement,
    OAuth2ClientCredentials,
)
from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsCredentialsConfig
from base.services.logging.console import ConsoleLoggerService


class OutboundCredentials:
    """Keeps one outbound credential valid, and fails `/readyz` when it is not.

    The capability owns the credential; what consumes it is the service's
    business. :attr:`auth` is a plain `httpx.Auth`, so it works anywhere httpx
    does — `HttpRequestService.acall_endpoint`, an OpenAI client, any SDK that
    accepts a prepared client::

        credentials = OutboundCredentials()
        app = create_app(capabilities=[credentials, ...])

        await HttpRequestService().acall_endpoint(url, auth=credentials.auth)

    Reads a token URL, scope and credential from settings, and does not know
    which authorization server is behind them. Pass `provider` for a source
    settings cannot describe, such as a platform-injected managed identity,
    and `name` to distinguish one credential's probe from another's.
    """

    def __init__(
        self,
        config: SupportsCredentialsConfig | None = None,
        provider: TokenProvider | None = None,
        name: str = "credentials",
    ) -> None:
        self.name = name
        self._config = config
        self._provider = provider
        self._auth: BearerTokenAuth | None = None
        self._logger = ConsoleLoggerService()

    @property
    def settings(self) -> SupportsCredentialsConfig:
        # Resolved lazily: capabilities are built at import, settings are
        # registered by the composition root.
        return self._config or settings_for(SupportsCredentialsConfig)

    @property
    def provider(self) -> TokenProvider:
        if self._provider is None:
            self._provider = self._build_provider()
        return self._provider

    @property
    def auth(self) -> BearerTokenAuth:
        """An `httpx.Auth` that refreshes behind every request it signs."""
        if self._auth is None:
            self._auth = BearerTokenAuth(self.provider)
        return self._auth

    def _build_provider(self) -> TokenProvider:
        config = self.settings
        return OAuth2ClientCredentials(
            token_url=config.outbound_token_url,
            client_id=config.outbound_client_id,
            client_secret=config.outbound_client_secret,
            scope=config.outbound_scope,
            credential_placement=cast(
                CredentialPlacement, config.outbound_credential_placement
            ),
            refresh_skew_seconds=float(config.outbound_token_refresh_skew_seconds),
        )

    async def astartup(self, app: FastAPI) -> None:
        del app
        # Built now so a malformed token URL fails startup, not a later request.
        self._logger.info(
            f"Outbound credential {self.name!r} using "
            f"{type(self.auth.provider).__name__}."
        )

    async def ashutdown(self, app: FastAPI) -> None:
        del app

    async def aprobe(self) -> ProbeResult:
        # Acquiring here is the point: a bad client secret fails readiness
        # instead of surfacing inside the first user request.
        try:
            await self.provider.aget_token()
        except Exception as exc:
            self._logger.error(f"Outbound credential {self.name!r} failed: {exc}")
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.FAILED,
                detail="could not acquire an outbound token",
            )
        return ProbeResult(name=self.name, status=ProbeStatus.OK, detail=None)
