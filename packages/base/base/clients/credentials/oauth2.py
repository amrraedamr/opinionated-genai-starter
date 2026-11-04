"""The OAuth2 client-credentials grant, cached and refreshed for the service."""

import asyncio
import time
from collections.abc import Mapping
from typing import Literal

from pydantic import ValidationError

from base.models.definitions.errors import AuthErrors
from base.models.view_models.http import HttpResponse
from base.models.view_models.tokens import AccessToken, TokenResponse
from base.services.http import HttpRequestService
from base.services.logging.console import ConsoleLoggerService
from base.tools.errors import raise_error

CredentialPlacement = Literal["basic", "form"]
"""Credentials in an HTTP Basic header, or in form fields. Providers differ."""


class OAuth2ClientCredentials:
    """Mint and cache a service access token, per RFC 6749.

    Implements :class:`~base.clients.credentials.TokenProvider`. Caches on the
    lifetime the provider reports rather than one invented here, and guards
    concurrent refresh so a burst mints one token.

    The parameters below are the entire difference between Azure AD, Okta,
    Auth0, Keycloak and Cognito, so there is no presets module — a named
    constructor for one provider belongs in the service that talks to it::

        provider = OAuth2ClientCredentials(
            token_url=settings.outbound_token_url,
            client_id=settings.outbound_client_id,
            client_secret=settings.outbound_client_secret,
            scope=settings.outbound_scope,
        )
        auth = BearerTokenAuth(provider)
    """

    DEFAULT_LIFETIME_SECONDS: float = 3600.0
    """Assumed lifetime when the token endpoint omits `expires_in`."""

    DEFAULT_REFRESH_SKEW_SECONDS: float = 60.0
    """How early a token counts as stale. A skew above the issued lifetime
    means nothing is ever served from cache."""

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str = "",
        credential_placement: CredentialPlacement = "basic",
        extra_form_fields: Mapping[str, str] | None = None,
        refresh_skew_seconds: float = DEFAULT_REFRESH_SKEW_SECONDS,
        default_lifetime_seconds: float = DEFAULT_LIFETIME_SECONDS,
    ) -> None:
        if not token_url:
            raise ValueError("token_url is required for the client-credentials grant")
        if not client_id or not client_secret:
            raise ValueError(
                "client_id and client_secret are required for the "
                "client-credentials grant"
            )

        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._credential_placement: CredentialPlacement = credential_placement
        self._extra_form_fields = dict(extra_form_fields or {})
        self._refresh_skew_seconds = max(0.0, refresh_skew_seconds)
        self._default_lifetime_seconds = default_lifetime_seconds

        self._http = HttpRequestService()
        self._logger = ConsoleLoggerService()
        self._lock = asyncio.Lock()
        self._token: AccessToken | None = None
        self._expires_at: float = 0.0

    @property
    def token_url(self) -> str:
        return self._token_url

    async def aget_token(self) -> AccessToken:
        """Return a cached token, minting a new one once it is near expiry."""
        # Checked before the lock to keep the common path lock-free, and again
        # under it so a burst of callers mints once.
        cached = self._cached_token()
        if cached is not None:
            return cached

        async with self._lock:
            cached = self._cached_token()
            if cached is not None:
                return cached
            return await self._amint_token()

    def invalidate(self) -> None:
        """Drop the cached token. For a dependency rejecting one still cached."""
        self._token = None
        self._expires_at = 0.0

    def _cached_token(self) -> AccessToken | None:
        if self._token is None:
            return None
        if time.monotonic() >= self._expires_at - self._refresh_skew_seconds:
            return None
        return self._token

    def _build_form(self) -> dict[str, str]:
        form: dict[str, str] = {"grant_type": "client_credentials"}
        if self._scope:
            form["scope"] = self._scope
        if self._credential_placement == "form":
            form["client_id"] = self._client_id
            form["client_secret"] = self._client_secret
        form.update(self._extra_form_fields)
        return form

    async def _amint_token(self) -> AccessToken:
        auth = (
            (self._client_id, self._client_secret)
            if self._credential_placement == "basic"
            else None
        )
        response: HttpResponse = await self._http.acall_endpoint(
            self._token_url,
            method="POST",
            data=self._build_form(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            auth=auth,
        )
        if response.failed or not isinstance(response.data, dict):
            self._logger.error(
                f"Outbound token request to {self._token_url} failed: "
                f"{response.error_detail or response.error}",
                status=response.status_code,
            )
            raise_error(AuthErrors.OUTBOUND_TOKEN_FAILED)

        try:
            token_response = TokenResponse.model_validate(response.data)
        except ValidationError as exc:
            self._logger.error(
                f"Outbound token endpoint {self._token_url} returned no usable "
                f"access token: {exc}"
            )
            raise_error(AuthErrors.OUTBOUND_TOKEN_FAILED)

        lifetime = float(token_response.expires_in or self._default_lifetime_seconds)
        self._token = token_response.access_token
        self._expires_at = time.monotonic() + lifetime
        return self._token


__all__ = ["CredentialPlacement", "OAuth2ClientCredentials"]
