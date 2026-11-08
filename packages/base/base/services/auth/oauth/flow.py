"""Tying an OAuth callback to the browser that began the flow.

A signature proves this server minted the state, not that this browser asked
for it, so the nonce is what binds the two. The PKCE verifier stays here
because the front channel already carries its challenge.
"""

import secrets
from typing import Any

from base.models.definitions.errors import AuthErrors
from base.models.definitions.oauth import OAUTH_FLOW_NAMESPACE
from base.models.view_models.oauth import PendingFlow
from base.services.cache import AbstractCacheService
from base.services.cache.factory import get_cache_service
from base.tools.errors import raise_error
from base.tools.security import constant_time_equals


class OAuthFlowStore:
    """Holds a flow between the authorize redirect and the callback."""

    def __init__(self, cache: AbstractCacheService | None = None) -> None:
        self._cache: AbstractCacheService = cache or get_cache_service()

    async def abegin(
        self, provider: str, code_verifier: str, code_challenge: str, ttl_seconds: int
    ) -> PendingFlow:
        """Record a flow and return what the caller must send out."""
        flow = PendingFlow(
            state_id=secrets.token_urlsafe(32),
            nonce=secrets.token_urlsafe(32),
            code_verifier=code_verifier,
            code_challenge=code_challenge,
        )
        await self._cache.aset(
            self._key(flow.state_id),
            OAUTH_FLOW_NAMESPACE,
            value={
                "provider": provider,
                "nonce": flow.nonce,
                "code_verifier": code_verifier,
            },
            ttl_seconds=ttl_seconds,
        )
        return flow

    async def aconsume(self, provider: str, state_id: str, nonce: str | None) -> str:
        """The verifier for a flow this browser began. Consumed on read."""
        if not state_id or not nonce:
            raise_error(AuthErrors.OAUTH_STATE_INVALID)

        pending: dict[str, Any] | None = await self._cache.aget(
            self._key(state_id), OAUTH_FLOW_NAMESPACE
        )
        await self._cache.aremove(self._key(state_id), OAUTH_FLOW_NAMESPACE)
        if not isinstance(pending, dict):
            raise_error(AuthErrors.OAUTH_STATE_INVALID)
        if pending.get("provider") != provider:
            raise_error(AuthErrors.OAUTH_STATE_INVALID)
        if not constant_time_equals(str(pending.get("nonce") or ""), nonce):
            raise_error(AuthErrors.OAUTH_STATE_INVALID)

        verifier = str(pending.get("code_verifier") or "")
        if not verifier:
            raise_error(AuthErrors.OAUTH_STATE_INVALID)
        return verifier

    @staticmethod
    def _key(state_id: str) -> str:
        return f"state:{state_id}"
