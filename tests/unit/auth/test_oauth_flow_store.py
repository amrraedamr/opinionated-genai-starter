"""What ties an OAuth callback to the browser that began the flow.

Without the binding, an attacker finishes a login as themselves and hands
the callback URL to a victim, who ends up signed in to the attacker's
account. The verifier is kept out of the front channel for the same reason.
"""

from typing import Any

import pytest

from base.models.definitions.errors import AuthErrors
from base.services.auth.oauth.flow import OAuthFlowStore
from base.services.cache.memory import InMemoryCacheService
from base.tools.errors import ErrorsException
from base.tools.security import pkce_pair

TTL = 300


@pytest.fixture
def store(make_settings: Any) -> OAuthFlowStore:
    make_settings()
    return OAuthFlowStore(cache=InMemoryCacheService())


async def _begin(store: OAuthFlowStore, provider: str = "google") -> Any:
    verifier, challenge = pkce_pair()
    return await store.abegin(provider, verifier, challenge, TTL)


async def test_the_verifier_stays_here(store: OAuthFlowStore) -> None:
    """Nothing the caller sends out may contain it."""
    flow = await _begin(store)
    assert flow.code_verifier not in flow.state_id
    assert flow.code_verifier not in flow.nonce
    assert flow.code_verifier not in flow.code_challenge


async def test_a_callback_from_the_same_browser_recovers_the_verifier(
    store: OAuthFlowStore,
) -> None:
    flow = await _begin(store)
    assert await store.aconsume("google", flow.state_id, flow.nonce) == (
        flow.code_verifier
    )


async def test_a_callback_with_no_binding_is_refused(store: OAuthFlowStore) -> None:
    """The attack: a valid state replayed in a browser that holds no cookie."""
    flow = await _begin(store)
    with pytest.raises(ErrorsException) as caught:
        await store.aconsume("google", flow.state_id, None)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_callback_with_another_browsers_binding_is_refused(
    store: OAuthFlowStore,
) -> None:
    flow = await _begin(store)
    other = await _begin(store)
    with pytest.raises(ErrorsException) as caught:
        await store.aconsume("google", flow.state_id, other.nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_state_from_another_provider_is_refused(
    store: OAuthFlowStore,
) -> None:
    """A flow begun for one provider must not finish at another."""
    flow = await _begin(store, "google")
    with pytest.raises(ErrorsException) as caught:
        await store.aconsume("github", flow.state_id, flow.nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_state_is_good_for_one_callback(store: OAuthFlowStore) -> None:
    """Even the entitled browser gets one use."""
    flow = await _begin(store)
    await store.aconsume("google", flow.state_id, flow.nonce)
    with pytest.raises(ErrorsException) as caught:
        await store.aconsume("google", flow.state_id, flow.nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_an_unknown_state_is_refused(store: OAuthFlowStore) -> None:
    with pytest.raises(ErrorsException) as caught:
        await store.aconsume("google", "never-issued", "whatever")
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_an_empty_state_is_refused(store: OAuthFlowStore) -> None:
    with pytest.raises(ErrorsException) as caught:
        await store.aconsume("google", "", "nonce")
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_two_flows_do_not_share_a_name_or_a_secret(
    store: OAuthFlowStore,
) -> None:
    first = await _begin(store)
    second = await _begin(store)
    assert first.state_id != second.state_id
    assert first.nonce != second.nonce
    assert first.code_verifier != second.code_verifier


async def test_each_flow_recovers_only_its_own_verifier(
    store: OAuthFlowStore,
) -> None:
    """Two browsers mid-flow at once must not cross."""
    first = await _begin(store)
    second = await _begin(store)
    assert await store.aconsume("google", second.state_id, second.nonce) == (
        second.code_verifier
    )
    assert await store.aconsume("google", first.state_id, first.nonce) == (
        first.code_verifier
    )
