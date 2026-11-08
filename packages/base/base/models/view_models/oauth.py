"""What an OAuth provider is configured as, and what it says about a user."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OAuthProviderConfig:
    """One provider's endpoints and credentials."""

    name: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: str
    discovery_url: str = ""
    authorize_url: str = ""
    token_url: str = ""
    userinfo_url: str = ""
    profile_style: str = "oidc"
    """oidc | facebook | github | apple"""
    response_mode: str = ""
    apple_team_id: str = ""
    apple_key_id: str = ""
    apple_private_key: str = ""
    """Apple ES256 client-secret material; the secret is regenerated at token exchange."""


@dataclass(frozen=True, slots=True)
class OAuthProfile:
    """The identity a provider returned, in terms every provider can express."""

    subject: str
    email: str
    email_verified: bool
    first_name: str | None = None
    last_name: str | None = None


@dataclass(frozen=True, slots=True)
class PendingFlow:
    """A login begun and not yet finished.

    `nonce` goes in a cookie, `state_id` in the authorize URL; neither is
    useful without the other.
    """

    state_id: str
    nonce: str
    code_verifier: str
    code_challenge: str
