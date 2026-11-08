"""Names the OAuth flow relies on being spelled identically everywhere."""

from typing import Final

OAUTH_NONCE_COOKIE: Final[str] = "__Host-oauth-nonce"
"""The `__Host-` prefix is the security property: a browser takes it only
over TLS, with no `Domain`, so a sibling subdomain cannot set one."""

OAUTH_FLOW_NAMESPACE: Final[str] = "oauth-flow"
