"""Construction of OpenAI-compatible clients, shared by every OpenAI-backed service."""

from urllib.parse import urlsplit

from httpx import Auth, Timeout

from base.models.definitions.model import ProviderType
from base.models.view_models.exceptions.provider import ProviderConfigError
from base.tools.errors import MissingExtraError

try:
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("openai", "openai", feature="OpenAI clients") from exc

DEFAULT_TIMEOUT_TOTAL: float = 60.0
DEFAULT_TIMEOUT_CONNECT: float = 5.0

UNUSED_API_KEY: str = "unused"
"""Satisfies the SDK's api_key requirement when `auth` supplies credentials."""

AZURE_OPENAI_HOST_SUFFIXES: tuple[str, ...] = (
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    ".services.ai.azure.com",
)
"""Hosts Azure itself serves, which reach the models under `/openai/v1`."""


def resolve_base_url(api_host: str | None, api_type: ProviderType) -> str | None:
    """Normalize a configured host to the base URL the OpenAI SDK expects.

    Azure serves its models under `/openai/v1`, and a configured host that
    names an Azure endpoint usually stops at the hostname, so the path is
    completed here rather than in every caller's config.

    That completion applies only to hosts Azure itself serves. A gateway or
    proxy carrying Azure models publishes whatever path it likes, commonly
    the plain `/v1` every OpenAI-compatible server uses, and appending to it
    turns a working URL into one that answers 401. Anything that is not an
    Azure hostname is therefore sent exactly as configured, including a
    private endpoint behind a custom domain, which has to name its own path.
    """
    base_url = api_host or None
    if api_type != ProviderType.AZURE_OPENAI:
        return base_url

    if not base_url:
        raise ProviderConfigError("api_host is required for Azure OpenAI")

    host = (urlsplit(base_url).hostname or "").lower()
    if not host.endswith(AZURE_OPENAI_HOST_SUFFIXES):
        return base_url

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"
    return f"{base_url}/"


def create_openai_clients(
    api_key: str | None,
    api_host: str | None,
    api_type: ProviderType,
    timeout_total: float = DEFAULT_TIMEOUT_TOTAL,
    timeout_connect: float = DEFAULT_TIMEOUT_CONNECT,
    auth: Auth | None = None,
    verify: bool = True,
) -> tuple[OpenAI, AsyncOpenAI]:
    """Build the sync/async client pair every OpenAI-compatible service needs.

    `auth` covers credentials the static `api_key` cannot express, such as an
    OAuth2 bearer that expires. An async-acquiring hook leaves the sync client
    unusable; it is still attached, because sending no credentials would be
    worse, and `OpenAIService` refuses a sync call outright.
    """
    base_url = resolve_base_url(api_host, api_type)
    timeout = Timeout(timeout_total, connect=timeout_connect)

    # The SDK refuses to construct without an api_key; the hook overwrites it.
    if auth is not None and not api_key:
        api_key = UNUSED_API_KEY

    return (
        OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=DefaultHttpxClient(http2=True, auth=auth, verify=verify),
        ),
        AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(http2=True, auth=auth, verify=verify),
        ),
    )
