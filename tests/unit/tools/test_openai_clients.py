"""`openai_clients.py`: the base URL a configured host turns into.

Every OpenAI-compatible provider is reached through the same SDK, so the only
thing that varies is the URL — and Azure is the one that serves its models
under a path the SDK does not add. Getting that wrong gives a 401 rather than
a connection error, which is why each host shape is pinned here rather than
discovered in a deployment.
"""

from typing import Any

import httpx
import pytest
from openai import OpenAIError

from base.models.definitions.model import ProviderType
from base.models.view_models.exceptions.provider import ProviderConfigError
from base.tools.openai_clients import (
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    UNUSED_API_KEY,
    create_openai_clients,
    resolve_base_url,
)


class Bearer(httpx.Auth):
    """Credentials the static `api_key` cannot express."""

    def auth_flow(self, request: httpx.Request) -> Any:
        request.headers["Authorization"] = "Bearer minted-just-now"
        yield request


@pytest.mark.parametrize(
    "api_type",
    [
        ProviderType.OPENAI,
        ProviderType.OLLAMA,
        ProviderType.VLLM,
        ProviderType.GROQ,
        ProviderType.LITELLM,
    ],
)
def test_a_non_azure_host_is_sent_exactly_as_configured(
    api_type: ProviderType,
) -> None:
    assert resolve_base_url("https://gateway.corp/v1", api_type) == (
        "https://gateway.corp/v1"
    )


def test_an_unset_host_means_the_sdk_default() -> None:
    assert resolve_base_url(None, ProviderType.OPENAI) is None
    assert resolve_base_url("", ProviderType.OPENAI) is None


def test_azure_without_a_host_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ProviderConfigError, match="api_host is required"):
        resolve_base_url(None, ProviderType.AZURE_OPENAI)


def test_azure_with_a_blank_host_is_refused_too() -> None:
    with pytest.raises(ProviderConfigError, match="api_host is required"):
        resolve_base_url("", ProviderType.AZURE_OPENAI)


@pytest.mark.parametrize(
    "host",
    [
        "https://contoso.openai.azure.com",
        "https://contoso.cognitiveservices.azure.com",
        "https://contoso.services.ai.azure.com",
    ],
)
def test_a_host_azure_itself_serves_gains_the_path_the_models_live_under(
    host: str,
) -> None:
    assert resolve_base_url(host, ProviderType.AZURE_OPENAI) == f"{host}/openai/v1/"


def test_a_trailing_slash_does_not_become_a_double_slash() -> None:
    assert (
        resolve_base_url("https://contoso.openai.azure.com/", ProviderType.AZURE_OPENAI)
        == "https://contoso.openai.azure.com/openai/v1/"
    )


def test_a_host_that_already_names_the_path_is_not_given_it_twice() -> None:
    assert (
        resolve_base_url(
            "https://contoso.openai.azure.com/openai/v1", ProviderType.AZURE_OPENAI
        )
        == "https://contoso.openai.azure.com/openai/v1/"
    )


def test_an_azure_host_is_recognised_whatever_its_casing() -> None:
    assert (
        resolve_base_url("https://CONTOSO.OpenAI.Azure.Com", ProviderType.AZURE_OPENAI)
        == "https://CONTOSO.OpenAI.Azure.Com/openai/v1/"
    )


@pytest.mark.parametrize(
    "host",
    [
        "https://gateway.corp.net/v1",
        "https://azure-openai.internal.corp/v1",
        "https://contoso.openai.azure.com.evil.test/v1",
        "http://localhost:8080/v1",
    ],
)
def test_a_gateway_carrying_azure_models_keeps_the_path_it_publishes(
    host: str,
) -> None:
    """Appending `/openai/v1` to a proxy's own path turns a working URL into a 401."""
    assert resolve_base_url(host, ProviderType.AZURE_OPENAI) == host


def test_a_host_with_no_scheme_has_no_hostname_to_match_on() -> None:
    """Pinned as it stands: `urlsplit` reads this as a path, so nothing is added."""
    assert (
        resolve_base_url("contoso.openai.azure.com", ProviderType.AZURE_OPENAI)
        == "contoso.openai.azure.com"
    )


def test_both_clients_are_built_and_point_at_the_same_place() -> None:
    sync_client, async_client = create_openai_clients(
        "sk-test", "https://gateway.corp/v1", ProviderType.OPENAI
    )

    assert str(sync_client.base_url) == "https://gateway.corp/v1/"
    assert str(async_client.base_url) == str(sync_client.base_url)


def test_the_azure_path_reaches_the_clients_themselves() -> None:
    sync_client, async_client = create_openai_clients(
        "sk-test", "https://contoso.openai.azure.com", ProviderType.AZURE_OPENAI
    )

    assert str(sync_client.base_url) == "https://contoso.openai.azure.com/openai/v1/"
    assert str(async_client.base_url) == str(sync_client.base_url)


def test_sdk_retries_are_off_because_this_repository_owns_the_retry_policy() -> None:
    sync_client, async_client = create_openai_clients(
        "sk-test", "https://gateway.corp/v1", ProviderType.OPENAI
    )

    assert (sync_client.max_retries, async_client.max_retries) == (0, 0)


def test_the_documented_timeouts_are_the_ones_applied() -> None:
    sync_client, _ = create_openai_clients(
        "sk-test", "https://gateway.corp/v1", ProviderType.OPENAI
    )

    assert sync_client.timeout.read == DEFAULT_TIMEOUT_TOTAL
    assert sync_client.timeout.connect == DEFAULT_TIMEOUT_CONNECT


def test_a_caller_can_shorten_the_timeouts() -> None:
    sync_client, _ = create_openai_clients(
        "sk-test",
        "https://gateway.corp/v1",
        ProviderType.OPENAI,
        timeout_total=2.0,
        timeout_connect=0.5,
    )

    assert (sync_client.timeout.read, sync_client.timeout.connect) == (2.0, 0.5)


def test_an_auth_hook_without_an_api_key_still_builds_a_client() -> None:
    """The SDK refuses to construct without one, so a placeholder stands in."""
    sync_client, async_client = create_openai_clients(
        None, "https://gateway.corp/v1", ProviderType.OPENAI, auth=Bearer()
    )

    assert sync_client.api_key == UNUSED_API_KEY
    assert async_client.api_key == UNUSED_API_KEY


def test_a_real_api_key_is_not_replaced_by_the_placeholder() -> None:
    sync_client, _ = create_openai_clients(
        "sk-real", "https://gateway.corp/v1", ProviderType.OPENAI, auth=Bearer()
    )

    assert sync_client.api_key == "sk-real"


def test_the_auth_hook_is_attached_to_both_clients() -> None:
    auth = Bearer()

    sync_client, async_client = create_openai_clients(
        None, "https://gateway.corp/v1", ProviderType.OPENAI, auth=auth
    )

    assert sync_client._client.auth is auth
    assert async_client._client.auth is auth


def test_no_credentials_at_all_is_refused_by_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the placeholder exists: the SDK will not construct without one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    with pytest.raises(OpenAIError, match="Missing credentials"):
        create_openai_clients(None, "https://gateway.corp/v1", ProviderType.OPENAI)


def test_certificate_verification_can_be_turned_off_for_a_private_endpoint() -> None:
    sync_client, async_client = create_openai_clients(
        "sk-test", "https://gateway.corp/v1", ProviderType.OPENAI, verify=False
    )

    assert sync_client._client is not None
    assert async_client._client is not None


def test_an_azure_configuration_with_no_host_never_reaches_the_sdk() -> None:
    with pytest.raises(ProviderConfigError):
        create_openai_clients("sk-test", None, ProviderType.AZURE_OPENAI)
