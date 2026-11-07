"""`OutboundCredentials` owns one credential; `AuthenticatedLLM` spends it.

Both are built at import, before a composition root has registered settings,
so everything they read has to be resolved lazily. That is the property these
pin: constructing either must touch no configuration, and the probe is where a
bad credential surfaces rather than inside the first user request.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

from base.capabilities.credentials import OutboundCredentials
from base.capabilities.llm import CONFIGURED_SERVICE_KEY, AuthenticatedLLM
from base.clients.credentials import BearerTokenAuth, StaticTokenProvider
from base.clients.credentials.oauth2 import OAuth2ClientCredentials
from base.models.definitions.health import ProbeStatus
from base.models.definitions.model import ProviderType
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.tokens import AccessToken
from base.services.common.meta import KeyedSingleton
from base.services.configuration.mixins import (
    AzureOpenAILLMSettingsMixin,
    CredentialsSettingsMixin,
    OpenAILLMSettingsMixin,
)
from base.services.llm.openai import OpenAIService
from base.tools.errors import MissingSettingsError
from base.tools.hashing import hash_args

APP = FastAPI()


class FailingProvider:
    async def aget_token(self) -> AccessToken:
        raise RuntimeError("bad client secret")


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def aget_token(self) -> AccessToken:
        self.calls += 1
        return AccessToken("token")


# OutboundCredentials


def test_constructing_the_capability_reads_no_settings() -> None:
    capability = OutboundCredentials()

    assert capability.name == "credentials"
    assert capability._auth is None
    assert capability._provider is None


def test_a_name_distinguishes_one_credential_from_another() -> None:
    assert OutboundCredentials(name="billing-api").name == "billing-api"


def test_the_auth_object_is_built_once_and_reused() -> None:
    capability = OutboundCredentials(provider=StaticTokenProvider("t"))

    first = capability.auth
    assert isinstance(first, BearerTokenAuth)
    assert capability.auth is first


def test_the_auth_object_carries_the_provider_it_was_given() -> None:
    provider = StaticTokenProvider("t")
    assert OutboundCredentials(provider=provider).auth.provider is provider


def test_a_provider_is_built_from_settings_when_none_is_supplied(
    make_settings: Any,
) -> None:
    make_settings(
        CredentialsSettingsMixin,
        outbound_token_url="https://idp.example/token",
        outbound_client_id="client",
        outbound_client_secret="secret",
        outbound_scope="api://scope/.default",
        outbound_credential_placement="form",
        outbound_token_refresh_skew_seconds=15,
    )

    provider = OutboundCredentials().provider

    assert isinstance(provider, OAuth2ClientCredentials)
    assert provider._token_url == "https://idp.example/token"
    assert provider._scope == "api://scope/.default"
    assert provider._credential_placement == "form"
    assert provider._refresh_skew_seconds == 15.0


def test_explicit_configuration_wins_over_the_registered_settings(
    make_settings: Any,
) -> None:
    make_settings(
        CredentialsSettingsMixin, outbound_token_url="https://registered/token"
    )
    explicit = type(
        "Config",
        (),
        {
            "outbound_token_url": "https://explicit/token",
            "outbound_client_id": "c",
            "outbound_client_secret": "s",
            "outbound_scope": "",
            "outbound_credential_placement": "basic",
            "outbound_token_refresh_skew_seconds": 60,
            "outbound_verify_tls": True,
        },
    )()

    assert OutboundCredentials(config=explicit).settings is explicit


def test_a_service_that_never_composed_the_mixin_is_told_which_one(
    make_settings: Any,
) -> None:
    make_settings(OpenAILLMSettingsMixin)

    with pytest.raises(MissingSettingsError, match="SupportsCredentialsConfig"):
        OutboundCredentials().settings.outbound_token_url


async def test_the_probe_acquires_a_token_so_a_bad_secret_fails_readiness() -> None:
    result = await OutboundCredentials(provider=FailingProvider()).aprobe()

    assert result.status is ProbeStatus.FAILED
    assert result.detail == "could not acquire an outbound token"


async def test_a_working_credential_probes_ok_under_its_own_name() -> None:
    result = await OutboundCredentials(
        provider=StaticTokenProvider("t"), name="billing-api"
    ).aprobe()

    assert (result.name, result.status, result.detail) == (
        "billing-api",
        ProbeStatus.OK,
        None,
    )


async def test_every_probe_asks_the_provider_again() -> None:
    provider = CountingProvider()
    capability = OutboundCredentials(provider=provider)

    await capability.aprobe()
    await capability.aprobe()

    assert provider.calls == 2


async def test_startup_builds_the_auth_so_a_malformed_url_fails_at_boot(
    make_settings: Any,
) -> None:
    make_settings(
        CredentialsSettingsMixin,
        outbound_token_url="",
        outbound_client_id="c",
        outbound_client_secret="s",
    )

    with pytest.raises(ValueError, match="token_url is required"):
        await OutboundCredentials().astartup(APP)


async def test_startup_with_a_usable_credential_leaves_the_auth_ready() -> None:
    capability = OutboundCredentials(provider=StaticTokenProvider("t"))

    await capability.astartup(APP)

    assert isinstance(capability.auth, BearerTokenAuth)
    await capability.ashutdown(APP)


# AuthenticatedLLM


def stand_in() -> type:
    """A uniquely keyed `AbstractLLMService` stand-in, built like the real ones."""

    class StandInLLM(metaclass=KeyedSingleton):
        def __init__(
            self,
            config: ModelConfigurations | None = None,
            auth: Any = None,
            verify: bool = True,
        ) -> None:
            self.config = config
            self.auth = auth
            self.verify = verify

    return StandInLLM


def test_the_default_provider_is_openai_over_azure_configuration() -> None:
    capability = AuthenticatedLLM(BearerTokenAuth(StaticTokenProvider("t")))

    assert capability._service_type is OpenAIService
    assert capability._provider_type is ProviderType.AZURE_OPENAI
    assert capability.name == "llm"


async def test_startup_installs_the_authenticated_client_where_a_bare_one_resolves(
    make_settings: Any,
) -> None:
    make_settings(
        CredentialsSettingsMixin,
        AzureOpenAILLMSettingsMixin,
        azure_openai_llm_endpoint="https://azure.example",
        azure_openai_llm_api_key="key",
        azure_openai_llm_deployment_name="gpt-4o",
    )
    service_type = stand_in()
    auth = BearerTokenAuth(StaticTokenProvider("t"))

    await AuthenticatedLLM(auth, service_type=service_type).astartup(APP)

    configured = service_type.get(CONFIGURED_SERVICE_KEY)
    assert configured is not None
    assert configured.auth is auth
    assert service_type.get(hash_args()) is configured


async def test_the_installed_client_carries_the_providers_default_configuration(
    make_settings: Any,
) -> None:
    make_settings(
        CredentialsSettingsMixin,
        OpenAILLMSettingsMixin,
        openai_llm_endpoint="https://openai.example",
        openai_llm_api_key="key",
        openai_llm_model="gpt-5.4",
    )
    service_type = stand_in()

    await AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        provider_type=ProviderType.OPENAI,
        service_type=service_type,
    ).astartup(APP)

    configured = service_type.get(CONFIGURED_SERVICE_KEY)
    assert configured.config.model_name == "gpt-5.4"
    assert configured.config.api_type is ProviderType.OPENAI


async def test_an_explicit_model_configuration_is_used_as_given(
    make_settings: Any,
) -> None:
    make_settings(CredentialsSettingsMixin)
    service_type = stand_in()
    model_config = ModelConfigurations(model_name="pinned")

    await AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=model_config,
        service_type=service_type,
    ).astartup(APP)

    assert service_type.get(CONFIGURED_SERVICE_KEY).config is model_config


async def test_tls_verification_follows_the_setting_by_default(
    make_settings: Any,
) -> None:
    make_settings(CredentialsSettingsMixin, outbound_verify_tls=False)
    service_type = stand_in()

    await AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=ModelConfigurations(),
        service_type=service_type,
    ).astartup(APP)

    assert service_type.get(CONFIGURED_SERVICE_KEY).verify is False


async def test_an_explicit_verify_flag_beats_the_setting(make_settings: Any) -> None:
    make_settings(CredentialsSettingsMixin, outbound_verify_tls=True)
    service_type = stand_in()

    await AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=ModelConfigurations(),
        verify=False,
        service_type=service_type,
    ).astartup(APP)

    assert service_type.get(CONFIGURED_SERVICE_KEY).verify is False


async def test_the_probe_fails_before_startup_has_installed_anything() -> None:
    capability = AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")), service_type=stand_in()
    )

    result = await capability.aprobe()

    assert result.status is ProbeStatus.FAILED
    assert result.detail == "authenticated client not installed"


async def test_the_probe_passes_once_the_client_is_installed(
    make_settings: Any,
) -> None:
    make_settings(CredentialsSettingsMixin)
    capability = AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=ModelConfigurations(),
        service_type=stand_in(),
    )
    await capability.astartup(APP)

    result = await capability.aprobe()

    assert (result.name, result.status, result.detail) == ("llm", ProbeStatus.OK, None)


async def test_the_probe_fails_again_if_a_bare_client_displaces_the_configured_one(
    make_settings: Any,
) -> None:
    make_settings(CredentialsSettingsMixin)
    service_type = stand_in()
    capability = AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=ModelConfigurations(),
        service_type=service_type,
    )
    await capability.astartup(APP)

    service_type.replace(hash_args(), service_type(_key="somewhere-else"))

    assert (await capability.aprobe()).status is ProbeStatus.FAILED


async def test_the_probe_never_calls_the_model(make_settings: Any) -> None:
    make_settings(CredentialsSettingsMixin)

    class CallCountingLLM(metaclass=KeyedSingleton):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls = 0

        async def acomplete(self, **kwargs: Any) -> None:
            self.calls += 1

    capability = AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=ModelConfigurations(),
        service_type=CallCountingLLM,
    )
    await capability.astartup(APP)
    await capability.aprobe()

    assert CallCountingLLM.get(CONFIGURED_SERVICE_KEY).calls == 0


async def test_shutdown_leaves_the_installed_client_in_place(
    make_settings: Any,
) -> None:
    make_settings(CredentialsSettingsMixin)
    service_type = stand_in()
    capability = AuthenticatedLLM(
        BearerTokenAuth(StaticTokenProvider("t")),
        model_config=ModelConfigurations(),
        service_type=service_type,
    )
    await capability.astartup(APP)

    await capability.ashutdown(APP)

    assert service_type.get(CONFIGURED_SERVICE_KEY) is not None
