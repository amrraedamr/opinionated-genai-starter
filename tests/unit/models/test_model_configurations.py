"""`ModelConfigurations`, `EmbeddingConfigurations` and `CacheHint`.

The two configuration builders are the only place a provider's settings turn
into the neutral shape an adapter reads, so each provider branch is exercised
against settings composed exactly as a service would compose them.
"""

from typing import Any

import pytest

from base.models.definitions.model import CachePersistence, ProviderType
from base.models.view_models.model import (
    CacheHint,
    EmbeddingConfigurations,
    ModelConfigurations,
)
from base.services.configuration.mixins import (
    AnthropicSettingsMixin,
    AzureOpenAIEmbeddingSettingsMixin,
    AzureOpenAILLMSettingsMixin,
    GeminiSettingsMixin,
    OpenAIEmbeddingSettingsMixin,
    OpenAILLMSettingsMixin,
)
from base.tools.errors import MissingSettingsError
from tests.conftest import MakeSettings

LLM_MIXINS = (
    OpenAILLMSettingsMixin,
    AzureOpenAILLMSettingsMixin,
    AnthropicSettingsMixin,
    GeminiSettingsMixin,
)

LLM_VALUES: dict[str, Any] = {
    "openai_llm_endpoint": "https://openai.example",
    "openai_llm_api_key": "openai-key",
    "openai_llm_model": "gpt-5",
    "azure_openai_llm_endpoint": "https://azure.example",
    "azure_openai_llm_api_key": "azure-key",
    "azure_openai_llm_deployment_name": "gpt-5-deployment",
    "azure_openai_api_version": "2026-01-01",
    "anthropic_llm_endpoint": "https://anthropic.example",
    "anthropic_api_key": "anthropic-key",
    "anthropic_llm_model": "claude-sonnet",
    "gemini_llm_endpoint": "https://gemini.example",
    "gemini_api_key": "gemini-key",
    "gemini_llm_model": "gemini-3",
}

EMBEDDING_MIXINS = (OpenAIEmbeddingSettingsMixin, AzureOpenAIEmbeddingSettingsMixin)

EMBEDDING_VALUES: dict[str, Any] = {
    "openai_embedding_endpoint": "https://openai.example",
    "openai_embedding_api_key": "openai-key",
    "openai_embedding_model": "text-embedding-3-small",
    "openai_embedding_dimensions": 1536,
    "openai_embedding_batch_size": 512,
    "azure_openai_embedding_endpoint": "https://azure.example",
    "azure_openai_embedding_api_key": "azure-key",
    "azure_openai_embedding_deployment_name": "embed-deployment",
    "azure_openai_embedding_dimensions": 3072,
    "azure_openai_embedding_batch_size": 1024,
    "azure_openai_api_version": "2026-01-01",
}


def test_openai_configuration_is_built_from_the_openai_settings(
    make_settings: MakeSettings,
) -> None:
    make_settings(*LLM_MIXINS, **LLM_VALUES)
    config = ModelConfigurations.get_default_for_provider(ProviderType.OPENAI)
    assert config.api_host == "https://openai.example"
    assert config.api_key == "openai-key"
    assert config.model_name == "gpt-5"
    assert config.api_type is ProviderType.OPENAI
    assert config.api_version is None


def test_azure_openai_configuration_carries_the_deployment_and_api_version(
    make_settings: MakeSettings,
) -> None:
    make_settings(*LLM_MIXINS, **LLM_VALUES)
    config = ModelConfigurations.get_default_for_provider(ProviderType.AZURE_OPENAI)
    assert config.model_name == "gpt-5-deployment"
    assert config.api_version == "2026-01-01"
    assert config.api_type is ProviderType.AZURE_OPENAI


def test_anthropic_configuration_is_built_from_the_anthropic_settings(
    make_settings: MakeSettings,
) -> None:
    make_settings(*LLM_MIXINS, **LLM_VALUES)
    config = ModelConfigurations.get_default_for_provider(ProviderType.ANTHROPIC)
    assert (config.api_host, config.api_key, config.model_name) == (
        "https://anthropic.example",
        "anthropic-key",
        "claude-sonnet",
    )
    assert config.api_type is ProviderType.ANTHROPIC


def test_gemini_configuration_is_built_from_the_gemini_settings(
    make_settings: MakeSettings,
) -> None:
    make_settings(*LLM_MIXINS, **LLM_VALUES)
    config = ModelConfigurations.get_default_for_provider(ProviderType.GEMINI)
    assert (config.api_host, config.api_key, config.model_name) == (
        "https://gemini.example",
        "gemini-key",
        "gemini-3",
    )
    assert config.api_type is ProviderType.GEMINI


@pytest.mark.parametrize(
    "provider",
    [
        ProviderType.OPENAI,
        ProviderType.AZURE_OPENAI,
        ProviderType.ANTHROPIC,
        ProviderType.GEMINI,
    ],
)
def test_every_supported_provider_gets_the_same_retry_defaults(
    make_settings: MakeSettings, provider: ProviderType
) -> None:
    make_settings(*LLM_MIXINS, **LLM_VALUES)
    config = ModelConfigurations.get_default_for_provider(provider)
    assert config.api_retries == 2
    assert config.wait_time == 1.0
    assert config.supports_responses_api is True


def test_an_empty_setting_becomes_none_rather_than_an_empty_string(
    make_settings: MakeSettings,
) -> None:
    make_settings(*LLM_MIXINS)
    config = ModelConfigurations.get_default_for_provider(ProviderType.OPENAI)
    assert config.api_host is None
    assert config.api_key is None
    assert config.model_name is None


@pytest.mark.parametrize(
    "provider",
    [
        ProviderType.OLLAMA,
        ProviderType.VLLM,
        ProviderType.GROQ,
        ProviderType.TOGETHER,
        ProviderType.LITELLM,
    ],
)
def test_a_provider_with_no_branch_is_refused_by_name(
    make_settings: MakeSettings, provider: ProviderType
) -> None:
    make_settings(*LLM_MIXINS, **LLM_VALUES)
    with pytest.raises(
        ValueError, match=f"Unsupported provider type: {provider.value}"
    ):
        ModelConfigurations.get_default_for_provider(provider)


def test_a_service_that_never_composed_the_provider_mixin_is_told_which_one(
    make_settings: MakeSettings,
) -> None:
    make_settings(OpenAILLMSettingsMixin, **{"openai_llm_model": "gpt-5"})
    with pytest.raises(MissingSettingsError, match="SupportsAnthropicConfig"):
        ModelConfigurations.get_default_for_provider(ProviderType.ANTHROPIC)


def test_openai_embedding_configuration_carries_its_batch_size_and_dimensions(
    make_settings: MakeSettings,
) -> None:
    make_settings(*EMBEDDING_MIXINS, **EMBEDDING_VALUES)
    config = EmbeddingConfigurations.get_default_for_provider(ProviderType.OPENAI)
    assert config.model_name == "text-embedding-3-small"
    assert config.batch_size == 512
    assert config.dimensions == 1536
    assert config.api_version is None


def test_azure_embedding_configuration_carries_the_deployment_and_api_version(
    make_settings: MakeSettings,
) -> None:
    make_settings(*EMBEDDING_MIXINS, **EMBEDDING_VALUES)
    config = EmbeddingConfigurations.get_default_for_provider(ProviderType.AZURE_OPENAI)
    assert config.model_name == "embed-deployment"
    assert config.dimensions == 3072
    assert config.batch_size == 1024
    assert config.api_version == "2026-01-01"


@pytest.mark.parametrize(
    "provider", [ProviderType.ANTHROPIC, ProviderType.GEMINI, ProviderType.OLLAMA]
)
def test_embeddings_refuse_a_provider_with_no_branch(
    make_settings: MakeSettings, provider: ProviderType
) -> None:
    make_settings(*EMBEDDING_MIXINS, **EMBEDDING_VALUES)
    with pytest.raises(ValueError, match="Unsupported provider type"):
        EmbeddingConfigurations.get_default_for_provider(provider)


def test_a_cache_hint_defaults_to_the_short_tier_and_a_derived_key() -> None:
    hint = CacheHint()
    assert hint.persistence is CachePersistence.SHORT
    assert hint.key is None


def test_a_cache_hint_can_name_the_group_calls_should_share() -> None:
    hint = CacheHint(persistence=CachePersistence.LONG, key="tenant-7")
    assert hint.persistence is CachePersistence.LONG
    assert hint.key == "tenant-7"
    assert hint.serialize() == {"persistence": "long", "key": "tenant-7"}
