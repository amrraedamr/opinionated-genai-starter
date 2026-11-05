
from base.models.definitions.model import ProviderType
from base.models.view_models import BaseModel
from base.services.configuration import settings_for
from base.services.configuration.protocols import (
    SupportsAnthropicConfig,
    SupportsAzureOpenAIEmbeddingConfig,
    SupportsAzureOpenAILLMConfig,
    SupportsGeminiConfig,
    SupportsOpenAIEmbeddingConfig,
    SupportsOpenAILLMConfig,
)


class ModelConfigurations(BaseModel):
    api_host: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    api_version: str | None = None
    api_type: ProviderType | None = None
    api_retries: int | None = None
    wait_time: float | None = None
    supports_responses_api: bool = True
    """False for OpenAI-compatible gateways that only serve chat completions,
    such as vLLM, Ollama, LiteLLM and Groq."""

    @staticmethod
    def get_default_for_provider(provider: ProviderType) -> "ModelConfigurations":
        if provider == ProviderType.OPENAI:
            settings = settings_for(SupportsOpenAILLMConfig)
            return ModelConfigurations(
                api_host=settings.openai_llm_endpoint or None,
                api_key=settings.openai_llm_api_key or None,
                model_name=settings.openai_llm_model or None,
                api_type=ProviderType.OPENAI,
                api_retries=2,
                wait_time=1.0,
            )
        elif provider == ProviderType.AZURE_OPENAI:
            settings = settings_for(SupportsAzureOpenAILLMConfig)
            return ModelConfigurations(
                api_host=settings.azure_openai_llm_endpoint or None,
                api_key=settings.azure_openai_llm_api_key or None,
                model_name=settings.azure_openai_llm_deployment_name or None,
                api_type=ProviderType.AZURE_OPENAI,
                api_version=settings.azure_openai_api_version or None,
                api_retries=2,
                wait_time=1.0,
            )
        elif provider == ProviderType.ANTHROPIC:
            settings = settings_for(SupportsAnthropicConfig)
            return ModelConfigurations(
                api_host=settings.anthropic_llm_endpoint or None,
                api_key=settings.anthropic_api_key or None,
                model_name=settings.anthropic_llm_model or None,
                api_type=ProviderType.ANTHROPIC,
                api_retries=2,
                wait_time=1.0,
            )
        elif provider == ProviderType.GEMINI:
            settings = settings_for(SupportsGeminiConfig)
            return ModelConfigurations(
                api_host=settings.gemini_llm_endpoint or None,
                api_key=settings.gemini_api_key or None,
                model_name=settings.gemini_llm_model or None,
                api_type=ProviderType.GEMINI,
                api_retries=2,
                wait_time=1.0,
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider}")


class EmbeddingConfigurations(BaseModel):
    api_host: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    api_version: str | None = None
    api_type: ProviderType | None = None
    batch_size: int | None = None
    dimensions: int | None = None

    @staticmethod
    def get_default_for_provider(provider: ProviderType) -> "EmbeddingConfigurations":
        if provider == ProviderType.OPENAI:
            settings = settings_for(SupportsOpenAIEmbeddingConfig)
            return EmbeddingConfigurations(
                api_host=settings.openai_embedding_endpoint or None,
                api_key=settings.openai_embedding_api_key or None,
                model_name=settings.openai_embedding_model or None,
                api_type=ProviderType.OPENAI,
                batch_size=settings.openai_embedding_batch_size,
                dimensions=settings.openai_embedding_dimensions,
            )
        elif provider == ProviderType.AZURE_OPENAI:
            settings = settings_for(SupportsAzureOpenAIEmbeddingConfig)
            return EmbeddingConfigurations(
                api_host=settings.azure_openai_embedding_endpoint or None,
                api_key=settings.azure_openai_embedding_api_key or None,
                model_name=settings.azure_openai_embedding_deployment_name or None,
                api_type=ProviderType.AZURE_OPENAI,
                api_version=settings.azure_openai_api_version or None,
                batch_size=settings.azure_openai_embedding_batch_size,
                dimensions=settings.azure_openai_embedding_dimensions,
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider}")
