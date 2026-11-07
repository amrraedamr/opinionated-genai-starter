from base.models.definitions.model import CachePersistence, ProviderType
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


class CacheHint(BaseModel):
    """Says a prompt's instructions and tools are worth caching.

    The hint marks the head of the prompt, which is the part that repeats:
    the same instructions and the same tools go out on every call of a
    conversation, while the messages after them never repeat. Where the
    boundary falls is the only thing a caller has to decide, because it is
    the only thing the providers agree on.

    What each does with it differs. Anthropic caches nothing unless a request
    marks where, so the hint becomes that mark. OpenAI and Gemini cache a
    repeated prefix on their own, so the hint cannot turn caching on; it can
    only improve the odds, which OpenAI exposes as a key that routes calls
    sharing a prefix together, and Gemini does not expose at all.
    """

    persistence: CachePersistence = CachePersistence.SHORT
    key: str | None = None
    """Groups calls sharing this head, for providers that route on one.

    Derived from the head itself when unset, which is usually what is wanted:
    two calls cache together exactly when their heads match.
    """


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
