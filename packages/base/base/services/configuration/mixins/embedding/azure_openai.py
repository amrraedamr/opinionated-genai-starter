from base.services.configuration.mixins.azure_openai import (
    AzureOpenAISettingsMixin,
)


class AzureOpenAIEmbeddingSettingsMixin(AzureOpenAISettingsMixin):
    """Azure OpenAI, for embeddings. Requires the `llm` extra."""

    azure_openai_embedding_endpoint: str = ""
    azure_openai_embedding_deployment_name: str = ""
    azure_openai_embedding_api_key: str = ""
    azure_openai_embedding_dimensions: int = 3072
    azure_openai_embedding_batch_size: int = 2048
