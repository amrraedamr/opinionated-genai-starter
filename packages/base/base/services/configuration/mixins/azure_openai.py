from pydantic import BaseModel


class AzureOpenAISettingsMixin(BaseModel):
    """What every Azure OpenAI deployment shares, whatever it serves.

    Declared once and inherited, so a service using both generation and
    embeddings carries one `azure_openai_api_version` rather than two
    declarations whose winner would be decided by MRO order.
    """

    azure_openai_api_version: str = ""


class AzureOpenAILLMSettingsMixin(AzureOpenAISettingsMixin):
    """Azure OpenAI, for generation. Requires the `llm` extra."""

    azure_openai_llm_endpoint: str = ""
    azure_openai_llm_deployment_name: str = ""
    azure_openai_llm_api_key: str = ""


class AzureOpenAIEmbeddingSettingsMixin(AzureOpenAISettingsMixin):
    """Azure OpenAI, for embeddings. Requires the `llm` extra."""

    azure_openai_embedding_endpoint: str = ""
    azure_openai_embedding_deployment_name: str = ""
    azure_openai_embedding_api_key: str = ""
    azure_openai_embedding_dimensions: int = 3072
    azure_openai_embedding_batch_size: int = 2048
