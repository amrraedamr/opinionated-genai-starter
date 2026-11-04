from pydantic import BaseModel


class OpenAILLMSettingsMixin(BaseModel):
    """OpenAI, for generation. Requires the `llm` extra."""

    openai_llm_endpoint: str = ""
    openai_llm_model: str = ""
    openai_llm_api_key: str = ""


class OpenAIEmbeddingSettingsMixin(BaseModel):
    """OpenAI, for embeddings. Requires the `llm` extra."""

    openai_embedding_endpoint: str = ""
    openai_embedding_model: str = ""
    openai_embedding_api_key: str = ""
    openai_embedding_dimensions: int = 1536
    openai_embedding_batch_size: int = 2048
