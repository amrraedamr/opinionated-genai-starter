from pydantic import BaseModel


class EmbeddingSettingsMixin(BaseModel):
    azure_openai_api_version: str = ""
    azure_openai_embedding_endpoint: str = ""
    azure_openai_embedding_deployment_name: str = ""
    azure_openai_embedding_api_key: str = ""
    azure_openai_embedding_dimensions: int = 3072
    azure_openai_embedding_batch_size: int = 2048

    openai_embedding_endpoint: str = ""
    openai_embedding_model: str = ""
    openai_embedding_api_key: str = ""
    openai_embedding_dimensions: int = 1536
    openai_embedding_batch_size: int = 2048
