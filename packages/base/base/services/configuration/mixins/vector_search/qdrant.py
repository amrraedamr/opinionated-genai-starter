from pydantic import BaseModel


class QdrantSettingsMixin(BaseModel):
    """Qdrant vector search. Azure AI Search is a separate mixin — a service
    picks the vector provider it actually runs."""

    qdrant_url: str = ""
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_timeout: int = 30
    qdrant_api_key: str = ""
