from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsQdrantConfig(Protocol):
    """Satisfied by `QdrantSettingsMixin`."""

    qdrant_url: str
    qdrant_grpc_port: int
    qdrant_prefer_grpc: bool
    qdrant_timeout: int
    qdrant_api_key: str
