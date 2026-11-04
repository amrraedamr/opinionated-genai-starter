from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsRerankConfig(Protocol):
    """Satisfied by `RerankSettingsMixin`."""

    rerank_overfetch: int


@runtime_checkable
class SupportsCohereConfig(Protocol):
    """Satisfied by `CohereSettingsMixin`."""

    cohere_api_key: str
    cohere_rerank_model: str
