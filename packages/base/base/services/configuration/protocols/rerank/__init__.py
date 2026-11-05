from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsRerankConfig(Protocol):
    """Satisfied by `RerankSettingsMixin`."""

    rerank_overfetch: int
