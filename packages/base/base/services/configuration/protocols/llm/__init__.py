from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsLLMConfig(Protocol):
    """Satisfied by `LLMSettingsMixin`."""

    capture_raw_traces: bool
