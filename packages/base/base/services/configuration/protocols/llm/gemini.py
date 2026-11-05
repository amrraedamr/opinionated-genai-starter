from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsGeminiConfig(Protocol):
    """Satisfied by `GeminiSettingsMixin`."""

    gemini_api_key: str
    gemini_llm_endpoint: str
    gemini_llm_model: str
    gemini_use_vertex: bool
    gemini_project: str
    gemini_location: str
