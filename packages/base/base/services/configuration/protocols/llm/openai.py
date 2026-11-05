from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsOpenAILLMConfig(Protocol):
    """Satisfied by `OpenAILLMSettingsMixin`."""

    openai_llm_endpoint: str
    openai_llm_model: str
    openai_llm_api_key: str
