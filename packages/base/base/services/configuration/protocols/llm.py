from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsLLMConfig(Protocol):
    """Satisfied by `LLMSettingsMixin`."""

    azure_openai_api_version: str
    azure_openai_llm_endpoint: str
    azure_openai_llm_deployment_name: str
    azure_openai_llm_api_key: str

    openai_llm_endpoint: str
    openai_llm_model: str
    openai_llm_api_key: str

    capture_raw_traces: bool
