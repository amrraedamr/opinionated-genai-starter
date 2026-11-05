from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAnthropicConfig(Protocol):
    """Satisfied by `AnthropicSettingsMixin`."""

    anthropic_api_key: str
    anthropic_llm_endpoint: str
    anthropic_llm_model: str
    anthropic_max_output_tokens: int
