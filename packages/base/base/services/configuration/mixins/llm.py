from pydantic import BaseModel


class LLMSettingsMixin(BaseModel):
    azure_openai_api_version: str = ""
    azure_openai_llm_endpoint: str = ""
    azure_openai_llm_deployment_name: str = ""
    azure_openai_llm_api_key: str = ""

    openai_llm_endpoint: str = ""
    openai_llm_model: str = ""
    openai_llm_api_key: str = ""

    capture_raw_traces: bool = False
    """Capture raw prompts and responses in the debug trace."""


class AnthropicSettingsMixin(BaseModel):
    """Anthropic. Requires the `anthropic` extra."""

    anthropic_api_key: str = ""
    anthropic_llm_endpoint: str = ""
    anthropic_llm_model: str = ""
    anthropic_max_output_tokens: int = 4096
    """Anthropic requires `max_tokens` on every call; this is the fallback."""
