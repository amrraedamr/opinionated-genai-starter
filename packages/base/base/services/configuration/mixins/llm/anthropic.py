from pydantic import BaseModel


class AnthropicSettingsMixin(BaseModel):
    """Anthropic, for generation. Requires the `anthropic` extra."""

    anthropic_api_key: str = ""
    anthropic_llm_endpoint: str = ""
    anthropic_llm_model: str = ""
    anthropic_max_output_tokens: int = 4096
    """Anthropic requires `max_tokens` on every call; this is the fallback."""
