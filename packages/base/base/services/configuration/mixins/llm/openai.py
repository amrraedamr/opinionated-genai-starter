from pydantic import BaseModel


class OpenAILLMSettingsMixin(BaseModel):
    """OpenAI, for generation. Requires the `llm` extra."""

    openai_llm_endpoint: str = ""
    openai_llm_model: str = ""
    openai_llm_api_key: str = ""
