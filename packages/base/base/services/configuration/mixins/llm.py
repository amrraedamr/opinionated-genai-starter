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
