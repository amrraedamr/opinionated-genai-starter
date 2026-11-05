from pydantic import BaseModel


class GeminiSettingsMixin(BaseModel):
    """Google Gemini, for generation. Requires the `gemini` extra."""

    gemini_api_key: str = ""
    gemini_llm_endpoint: str = ""
    gemini_llm_model: str = ""

    gemini_use_vertex: bool = False
    """Talk to Vertex AI rather than the Gemini Developer API. Vertex reads
    credentials from the environment instead of an API key, and needs a
    project and a location."""

    gemini_project: str = ""
    gemini_location: str = "global"
