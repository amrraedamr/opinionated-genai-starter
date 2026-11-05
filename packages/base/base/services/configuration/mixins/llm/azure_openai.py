from base.services.configuration.mixins.azure_openai import (
    AzureOpenAISettingsMixin,
)


class AzureOpenAILLMSettingsMixin(AzureOpenAISettingsMixin):
    """Azure OpenAI, for generation. Requires the `llm` extra."""

    azure_openai_llm_endpoint: str = ""
    azure_openai_llm_deployment_name: str = ""
    azure_openai_llm_api_key: str = ""
