from pydantic import BaseModel


class AzureOpenAISettingsMixin(BaseModel):
    """What every Azure OpenAI deployment shares, whatever it serves.

    Declared once and inherited, so a service using both generation and
    embeddings carries one `azure_openai_api_version` rather than two
    declarations whose winner would be decided by MRO order.
    """

    azure_openai_api_version: str = ""
