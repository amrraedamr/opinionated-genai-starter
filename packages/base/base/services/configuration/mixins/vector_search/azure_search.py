from pydantic import BaseModel


class AzureSearchSettingsMixin(BaseModel):
    azure_search_endpoint: str = ""
    azure_search_vector_field: str = "vector"
    azure_search_key_field: str = "id"
    azure_search_api_key: str = ""
