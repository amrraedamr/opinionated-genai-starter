from pydantic import BaseModel


class CosmosSettingsMixin(BaseModel):
    azure_cosmos_account_uri: str = ""
    azure_cosmos_account_key: str = ""
    azure_cosmos_database_name: str = ""
