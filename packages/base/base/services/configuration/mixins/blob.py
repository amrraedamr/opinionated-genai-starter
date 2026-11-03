from pydantic import BaseModel


class BlobSettingsMixin(BaseModel):
    azure_blob_account_url: str = ""
