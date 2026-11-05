from pydantic import BaseModel


class AzureBlobSettingsMixin(BaseModel):
    """Azure Blob Storage. Requires the `azure` extra."""

    azure_blob_account_url: str = ""
