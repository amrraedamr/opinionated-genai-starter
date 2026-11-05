from pydantic import BaseModel


class KeyVaultSettingsMixin(BaseModel):
    """Runtime secret API only — not a Settings backend. See `CoreSettings`."""

    keyvault_url: str = ""
    system_keyvault_tag: str = "appconfig"
    model_keyvault_tag: str = "modelconfig"
