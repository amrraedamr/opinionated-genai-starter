from enum import Enum


class CacheKeys(Enum):
    TEST = "test_cache_key"


class CacheChannels(Enum):
    TEST = "test_channel"
    # Reserved for a future Settings reload (Key Vault source + pub/sub). See Settings / AzureKeyVaultService.
    KEYVAULT_RELOAD = "keyvault_reload"
    LOGS = "logs"
    PROMPT_RELOAD = "prompt_reload"

    def for_app(self, app_name: str) -> str:
        return f"{app_name}:{self.value}"
