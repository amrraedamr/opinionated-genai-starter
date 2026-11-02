"""chat_api settings: `CoreSettings` plus the mixins this service uses."""

from pydantic_settings import SettingsConfigDict

from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import (
    LLMSettingsMixin,
    PostgresSettingsMixin,
    PromptSettingsMixin,
    RedisSettingsMixin,
    ServerSettingsMixin,
    SQLSettingsMixin,
    TokenVerificationMixin,
)


class ChatApiSettings(
    CoreSettings,
    ServerSettingsMixin,
    SQLSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    LLMSettingsMixin,
    TokenVerificationMixin,
    PromptSettingsMixin,
):
    model_config = SettingsConfigDict(
        # Later files win; real process environment beats every file.
        env_file=("../../.env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "chat_api"
    api_prefix: str = "/api/7r"

    cors_origins: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = ChatApiSettings()


if __name__ == "__main__":  # pragma: no cover
    # Validation entrypoint for CI and container preStart probes.
    print(f"chat_api settings OK (service_name={settings.service_name!r}, environment={settings.environment!r})")
