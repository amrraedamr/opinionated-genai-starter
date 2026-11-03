"""identity settings: `CoreSettings` plus the mixins this service uses."""

from pydantic_settings import SettingsConfigDict

from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    IdpResourceServerMixin,
    OAuthSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    ServerSettingsMixin,
    SQLSettingsMixin,
    TokenIssuingMixin,
)


class IdentitySettings(
    CoreSettings,
    ServerSettingsMixin,
    SQLSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    BrokerSettingsMixin,
    TokenIssuingMixin,
    OAuthSettingsMixin,
    IdpResourceServerMixin,
):
    model_config = SettingsConfigDict(
        env_file=("../../.env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "identity"
    api_prefix: str = "/api/identity"


settings = IdentitySettings()


if __name__ == "__main__":  # pragma: no cover
    # Validation entrypoint for CI and container preStart probes.
    print(f"identity settings OK (environment={settings.environment!r})")
