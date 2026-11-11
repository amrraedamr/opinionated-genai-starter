"""mcp_server settings: `CoreSettings` plus the mixins this service uses."""

from pydantic_settings import SettingsConfigDict

from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import (
    GuardrailSettingsMixin,
    McpServerSettingsMixin,
    RateLimitSettingsMixin,
    ServerSettingsMixin,
    TelemetrySettingsMixin,
    TenancySettingsMixin,
    TokenVerificationSettingsMixin,
)


class McpServerSettings(
    CoreSettings,
    ServerSettingsMixin,
    RateLimitSettingsMixin,
    TokenVerificationSettingsMixin,
    TenancySettingsMixin,
    TelemetrySettingsMixin,
    GuardrailSettingsMixin,
    McpServerSettingsMixin,
):
    model_config = SettingsConfigDict(
        # Later files win; real process environment beats every file.
        env_file=("../../.env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "mcp_server"
    api_prefix: str = "/api/mcp_server"
    mcp_server_enabled: bool = True
    """On here: offering tools over MCP is this service's whole purpose."""


settings = McpServerSettings()


if __name__ == "__main__":  # pragma: no cover
    # Validation entrypoint for CI and container preStart probes.
    print(f"mcp_server settings OK (environment={settings.environment!r})")
