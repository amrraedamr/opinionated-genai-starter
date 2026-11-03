from pydantic_settings import SettingsConfigDict

from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    SQLSettingsMixin,
)


class IngestionWorkerSettings(
    CoreSettings,
    SQLSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    BrokerSettingsMixin,
):
    model_config = SettingsConfigDict(
        env_file=("../../.env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "ingestion_worker"
    consumer_group: str = "ingestion_worker"
    sqlite_db_path: str = "database/ingestion_worker.db"
    inbox_claim_timeout_seconds: int = 300


settings = IngestionWorkerSettings()


if __name__ == "__main__":
    print(f"ingestion_worker settings OK (environment={settings.environment!r})")
