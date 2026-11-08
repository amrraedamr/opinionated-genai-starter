from pydantic_settings import BaseSettings, SettingsConfigDict

from base.models.definitions.environment import Environment
from base.services.common.meta import SingletonSettings


class CoreSettings(BaseSettings, metaclass=SingletonSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str
    environment: Environment = Environment.LOCAL

    log_level: str = "INFO"
