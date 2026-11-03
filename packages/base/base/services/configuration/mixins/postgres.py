from pydantic import AliasChoices, BaseModel, Field


class PostgresSettingsMixin(BaseModel):
    pg_host: str = ""
    pg_port: str = "5432"
    pg_database: str = ""
    pg_ssl_root_cert: str = ""
    pg_username: str = Field(
        default="",
        validation_alias=AliasChoices("POSTGRES_USER", "PG_USERNAME"),
    )
    pg_password: str = Field(
        default="",
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "PG_PASSWORD"),
    )
