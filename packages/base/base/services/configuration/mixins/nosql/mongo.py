from pydantic import AliasChoices, BaseModel, Field


class MongoSettingsMixin(BaseModel):
    mongodb_username: str = Field(
        default="",
        validation_alias=AliasChoices("MONGODB_USER", "MONGODB_USERNAME"),
    )
    mongodb_password: str = Field(
        default="",
        validation_alias=AliasChoices("MONGODB_PASSWORD"),
    )
    mongodb_base_path: str = "mongodb"
    mongodb_host: str = "localhost"
    mongodb_database: str = ""
    mongodb_params: str = ""
    mongodb_min_pool_size: int = 5
    mongodb_max_pool_size: int = 50
    mongodb_connect_timeout_ms: int = 10_000
    mongodb_socket_timeout_ms: int = 30_000
    mongodb_server_selection_timeout_ms: int = 30_000
