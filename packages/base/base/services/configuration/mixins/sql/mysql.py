from pydantic import AliasChoices, BaseModel, Field


class MySQLSettingsMixin(BaseModel):
    mysql_host: str = ""
    mysql_database: str = ""
    mysql_username: str = Field(
        default="",
        validation_alias=AliasChoices("MYSQL_USER", "MYSQL_USERNAME"),
    )
    mysql_password: str = Field(
        default="",
        validation_alias=AliasChoices("MYSQL_PASSWORD"),
    )
