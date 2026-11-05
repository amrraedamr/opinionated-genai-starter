from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsMongoConfig(Protocol):
    """Satisfied by `MongoSettingsMixin`."""

    mongodb_username: str
    mongodb_password: str
    mongodb_base_path: str
    mongodb_host: str
    mongodb_database: str
    mongodb_params: str
    mongodb_min_pool_size: int
    mongodb_max_pool_size: int
    mongodb_connect_timeout_ms: int
    mongodb_socket_timeout_ms: int
    mongodb_server_selection_timeout_ms: int
