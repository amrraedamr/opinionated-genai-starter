from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsServerConfig(Protocol):
    """Satisfied by `CoreSettings` plus `ServerSettingsMixin`.

    Required by `create_app` and the webserver, so a service with no HTTP
    surface cannot be passed to either.
    """

    service_name: str
    environment: str

    api_host: str
    api_port: int
    max_gunicorn_workers: int

    trusted_hosts: str
    request_max_bytes: int
    request_concurrency_limit: int
    request_rate_limit_per_minute: int
    auth_rate_limit_per_minute: int
