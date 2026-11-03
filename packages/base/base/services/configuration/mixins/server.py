"""HTTP serving. Pairs with the `server` extra (uvicorn + gunicorn)."""

from pydantic import BaseModel


class ServerSettingsMixin(BaseModel):
    """Only a service that exposes an HTTP surface should include this."""

    api_prefix: str = "/api"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_gunicorn_workers: int = 2

    trusted_hosts: str = "localhost,127.0.0.1,testserver"
    request_max_bytes: int = 1_048_576
    request_concurrency_limit: int = 100
    request_rate_limit_per_minute: int = 300
    auth_rate_limit_per_minute: int = 20
