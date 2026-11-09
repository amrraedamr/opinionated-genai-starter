"""HTTP serving. Pairs with the `server` extra (uvicorn + gunicorn)."""

from pydantic import BaseModel


class ServerSettingsMixin(BaseModel):
    """Only a service that exposes an HTTP surface should include this.

    Rate-limit budgets are not here. They live on `RateLimitSettingsMixin`,
    because a worker spends against them without serving anything, and a
    service that does not compose it gets that mixin's defaults.
    """

    api_prefix: str = "/api"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_gunicorn_workers: int = 2

    trusted_hosts: str = "localhost,127.0.0.1,testserver"
    request_max_bytes: int = 1_048_576
    request_concurrency_limit: int = 100
