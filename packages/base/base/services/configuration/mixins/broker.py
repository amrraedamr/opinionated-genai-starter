from pydantic import BaseModel


class BrokerSettingsMixin(BaseModel):
    """Event transport. `redis` requires the Redis settings and extra as well."""

    broker: str = "memory"
    broker_max_attempts: int = 5
    broker_claim_idle_ms: int = 30_000
    broker_retry_backoff_seconds: float = 0.25
