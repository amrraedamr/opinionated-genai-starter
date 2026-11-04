from typing import Protocol, runtime_checkable

from base.models.definitions.broker import BrokerBackend


@runtime_checkable
class SupportsBrokerConfig(Protocol):
    """Satisfied by `BrokerSettingsMixin`."""

    broker: BrokerBackend
    broker_max_attempts: int
    broker_claim_idle_ms: int
    broker_retry_backoff_seconds: float
