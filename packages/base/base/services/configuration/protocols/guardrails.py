from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsGuardrailConfig(Protocol):
    """Satisfied by `GuardrailSettingsMixin`."""

    pii_extra_patterns: dict[str, str]
    moderation_model: str
    moderation_threshold: float
    moderation_categories: list[str]
    injection_block_enabled: bool
