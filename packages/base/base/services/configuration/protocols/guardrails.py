from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsGuardrailConfig(Protocol):
    """Satisfied by `GuardrailSettingsMixin`."""

    guardrails_enabled: bool
    pii_redaction_enabled: bool
    moderation_enabled: bool
    pii_extra_patterns: dict[str, str]
    moderation_model: str
    moderation_threshold: float
    moderation_categories: list[str]
    injection_block_enabled: bool
    pii_stream_tail: int
