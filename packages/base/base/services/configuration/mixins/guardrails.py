from pydantic import BaseModel


class GuardrailSettingsMixin(BaseModel):
    """Which guardrails run, what they look for, and how hard they look."""

    guardrails_enabled: bool = False
    """Off unless a service asks. Enabling changes what the model is shown."""

    pii_redaction_enabled: bool = True
    moderation_enabled: bool = False
    """Off even when guardrails are on: it bills a provider call per check."""

    pii_extra_patterns: dict[str, str] = {}
    """Extra named patterns to redact, as `category` to regular expression."""

    moderation_model: str = "omni-moderation-latest"
    moderation_threshold: float = 0.5
    moderation_categories: list[str] = []
    """Categories worth refusing. Empty means every category the model scores."""

    injection_block_enabled: bool = True

    pii_stream_tail: int = 32
    """Characters held back between chunks so a value split across two is seen."""
