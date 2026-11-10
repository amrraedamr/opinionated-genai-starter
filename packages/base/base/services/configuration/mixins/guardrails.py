import re

from pydantic import BaseModel, field_validator


class GuardrailSettingsMixin(BaseModel):
    """Which guardrails run, what they look for, and how hard they look."""

    guardrails_enabled: bool = False

    pii_redaction_enabled: bool = True
    moderation_enabled: bool = False

    pii_extra_patterns: dict[str, str] = {}
    """Extra named patterns to redact, as `category` to regular expression."""

    moderation_model: str = "omni-moderation-latest"
    moderation_threshold: float = 0.5
    moderation_categories: list[str] = []
    """API category names worth refusing, e.g. `self-harm`. Empty means every one."""

    injection_block_enabled: bool = True

    pii_stream_tail: int = 48
    """Characters held back while streaming; 48 covers the longest spaced IBAN."""

    @field_validator("pii_extra_patterns")
    @classmethod
    def _compiles(cls, patterns: dict[str, str]) -> dict[str, str]:
        """Refuse a pattern that does not compile at startup, not on first use."""
        for name, pattern in patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"pii_extra_patterns[{name!r}]: {exc}") from exc
        return patterns
