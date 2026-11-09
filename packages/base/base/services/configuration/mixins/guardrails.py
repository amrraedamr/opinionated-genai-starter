from pydantic import BaseModel


class GuardrailSettingsMixin(BaseModel):
    """What the built-in guardrails look for, and how hard they look."""

    pii_extra_patterns: dict[str, str] = {}
    """Extra named patterns to redact, as `category` to regular expression."""

    moderation_model: str = "omni-moderation-latest"
    moderation_threshold: float = 0.5
    moderation_categories: list[str] = []
    """Categories worth refusing. Empty means every category the model scores."""

    injection_block_enabled: bool = True
