"""The answer to one spend against one budget."""

from pydantic import Field

from base.models.view_models import BaseModel


class RateDecision(BaseModel):
    """Whether a spend was allowed, and what the caller should do next."""

    allowed: bool

    remaining: int = Field(ge=0)
    """Units still available on this budget, after the spend when it was allowed."""

    retry_after_seconds: float = Field(default=0.0, ge=0.0)
    """How long until the refused spend would fit. Zero when it was allowed."""

    @property
    def retry_after_header(self) -> str:
        """`Retry-After` wants whole seconds, and never zero when refused."""
        return str(max(1, int(self.retry_after_seconds + 0.999)))
