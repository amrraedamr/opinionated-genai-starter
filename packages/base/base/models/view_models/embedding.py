"""What an embedding call returned."""

from base.models.view_models import BaseModel
from base.models.view_models.usage import ProviderUsage


class EmbeddingBatch(BaseModel):
    """One request's vectors, and what it cost.

    Usage travels with the vectors so the abstraction can record the call
    without each adapter remembering to.
    """

    vectors: list[list[float]]
    usage: ProviderUsage | None = None
