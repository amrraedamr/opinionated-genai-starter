from enum import StrEnum

from base.tools.errors import MissingExtraError

try:
    from qdrant_client.models import Distance
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "vectors", "qdrant-client", feature="Vector search definitions"
    ) from exc


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


VECTOR_UPSERT_BATCH_SIZE: int = 100
"""Default batch size for vector upsert operations.

Qdrant recommends ~100 points per request. Azure AI Search supports up to
1 000 documents per batch but 100 keeps behavior consistent across backends.
"""

DISTANCE_MAP: dict[str, Distance] = {
    DistanceMetric.COSINE: Distance.COSINE,
    DistanceMetric.DOT: Distance.DOT,
    DistanceMetric.EUCLIDEAN: Distance.EUCLID,
}
