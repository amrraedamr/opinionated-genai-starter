from enum import StrEnum


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


VECTOR_UPSERT_BATCH_SIZE: int = 100
"""Default batch size for vector upsert operations.

Qdrant recommends ~100 points per request. Azure AI Search supports up to
1 000 documents per batch but 100 keeps behavior consistent across backends.
"""
