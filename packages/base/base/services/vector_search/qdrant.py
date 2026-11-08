from typing import Any, ClassVar, cast
from uuid import NAMESPACE_OID, UUID, uuid5

from base.tools.errors import MissingExtraError

try:
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
    from qdrant_client.models import (
        Condition,
        Distance,
        ExtendedPointId,
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        PointIdsList,
        PointStruct,
        VectorParams,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "vectors", "qdrant-client", feature="QdrantVectorSearchService"
    ) from exc

from base.clients.vectors.qdrant import QdrantClient
from base.models.definitions.vector_search import (
    VECTOR_UPSERT_BATCH_SIZE,
    DistanceMetric,
)
from base.models.view_models.vector_search import VectorRecord, VectorSearchResult
from base.services.common.meta import SingletonABC
from base.services.error_handling import aretrying, retrying
from base.services.error_handling.vector_search.qdrant import QdrantErrorHandler
from base.services.vector_search import AbstractVectorSearchService
from base.tools.errors import ServiceHealthCheckError

# Qdrant's own mapping. In core it made every vector backend import
# qdrant-client.
_DISTANCE_MAP: dict[str, Distance] = {
    DistanceMetric.COSINE: Distance.COSINE,
    DistanceMetric.DOT: Distance.DOT,
    DistanceMetric.EUCLIDEAN: Distance.EUCLID,
}


class QdrantVectorSearchService(AbstractVectorSearchService, metaclass=SingletonABC):
    """Qdrant-backed vector search service with sync and async support."""

    SOURCE_ID_FIELD: ClassVar[str] = "_source_id"

    def __init__(self) -> None:
        super().__init__()
        self._client: QdrantClient = QdrantClient()

        # Collection Management
        self._error_handler: QdrantErrorHandler = QdrantErrorHandler()

    # Point ids

    @classmethod
    def _point_id(cls, record_id: str) -> ExtendedPointId:
        """Qdrant takes a UUID or an unsigned integer; any other id becomes a UUID."""
        if record_id.isdigit():
            return int(record_id)
        try:
            return str(UUID(record_id))
        except ValueError:
            return str(uuid5(NAMESPACE_OID, record_id))

    @classmethod
    def _payload_for(cls, record: VectorRecord) -> dict[str, Any]:
        """Keep the caller's id alongside the metadata when it had to be rewritten."""
        payload = dict(record.metadata)
        if str(cls._point_id(record.id)) != record.id:
            payload[cls.SOURCE_ID_FIELD] = record.id
        return payload

    @classmethod
    def _read_payload(cls, payload: Any, point_id: Any) -> tuple[str, dict[str, Any]]:
        """Split a stored payload back into the caller's id and their metadata."""
        metadata = dict(payload) if payload else {}
        return metadata.pop(cls.SOURCE_ID_FIELD, str(point_id)), metadata

    def _as_record(self, point: Any) -> VectorRecord:
        record_id, metadata = self._read_payload(point.payload, point.id)
        return VectorRecord(
            id=record_id,
            vector=self._extract_vector(point.vector),
            metadata=metadata,
        )

    def _as_result(self, point: Any) -> VectorSearchResult:
        record_id, metadata = self._read_payload(point.payload, point.id)
        return VectorSearchResult(
            id=record_id,
            score=point.score,
            vector=self._extract_vector(point.vector),
            metadata=metadata,
        )

    @retrying
    def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        self._client.sync_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=_DISTANCE_MAP.get(
                    distance, _DISTANCE_MAP[DistanceMetric.COSINE]
                ),
            ),
            **kwargs,
        )

    @aretrying
    async def acreate_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        await self._client.async_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=_DISTANCE_MAP.get(
                    distance, _DISTANCE_MAP[DistanceMetric.COSINE]
                ),
            ),
            **kwargs,
        )

    @retrying
    def delete_collection(self, collection: str) -> None:
        self._client.sync_client.delete_collection(collection_name=collection)

    @aretrying
    async def adelete_collection(self, collection: str) -> None:
        await self._client.async_client.delete_collection(collection_name=collection)

    @retrying
    def collection_exists(self, collection: str) -> bool:
        return self._client.sync_client.collection_exists(collection_name=collection)

    @aretrying
    async def acollection_exists(self, collection: str) -> bool:
        return await self._client.async_client.collection_exists(
            collection_name=collection
        )

    # Upsert

    @retrying
    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        for i in range(0, len(records), VECTOR_UPSERT_BATCH_SIZE):
            batch = records[i : i + VECTOR_UPSERT_BATCH_SIZE]
            self._client.sync_client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=self._point_id(r.id),
                        vector=r.vector,
                        payload=self._payload_for(r),
                    )
                    for r in batch
                ],
            )

    @aretrying
    async def aupsert(self, collection: str, records: list[VectorRecord]) -> None:
        for i in range(0, len(records), VECTOR_UPSERT_BATCH_SIZE):
            batch = records[i : i + VECTOR_UPSERT_BATCH_SIZE]
            await self._client.async_client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=self._point_id(r.id),
                        vector=r.vector,
                        payload=self._payload_for(r),
                    )
                    for r in batch
                ],
            )

    # Search

    @retrying
    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        results = self._client.sync_client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            query_filter=self._build_filter(filters),
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=True,
        )
        return [self._as_result(point) for point in results.points]

    @aretrying
    async def asearch(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        results = await self._client.async_client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            query_filter=self._build_filter(filters),
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=True,
        )
        return [self._as_result(point) for point in results.points]

    # Get by ID

    @retrying
    def get(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        records = self._client.sync_client.retrieve(
            collection_name=collection,
            ids=[self._point_id(i) for i in ids],
            with_vectors=with_vectors,
        )
        return [self._as_record(r) for r in records]

    @aretrying
    async def aget(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        records = await self._client.async_client.retrieve(
            collection_name=collection,
            ids=[self._point_id(i) for i in ids],
            with_vectors=with_vectors,
        )
        return [self._as_record(r) for r in records]

    # Delete

    @retrying
    def delete(self, collection: str, ids: list[str]) -> None:
        self._client.sync_client.delete(
            collection_name=collection,
            points_selector=PointIdsList(
                points=cast(list[ExtendedPointId], [self._point_id(i) for i in ids])
            ),
        )

    @aretrying
    async def adelete(self, collection: str, ids: list[str]) -> None:
        await self._client.async_client.delete(
            collection_name=collection,
            points_selector=PointIdsList(
                points=cast(list[ExtendedPointId], [self._point_id(i) for i in ids])
            ),
        )

    # Count

    @retrying
    def count(self, collection: str) -> int:
        result = self._client.sync_client.count(collection_name=collection)
        return result.count

    @aretrying
    async def acount(self, collection: str) -> int:
        result = await self._client.async_client.count(collection_name=collection)
        return result.count

    # Scroll

    @retrying
    def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        records, next_offset = self._client.sync_client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            scroll_filter=self._build_filter(filters),
            with_vectors=with_vectors,
        )
        return (
            [self._as_record(r) for r in records],
            str(next_offset) if next_offset is not None else None,
        )

    @aretrying
    async def ascroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        records, next_offset = await self._client.async_client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            scroll_filter=self._build_filter(filters),
            with_vectors=with_vectors,
        )
        return (
            [self._as_record(r) for r in records],
            str(next_offset) if next_offset is not None else None,
        )

    # Health Check

    def health_check(self) -> bool:
        try:
            self._client.sync_client.get_collections()
            return True
        except (UnexpectedResponse, ResponseHandlingException, OSError) as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e

    async def ahealth_check(self) -> bool:
        try:
            await self._client.async_client.get_collections()
            return True
        except (UnexpectedResponse, ResponseHandlingException, OSError) as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e

    # Metadata & Lifecycle

    def get_service_name(self) -> str:
        return "Qdrant"

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()

    # Internals

    @staticmethod
    def _extract_vector(vector: Any) -> list[float]:
        """Extract a flat float vector from Qdrant's ``VectorStructOutput``.

        Qdrant's vector field is a union of ``List[float]``,
        ``List[List[float]]`` (multi-vectors), or
        ``Dict[str, ...]`` (named vectors).  We only handle the single
        unnamed-vector case; everything else returns ``[]``.
        """
        if isinstance(vector, list) and vector and isinstance(vector[0], (int, float)):
            return list(vector)
        return []

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
        """Convert simple ``{field: value}`` dict to a Qdrant ``Filter``.

        - Scalar values become ``MatchValue`` conditions.
        - List values become ``MatchAny`` conditions.
        """
        if not filters:
            return None
        conditions: list[Condition] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions)
