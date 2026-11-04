"""pgvector-backed vector search, on the Postgres the service already runs.

Needs no Python package beyond the `sql` and `postgres` extras: vectors are
passed in pgvector's text form and cast in SQL. The database does need the
extension, which `create_collection` installs if it is missing.
"""

import json
import re
from typing import Any

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import text
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "sql", "sqlalchemy", feature="PgVectorSearchService"
    ) from exc

from base.models.definitions.vector_search import (
    VECTOR_UPSERT_BATCH_SIZE,
    DistanceMetric,
)
from base.models.view_models.vector_search import VectorRecord, VectorSearchResult
from base.services.common.meta import SingletonABC
from base.services.sql.service import SQLService
from base.services.vector_search import AbstractVectorSearchService

# pgvector's operator and index opclass per metric, plus the expression that
# turns the distance Postgres returns into a score where higher means closer.
_DISTANCE_MAP: dict[str, tuple[str, str, str]] = {
    DistanceMetric.COSINE: ("<=>", "vector_cosine_ops", "1 - ({distance})"),
    DistanceMetric.DOT: ("<#>", "vector_ip_ops", "-({distance})"),
    DistanceMetric.EUCLIDEAN: ("<->", "vector_l2_ops", "-({distance})"),
}


class PgVectorSearchService(AbstractVectorSearchService, metaclass=SingletonABC):
    """Vector search over pgvector, sharing the service's Postgres pool."""

    _IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    # Index opclass -> metric, for recovering a collection's metric from the
    # database rather than assuming one at query time.
    _OPCLASS_METRIC: dict[str, str] = {
        "vector_cosine_ops": DistanceMetric.COSINE,
        "vector_ip_ops": DistanceMetric.DOT,
        "vector_l2_ops": DistanceMetric.EUCLIDEAN,
    }

    _METRIC_SQL = (
        "SELECT opc.opcname AS opcname "
        "FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_class rel ON rel.oid = i.indrelid "
        "JOIN pg_am am ON am.oid = c.relam "
        "JOIN pg_opclass opc ON opc.oid = i.indclass[0] "
        "WHERE rel.relname = :name AND am.amname IN ('hnsw', 'ivfflat') "
        "LIMIT 1"
    )

    def __init__(self) -> None:
        super().__init__()
        self._sql: SQLService = SQLService()
        self._metrics: dict[str, str] = {}

    @classmethod
    def _table(cls, collection: str) -> str:
        """Validate a collection name usable as a table identifier.

        Table names cannot be bound as parameters, so the name is checked
        against a strict pattern rather than escaped.
        """
        if not cls._IDENTIFIER.match(collection):
            raise ValueError(
                f"invalid collection name {collection!r}: letters, digits and "
                f"underscores only, not starting with a digit"
            )
        return f'"{collection}"'

    @staticmethod
    def _literal(vector: list[float]) -> str:
        """pgvector's text form, which the server casts to `vector`."""
        return "[" + ",".join(repr(float(v)) for v in vector) + "]"

    @staticmethod
    def _metric(distance: str) -> tuple[str, str, str]:
        return _DISTANCE_MAP.get(distance, _DISTANCE_MAP[DistanceMetric.COSINE])

    @staticmethod
    def _metric_name(distance: str) -> str:
        return distance if distance in _DISTANCE_MAP else DistanceMetric.COSINE

    @staticmethod
    def _where(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        """Equality filters over the metadata column. Lists mean any-of."""
        if not filters:
            return "", {}
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for index, (field, value) in enumerate(filters.items()):
            key = f"f{index}"
            params[f"{key}_field"] = field
            if isinstance(value, list | tuple | set):
                params[key] = [str(v) for v in value]
                clauses.append(f"metadata ->> :{key}_field = ANY(:{key})")
            else:
                params[key] = str(value)
                clauses.append(f"metadata ->> :{key}_field = :{key}")
        return " AND ".join(clauses), params

    def _resolve_metric(self, collection: str, opcname: str | None) -> str:
        """Cache and return the metric a collection was indexed with."""
        metric = self._OPCLASS_METRIC.get(opcname or "", DistanceMetric.COSINE)
        self._metrics[collection] = metric
        return metric

    def _metric_for(self, collection: str) -> str:
        if collection in self._metrics:
            return self._metrics[collection]
        with self._sql.transaction() as session:
            row = session.execute(
                text(self._METRIC_SQL), {"name": collection}
            ).scalar()
        return self._resolve_metric(collection, row)

    async def _ametric_for(self, collection: str) -> str:
        if collection in self._metrics:
            return self._metrics[collection]
        async with self._sql.atransaction() as session:
            row = (
                await session.execute(text(self._METRIC_SQL), {"name": collection})
            ).scalar()
        return self._resolve_metric(collection, row)

    # Collection management

    def _create_sql(self, collection: str, vector_size: int, distance: str) -> list[str]:
        table = self._table(collection)
        _, index_ops, _ = self._metric(distance)
        return [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"  id TEXT PRIMARY KEY,"
            f"  embedding vector({int(vector_size)}) NOT NULL,"
            f"  metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb)",
            f'CREATE INDEX IF NOT EXISTS "{collection}_embedding_idx" '
            f"ON {table} USING hnsw (embedding {index_ops})",
            f'CREATE INDEX IF NOT EXISTS "{collection}_metadata_idx" '
            f"ON {table} USING gin (metadata)",
            f'CREATE INDEX IF NOT EXISTS "{collection}_text_idx" ON {table} '
            f"USING gin (to_tsvector('simple', coalesce(metadata ->> 'text', '')))",
        ]

    def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        del kwargs
        with self._sql.transaction() as session:
            for statement in self._create_sql(collection, vector_size, distance):
                session.execute(text(statement))
        self._metrics[collection] = self._metric_name(distance)

    async def acreate_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        del kwargs
        async with self._sql.atransaction() as session:
            for statement in self._create_sql(collection, vector_size, distance):
                await session.execute(text(statement))
        self._metrics[collection] = self._metric_name(distance)

    def delete_collection(self, collection: str) -> None:
        with self._sql.transaction() as session:
            session.execute(text(f"DROP TABLE IF EXISTS {self._table(collection)}"))
        self._metrics.pop(collection, None)

    async def adelete_collection(self, collection: str) -> None:
        async with self._sql.atransaction() as session:
            await session.execute(text(f"DROP TABLE IF EXISTS {self._table(collection)}"))
        self._metrics.pop(collection, None)

    _EXISTS = "SELECT to_regclass(:name) IS NOT NULL AS present"

    def collection_exists(self, collection: str) -> bool:
        with self._sql.transaction() as session:
            result = session.execute(text(self._EXISTS), {"name": collection})
            return bool(result.scalar())

    async def acollection_exists(self, collection: str) -> bool:
        async with self._sql.atransaction() as session:
            result = await session.execute(text(self._EXISTS), {"name": collection})
            return bool(result.scalar())

    # Upsert

    def _upsert_sql(self, collection: str) -> str:
        return (
            f"INSERT INTO {self._table(collection)} (id, embedding, metadata) "
            f"VALUES (:id, CAST(:embedding AS vector), CAST(:metadata AS jsonb)) "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata"
        )

    @classmethod
    def _upsert_params(cls, records: list[VectorRecord]) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "embedding": cls._literal(r.vector),
                "metadata": json.dumps(r.metadata or {}),
            }
            for r in records
        ]

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        statement = text(self._upsert_sql(collection))
        with self._sql.transaction() as session:
            for i in range(0, len(records), VECTOR_UPSERT_BATCH_SIZE):
                batch = records[i : i + VECTOR_UPSERT_BATCH_SIZE]
                session.execute(statement, self._upsert_params(batch))

    async def aupsert(self, collection: str, records: list[VectorRecord]) -> None:
        statement = text(self._upsert_sql(collection))
        async with self._sql.atransaction() as session:
            for i in range(0, len(records), VECTOR_UPSERT_BATCH_SIZE):
                batch = records[i : i + VECTOR_UPSERT_BATCH_SIZE]
                await session.execute(statement, self._upsert_params(batch))

    # Search

    def _search_sql(
        self,
        collection: str,
        metric: str,
        limit: int,
        filters: dict[str, Any] | None,
        score_threshold: float | None,
    ) -> tuple[str, dict[str, Any]]:
        operator, _, score_expr = self._metric(metric)
        distance = f"embedding {operator} CAST(:query AS vector)"
        score = score_expr.format(distance=distance)
        where, params = self._where(filters)
        conditions = [where] if where else []
        if score_threshold is not None:
            conditions.append(f"{score} >= :score_threshold")
            params["score_threshold"] = score_threshold
        clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params["limit"] = limit
        return (
            f"SELECT id, metadata, {score} AS score FROM {self._table(collection)}"
            f"{clause} ORDER BY {distance} LIMIT :limit",
            params,
        )

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        sql, params = self._search_sql(
            collection, self._metric_for(collection), limit, filters, score_threshold
        )
        params["query"] = self._literal(vector)
        with self._sql.transaction() as session:
            rows = session.execute(text(sql), params).mappings().all()
        return [self._to_result(row) for row in rows]

    async def asearch(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        sql, params = self._search_sql(
            collection,
            await self._ametric_for(collection),
            limit,
            filters,
            score_threshold,
        )
        params["query"] = self._literal(vector)
        async with self._sql.atransaction() as session:
            rows = (await session.execute(text(sql), params)).mappings().all()
        return [self._to_result(row) for row in rows]

    @staticmethod
    def _to_result(row: Any) -> VectorSearchResult:
        metadata = row["metadata"]
        return VectorSearchResult(
            id=row["id"],
            score=float(row["score"]),
            metadata=metadata if isinstance(metadata, dict) else json.loads(metadata or "{}"),
        )

    # Get by id

    def _get_sql(self, collection: str, with_vectors: bool) -> str:
        columns = "id, metadata" + (", embedding::text AS embedding" if with_vectors else "")
        return f"SELECT {columns} FROM {self._table(collection)} WHERE id = ANY(:ids)"

    def get(
        self, collection: str, ids: list[str], with_vectors: bool = False
    ) -> list[VectorRecord]:
        with self._sql.transaction() as session:
            rows = session.execute(
                text(self._get_sql(collection, with_vectors)), {"ids": list(ids)}
            ).all()
        return [self._to_record(row, with_vectors) for row in rows]

    async def aget(
        self, collection: str, ids: list[str], with_vectors: bool = False
    ) -> list[VectorRecord]:
        async with self._sql.atransaction() as session:
            rows = (
                await session.execute(
                    text(self._get_sql(collection, with_vectors)), {"ids": list(ids)}
                )
            ).all()
        return [self._to_record(row, with_vectors) for row in rows]

    @staticmethod
    def _to_record(row: Any, with_vectors: bool) -> VectorRecord:
        metadata = row.metadata
        return VectorRecord(
            id=row.id,
            vector=json.loads(row.embedding) if with_vectors else [],
            metadata=metadata if isinstance(metadata, dict) else json.loads(metadata or "{}"),
        )

    # Delete

    def delete(self, collection: str, ids: list[str]) -> None:
        sql = f"DELETE FROM {self._table(collection)} WHERE id = ANY(:ids)"
        with self._sql.transaction() as session:
            session.execute(text(sql), {"ids": list(ids)})

    async def adelete(self, collection: str, ids: list[str]) -> None:
        sql = f"DELETE FROM {self._table(collection)} WHERE id = ANY(:ids)"
        async with self._sql.atransaction() as session:
            await session.execute(text(sql), {"ids": list(ids)})

    # Count

    def count(self, collection: str) -> int:
        sql = f"SELECT count(*) FROM {self._table(collection)}"
        with self._sql.transaction() as session:
            return int(session.execute(text(sql)).scalar() or 0)

    async def acount(self, collection: str) -> int:
        sql = f"SELECT count(*) FROM {self._table(collection)}"
        async with self._sql.atransaction() as session:
            return int((await session.execute(text(sql))).scalar() or 0)

    # Scroll

    def _scroll_sql(
        self,
        collection: str,
        limit: int,
        offset: str | None,
        filters: dict[str, Any] | None,
        with_vectors: bool,
    ) -> tuple[str, dict[str, Any]]:
        columns = "id, metadata" + (", embedding::text AS embedding" if with_vectors else "")
        where, params = self._where(filters)
        conditions = [where] if where else []
        if offset is not None:
            conditions.append("id > :offset")
            params["offset"] = offset
        clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params["limit"] = limit
        return (
            f"SELECT {columns} FROM {self._table(collection)}{clause} "
            f"ORDER BY id LIMIT :limit",
            params,
        )

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        sql, params = self._scroll_sql(collection, limit, offset, filters, with_vectors)
        with self._sql.transaction() as session:
            rows = session.execute(text(sql), params).all()
        return self._page(rows, limit, with_vectors)

    async def ascroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        sql, params = self._scroll_sql(collection, limit, offset, filters, with_vectors)
        async with self._sql.atransaction() as session:
            rows = (await session.execute(text(sql), params)).all()
        return self._page(rows, limit, with_vectors)

    def _page(
        self, rows: list[Any], limit: int, with_vectors: bool
    ) -> tuple[list[VectorRecord], str | None]:
        records = [self._to_record(row, with_vectors) for row in rows]
        # Keyset paging on the primary key, so a full page implies another.
        next_offset = records[-1].id if len(records) == limit else None
        return records, next_offset

    # Lexical search, the other half of hybrid retrieval

    _LEXICAL_SQL = (
        "SELECT id, metadata, "
        "ts_rank_cd(to_tsvector('simple', coalesce(metadata ->> 'text', '')), "
        "plainto_tsquery('simple', :query)) AS score "
        "FROM {table} WHERE to_tsvector('simple', "
        "coalesce(metadata ->> 'text', '')) @@ plainto_tsquery('simple', :query)"
        "{filters} ORDER BY score DESC LIMIT :limit"
    )

    def _lexical_sql(
        self, collection: str, filters: dict[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        where, params = self._where(filters)
        return (
            self._LEXICAL_SQL.format(
                table=self._table(collection),
                filters=f" AND {where}" if where else "",
            ),
            params,
        )

    def lexical_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Full-text search over the stored passage, for fusing with vectors.

        Uses the `simple` text configuration: no stemming, so it behaves the
        same whatever language the passage is in.
        """
        sql, params = self._lexical_sql(collection, filters)
        params |= {"query": query, "limit": limit}
        with self._sql.transaction() as session:
            rows = session.execute(text(sql), params).mappings().all()
        return [self._to_result(row) for row in rows]

    async def alexical_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Full-text search over the stored passage, for fusing with vectors."""
        sql, params = self._lexical_sql(collection, filters)
        params |= {"query": query, "limit": limit}
        async with self._sql.atransaction() as session:
            rows = (await session.execute(text(sql), params)).mappings().all()
        return [self._to_result(row) for row in rows]

    # Health and lifecycle

    def health_check(self) -> bool:
        with self._sql.transaction() as session:
            return session.execute(text("SELECT 1")).scalar() == 1

    async def ahealth_check(self) -> bool:
        async with self._sql.atransaction() as session:
            return (await session.execute(text("SELECT 1"))).scalar() == 1

    def get_service_name(self) -> str:
        return "pgvector"

    def dispose(self) -> None:
        self._sql.dispose()

    async def adispose(self) -> None:
        await self._sql.adispose()
