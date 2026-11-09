"""What the pgvector adapter sends, and how often, with no server behind it.

The shaping tests pin the SQL; these pin the calls — that a large upsert is
split, that a collection's metric is looked up once and then remembered, that
a failed statement is repeated and arrives as the vector search family rather
than SQLAlchemy's. A recorder stands in for `SQLService`, so both the sync
and the async path are walked without a container.
"""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from base.models.definitions.vector_search import (
    VECTOR_UPSERT_BATCH_SIZE,
    DistanceMetric,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.models.view_models.exceptions.vector_search import (
    VectorSearchBadRequestError,
    VectorSearchError,
    VectorSearchTimeoutError,
    VectorSearchUnexpectedError,
)
from base.models.view_models.vector_search import VectorRecord
from base.services.configuration.mixins import SQLSettingsMixin
from base.services.error_handling.sql import SQLErrorHandler
from base.services.vector_search.pgvector import PgVectorSearchService
from tests.conftest import MakeSettings
from tests.fakes import no_sleep


class Row:
    """A row as SQLAlchemy hands one back, read by attribute."""

    def __init__(self, identifier: str, embedding: str | None = None) -> None:
        self.id = identifier
        self.metadata: dict[str, Any] = {}
        self.embedding = embedding


class Result:
    """Only the three shapes this adapter reads off a result."""

    def __init__(self, rows: list[Any], scalar: Any) -> None:
        self._rows = rows
        self._scalar = scalar

    def scalar(self) -> Any:
        return self._scalar

    def all(self) -> list[Any]:
        return self._rows

    def mappings(self) -> "Result":
        return self


class Call:
    """One statement the adapter executed, with what it bound to it."""

    def __init__(self, statement: Any, params: Any) -> None:
        self.sql = str(statement)
        self.params = params


class RecordingSQL:
    """Stands in for `SQLService`, keeping every statement and answering canned rows.

    `transaction` translates a failure into the SQL family and `_get_session`
    does not, exactly as the real service does, because which of the two this
    adapter reaches for decides whether its own handler ever sees the error.
    """

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.rows: list[Any] = []
        self.scalar: Any = None
        self.failures: int = 0
        self.error: Exception = PoolTimeoutError("pool exhausted")
        self.disposed: bool = False
        self.adisposed: bool = False

    def _execute(self, statement: Any, params: Any) -> Result:
        self.calls.append(Call(statement, params))
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return Result(self.rows, self.scalar)

    @contextmanager
    def _get_session(self) -> Generator["Session"]:
        yield Session(self)

    @asynccontextmanager
    async def _aget_session(self) -> AsyncGenerator["AsyncSession"]:
        yield AsyncSession(self)

    @contextmanager
    def transaction(self) -> Generator["Session"]:
        try:
            with self._get_session() as session:
                yield session
        except SQLAlchemyError as exc:
            raise SQLErrorHandler().as_domain_error(exc) from exc

    @asynccontextmanager
    async def atransaction(self) -> AsyncGenerator["AsyncSession"]:
        try:
            async with self._aget_session() as session:
                yield session
        except SQLAlchemyError as exc:
            raise SQLErrorHandler().as_domain_error(exc) from exc

    def dispose(self) -> None:
        self.disposed = True

    async def adispose(self) -> None:
        self.adisposed = True

    def matching(self, fragment: str) -> list[Call]:
        return [call for call in self.calls if fragment in call.sql]

    def only(self, fragment: str) -> Call:
        found = self.matching(fragment)
        assert len(found) == 1, f"{len(found)} statements matched {fragment!r}"
        return found[0]


class Session:
    def __init__(self, recorder: RecordingSQL) -> None:
        self._recorder = recorder

    def execute(self, statement: Any, params: Any = None) -> Result:
        return self._recorder._execute(statement, params)


class AsyncSession(Session):
    async def execute(self, statement: Any, params: Any = None) -> Result:  # type: ignore[override]
        return self._recorder._execute(statement, params)


@pytest.fixture
def service(make_settings: MakeSettings) -> PgVectorSearchService:
    make_settings(SQLSettingsMixin)
    return PgVectorSearchService()


@pytest.fixture
def sql(service: PgVectorSearchService) -> RecordingSQL:
    recorder = RecordingSQL()
    service._sql = recorder  # type: ignore[assignment]
    return recorder


def records(count: int) -> list[VectorRecord]:
    return [VectorRecord(id=str(i), vector=[float(i)]) for i in range(count)]


# Creating and dropping a collection


def test_creating_a_collection_sends_every_statement_in_one_transaction(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.create_collection("chunks", 3)
    assert len(sql.calls) == 5


async def test_the_async_path_creates_a_collection_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.acreate_collection("chunks", 3)
    assert [call.sql for call in sql.calls] == service._create_sql(
        "chunks", 3, DistanceMetric.COSINE
    )


def test_creating_a_collection_remembers_the_metric_it_was_given(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.create_collection("chunks", 3, distance=DistanceMetric.DOT)
    assert service._metrics["chunks"] == DistanceMetric.DOT


async def test_the_async_path_remembers_the_metric_too(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.acreate_collection("chunks", 3, distance=DistanceMetric.EUCLIDEAN)
    assert service._metrics["chunks"] == DistanceMetric.EUCLIDEAN


def test_a_metric_this_backend_does_not_have_is_remembered_as_cosine(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.create_collection("chunks", 3, distance="manhattan")
    assert service._metrics["chunks"] == DistanceMetric.COSINE


def test_backend_options_this_one_has_no_use_for_are_ignored(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.create_collection("chunks", 3, on_disk_payload=True)
    assert len(sql.calls) == 5


def test_dropping_a_collection_tolerates_one_that_is_not_there(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.delete_collection("chunks")
    assert sql.only("DROP TABLE").sql == 'DROP TABLE IF EXISTS "chunks"'


async def test_the_async_path_drops_a_collection_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.adelete_collection("chunks")
    assert sql.only("DROP TABLE").sql == 'DROP TABLE IF EXISTS "chunks"'


def test_dropping_a_collection_forgets_its_metric(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.create_collection("chunks", 3, distance=DistanceMetric.DOT)
    service.delete_collection("chunks")
    assert "chunks" not in service._metrics


async def test_the_async_path_forgets_the_metric_too(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.acreate_collection("chunks", 3, distance=DistanceMetric.DOT)
    await service.adelete_collection("chunks")
    assert "chunks" not in service._metrics


def test_asking_whether_a_collection_exists_binds_its_name(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = True
    assert service.collection_exists("Chunks") is True
    assert sql.calls[0].params == {"name": "Chunks"}


async def test_the_async_path_asks_the_same_question(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = False
    assert await service.acollection_exists("chunks") is False


# Upsert, which batches


def test_an_empty_upsert_sends_no_statement(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.upsert("chunks", [])
    assert sql.calls == []


async def test_an_empty_async_upsert_sends_no_statement(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.aupsert("chunks", [])
    assert sql.calls == []


def test_a_batch_at_the_limit_is_one_statement(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.upsert("chunks", records(VECTOR_UPSERT_BATCH_SIZE))
    assert len(sql.calls) == 1


def test_a_batch_over_the_limit_is_split(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.upsert("chunks", records(VECTOR_UPSERT_BATCH_SIZE + 1))
    assert [len(call.params) for call in sql.calls] == [VECTOR_UPSERT_BATCH_SIZE, 1]


async def test_the_async_path_splits_a_batch_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.aupsert("chunks", records(VECTOR_UPSERT_BATCH_SIZE * 2 + 3))
    assert [len(call.params) for call in sql.calls] == [
        VECTOR_UPSERT_BATCH_SIZE,
        VECTOR_UPSERT_BATCH_SIZE,
        3,
    ]


def test_every_batch_uses_the_same_statement(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.upsert("chunks", records(VECTOR_UPSERT_BATCH_SIZE + 1))
    assert len({call.sql for call in sql.calls}) == 1


def test_a_record_reaches_the_statement_as_text_to_be_cast(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.upsert(
        "chunks", [VectorRecord(id="a", vector=[1.0], metadata={"text": "hi"})]
    )
    assert sql.calls[0].params == [
        {"id": "a", "embedding": "[1.0]", "metadata": '{"text": "hi"}'}
    ]


# Search, and the metric it needs first


def test_a_search_looks_the_metric_up_before_querying(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.search("chunks", [1.0])
    assert sql.calls[0].params == {"name": "chunks"}
    assert "ORDER BY embedding" in sql.calls[1].sql


def test_a_metric_already_known_is_not_looked_up_again(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.create_collection("chunks", 3, distance=DistanceMetric.DOT)
    sql.calls.clear()
    service.search("chunks", [1.0])
    assert len(sql.calls) == 1
    assert "<#>" in sql.calls[0].sql


def test_the_metric_is_looked_up_once_and_then_remembered(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = "vector_l2_ops"
    service.search("chunks", [1.0])
    service.search("chunks", [1.0])
    assert len(sql.matching("pg_index")) == 1
    assert service._metrics["chunks"] == DistanceMetric.EUCLIDEAN


async def test_the_async_path_remembers_the_metric_it_looked_up(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = "vector_ip_ops"
    await service.asearch("chunks", [1.0])
    await service.asearch("chunks", [1.0])
    assert len(sql.matching("pg_index")) == 1
    assert service._metrics["chunks"] == DistanceMetric.DOT


def test_a_collection_with_no_vector_index_is_searched_as_cosine(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = None
    service.search("chunks", [1.0])
    assert "<=>" in sql.calls[1].sql


def test_a_search_binds_the_query_vector_as_pgvectors_text(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.search("chunks", [1.0, 0.5], limit=3)
    assert sql.calls[1].params["query"] == "[1.0,0.5]"
    assert sql.calls[1].params["limit"] == 3


async def test_the_async_path_binds_the_query_vector_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.asearch("chunks", [1.0, 0.5])
    assert sql.calls[1].params["query"] == "[1.0,0.5]"


def test_a_search_turns_each_row_into_a_result(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [{"id": "a", "metadata": {"t": "x"}, "score": 0.9}]
    result = service.search("chunks", [1.0])[0]
    assert (result.id, result.score, result.metadata) == ("a", 0.9, {"t": "x"})


async def test_the_async_path_maps_rows_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [{"id": "a", "metadata": {}, "score": 0.1}]
    assert (await service.asearch("chunks", [1.0]))[0].id == "a"


# Get, delete and count


def test_get_binds_the_ids_as_a_list(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.get("chunks", ["a", "b"])
    assert sql.calls[0].params == {"ids": ["a", "b"]}


async def test_the_async_path_binds_the_ids_as_a_list_too(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.aget("chunks", ["a"])
    assert sql.calls[0].params == {"ids": ["a"]}


def test_get_turns_each_row_into_a_record(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [Row("a", "[1.0,2.0]")]
    assert service.get("chunks", ["a"], with_vectors=True)[0].vector == [1.0, 2.0]


def test_delete_binds_the_ids_as_a_list(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.delete("chunks", ["a", "b"])
    assert sql.only("DELETE FROM").params == {"ids": ["a", "b"]}


async def test_the_async_path_deletes_by_the_same_list(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.adelete("chunks", ["a"])
    assert sql.only("DELETE FROM").params == {"ids": ["a"]}


def test_count_reads_the_number_off_the_result(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = 7
    assert service.count("chunks") == 7


async def test_the_async_path_counts_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = 7
    assert await service.acount("chunks") == 7


@pytest.mark.parametrize("scalar", [None, 0])
def test_a_count_of_nothing_is_zero(
    service: PgVectorSearchService, sql: RecordingSQL, scalar: Any
) -> None:
    sql.scalar = scalar
    assert service.count("chunks") == 0


# Scroll


def test_scroll_binds_its_limit_and_reports_the_next_offset(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [Row("a"), Row("b")]
    page, offset = service.scroll("chunks", limit=2)
    assert [r.id for r in page] == ["a", "b"]
    assert offset == "b"


async def test_the_async_path_scrolls_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [Row("a")]
    page, offset = await service.ascroll("chunks", limit=2)
    assert ([r.id for r in page], offset) == (["a"], None)


def test_scroll_resumes_from_the_offset_it_was_given(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.scroll("chunks", offset="doc-3")
    assert sql.calls[0].params["offset"] == "doc-3"


# Lexical search


def test_lexical_search_binds_the_question_and_the_limit(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.lexical_search("chunks", "robot vacuum", limit=4)
    assert sql.calls[0].params == {"query": "robot vacuum", "limit": 4}


async def test_the_async_path_binds_the_question_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.alexical_search("chunks", "robot vacuum")
    assert sql.calls[0].params["query"] == "robot vacuum"


def test_lexical_search_keeps_its_filters_beside_the_question(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.lexical_search("chunks", "hi", filters={"tenant": "acme"})
    assert sql.calls[0].params == {
        "f0_field": "tenant",
        "f0": "acme",
        "query": "hi",
        "limit": 10,
    }


def test_lexical_search_turns_each_row_into_a_result(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [{"id": "a", "metadata": {}, "score": 0.2}]
    assert service.lexical_search("chunks", "hi")[0].score == 0.2


async def test_the_async_lexical_path_maps_rows_the_same_way(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.rows = [{"id": "a", "metadata": {}, "score": 0.2}]
    assert (await service.alexical_search("chunks", "hi"))[0].id == "a"


# Health and lifecycle


def test_the_health_check_asks_the_cheapest_question_there_is(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = 1
    assert service.health_check() is True
    assert sql.calls[0].sql == "SELECT 1"


async def test_the_async_health_check_asks_the_same_question(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = 1
    assert await service.ahealth_check() is True


def test_a_server_that_answers_something_else_is_not_healthy(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    sql.scalar = 0
    assert service.health_check() is False


def test_disposing_releases_the_pool_it_shares_with_the_rest_of_the_service(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    service.dispose()
    assert sql.disposed is True


async def test_disposing_asynchronously_releases_the_async_pool(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    await service.adispose()
    assert sql.adisposed is True


# Failures


def test_a_statement_the_server_refused_arrives_as_a_vector_search_error(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    """Not as the SQL family: a caller catches one family whichever backend ran."""
    no_sleep(monkeypatch)
    sql.error = ProgrammingError("SELECT 1", {}, Exception("no such table"))
    sql.failures = 99
    with pytest.raises(VectorSearchBadRequestError):
        service.count("chunks")


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.create_collection("chunks", 3),
        lambda s: s.delete_collection("chunks"),
        lambda s: s.collection_exists("chunks"),
        lambda s: s.upsert("chunks", records(1)),
        lambda s: s.search("chunks", [1.0]),
        lambda s: s.get("chunks", ["a"]),
        lambda s: s.delete("chunks", ["a"]),
        lambda s: s.count("chunks"),
        lambda s: s.scroll("chunks"),
        lambda s: s.lexical_search("chunks", "hi"),
    ],
)
def test_no_method_lets_a_database_failure_escape_as_anything_else(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any, call: Any
) -> None:
    no_sleep(monkeypatch)
    sql.error = ProgrammingError("SELECT 1", {}, Exception("no such table"))
    sql.failures = 99
    with pytest.raises(VectorSearchError):
        call(service)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.acreate_collection("chunks", 3),
        lambda s: s.adelete_collection("chunks"),
        lambda s: s.acollection_exists("chunks"),
        lambda s: s.aupsert("chunks", records(1)),
        lambda s: s.asearch("chunks", [1.0]),
        lambda s: s.aget("chunks", ["a"]),
        lambda s: s.adelete("chunks", ["a"]),
        lambda s: s.acount("chunks"),
        lambda s: s.ascroll("chunks"),
        lambda s: s.alexical_search("chunks", "hi"),
    ],
)
async def test_no_async_method_does_either(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any, call: Any
) -> None:
    no_sleep(monkeypatch)
    sql.error = ProgrammingError("SELECT 1", {}, Exception("no such table"))
    sql.failures = 99
    with pytest.raises(VectorSearchError):
        await call(service)


def test_a_refused_statement_is_not_repeated(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    no_sleep(monkeypatch)
    sql.error = ProgrammingError("SELECT 1", {}, Exception("no such table"))
    sql.failures = 99
    with pytest.raises(VectorSearchBadRequestError):
        service.count("chunks")
    assert len(sql.calls) == 1


def test_a_timeout_waiting_for_a_connection_is_tried_again(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    no_sleep(monkeypatch)
    sql.failures = 2
    sql.scalar = 3
    assert service.count("chunks") == 3
    assert len(sql.calls) == 3


async def test_the_async_path_tries_again_too(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    no_sleep(monkeypatch)
    sql.failures = 1
    sql.scalar = 3
    assert await service.acount("chunks") == 3
    assert len(sql.calls) == 2


def test_an_upsert_that_fails_part_way_through_starts_the_whole_batch_again(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    """Which is only safe because the statement is an upsert on the primary key."""
    no_sleep(monkeypatch)
    sql.failures = 1
    service.upsert("chunks", records(VECTOR_UPSERT_BATCH_SIZE + 1))
    assert [len(call.params or []) for call in sql.calls] == [
        VECTOR_UPSERT_BATCH_SIZE,
        VECTOR_UPSERT_BATCH_SIZE,
        1,
    ]


def test_a_statement_the_server_refused_is_not_retried_just_because_it_failed(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    """An `OperationalError` on a live connection is a refusal, not a blip."""
    no_sleep(monkeypatch)
    sql.error = OperationalError("SELECT 1", {}, Exception("syntax error"))
    sql.failures = 99
    with pytest.raises(VectorSearchBadRequestError):
        service.count("chunks")
    assert len(sql.calls) == 1


def test_a_failure_that_never_clears_runs_out_of_attempts(
    service: PgVectorSearchService, sql: RecordingSQL, monkeypatch: Any
) -> None:
    no_sleep(monkeypatch)
    sql.failures = 99
    with pytest.raises(VectorSearchTimeoutError):
        service.count("chunks")
    assert len(sql.calls) == 4


def test_the_health_check_is_not_repeated_and_reports_a_domain_error(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    """It answers for the moment, so a retry would hide a dependency being down."""
    sql.failures = 99
    with pytest.raises(ProviderError):
        service.health_check()
    assert len(sql.calls) == 1


def test_a_collection_name_that_is_not_an_identifier_never_reaches_the_server(
    service: PgVectorSearchService, sql: RecordingSQL
) -> None:
    """The check is a `ValueError`, which the retry boundary reports as its own."""
    with pytest.raises(VectorSearchUnexpectedError, match="invalid collection name"):
        service.upsert("my chunks", records(1))
    assert sql.calls == []


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.create_collection("my chunks", 3),
        lambda s: s.delete_collection("my chunks"),
        lambda s: s.upsert("my chunks", []),
        lambda s: s.search("my chunks", [1.0]),
        lambda s: s.get("my chunks", ["a"]),
        lambda s: s.delete("my chunks", ["a"]),
        lambda s: s.count("my chunks"),
        lambda s: s.scroll("my chunks"),
        lambda s: s.lexical_search("my chunks", "hi"),
    ],
)
def test_no_method_will_touch_a_collection_name_it_cannot_validate(
    service: PgVectorSearchService, sql: RecordingSQL, call: Any
) -> None:
    with pytest.raises(VectorSearchUnexpectedError, match="invalid collection name"):
        call(service)
