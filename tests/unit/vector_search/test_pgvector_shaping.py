"""The SQL the pgvector adapter writes, before a server sees any of it.

A table name cannot be bound as a parameter, so this backend validates one
itself and quotes it; everything else — the query vector, the filter fields,
their values, the limit — is bound. Which of those is true of each statement
is the whole safety argument, and it is answerable without a database, as are
the row mapping on the way back and the keyset paging.
"""

from typing import Any

import pytest

from base.models.definitions.vector_search import DistanceMetric
from base.models.view_models.vector_search import VectorRecord
from base.services.configuration.mixins import SQLSettingsMixin
from base.services.vector_search.pgvector import PgVectorSearchService

INJECTIONS: list[str] = [
    'chunks"; DROP TABLE users; --',
    "chunks; DROP TABLE users",
    "chunks' OR '1'='1",
    "chunks--",
    "chunks/*x*/",
    "pg_catalog.pg_class",
]


class Row:
    """A row as SQLAlchemy hands one back, read by attribute."""

    def __init__(
        self, identifier: str, metadata: Any = None, embedding: str | None = None
    ) -> None:
        self.id = identifier
        self.metadata = {} if metadata is None else metadata
        self.embedding = embedding


@pytest.fixture
def service(make_settings: Any) -> PgVectorSearchService:
    """The adapter with no engine behind it: nothing here opens a connection."""
    make_settings(SQLSettingsMixin)
    return PgVectorSearchService()


# Table names, which cannot be bound


def test_a_collection_name_becomes_a_quoted_identifier(
    service: PgVectorSearchService,
) -> None:
    assert service._table("chunks") == '"chunks"'


def test_a_name_keeps_its_case_inside_the_quotes(
    service: PgVectorSearchService,
) -> None:
    assert service._table("Chunks") == '"Chunks"'


@pytest.mark.parametrize("name", ["_chunks", "c1", "a_b_9", "A", "_"])
def test_letters_digits_and_underscores_are_accepted(
    service: PgVectorSearchService, name: str
) -> None:
    assert service._table(name) == f'"{name}"'


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "1chunks",
        "9",
        "my chunks",
        "my-chunks",
        "my.chunks",
        "chunks\n",
        "naïve",
        '"chunks"',
        "chunks$",
    ],
)
def test_a_name_that_is_not_an_identifier_is_refused(
    service: PgVectorSearchService, name: str
) -> None:
    with pytest.raises(ValueError):
        service._table(name)


@pytest.mark.parametrize("name", INJECTIONS)
def test_a_name_carrying_sql_never_reaches_a_statement(
    service: PgVectorSearchService, name: str
) -> None:
    with pytest.raises(ValueError):
        service._table(name)


def test_the_refusal_says_which_name_it_refused(
    service: PgVectorSearchService,
) -> None:
    with pytest.raises(ValueError, match="my chunks"):
        service._table("my chunks")


@pytest.mark.parametrize("name", INJECTIONS)
def test_no_statement_builder_accepts_a_name_carrying_sql(
    service: PgVectorSearchService, name: str
) -> None:
    builders = [
        lambda: service._create_sql(name, 3, "cosine"),
        lambda: service._upsert_sql(name),
        lambda: service._search_sql(name, "cosine", 10, None, None),
        lambda: service._get_sql(name, False),
        lambda: service._scroll_sql(name, 10, None, None, False),
        lambda: service._lexical_sql(name, None),
    ]
    for build in builders:
        with pytest.raises(ValueError):
            build()


# Vector literals


def test_a_vector_becomes_pgvectors_bracketed_text(
    service: PgVectorSearchService,
) -> None:
    assert service._literal([1.0, 0.5, -2.0]) == "[1.0,0.5,-2.0]"


def test_an_integer_component_is_written_as_a_float(
    service: PgVectorSearchService,
) -> None:
    assert service._literal([1, 0]) == "[1.0,0.0]"


def test_an_empty_vector_is_still_bracketed(service: PgVectorSearchService) -> None:
    assert service._literal([]) == "[]"


def test_a_component_keeps_the_precision_it_came_with(
    service: PgVectorSearchService,
) -> None:
    assert service._literal([0.1 + 0.2]) == "[0.30000000000000004]"


# Distance metrics


@pytest.mark.parametrize(
    ("metric", "operator", "opclass"),
    [
        (DistanceMetric.COSINE, "<=>", "vector_cosine_ops"),
        (DistanceMetric.DOT, "<#>", "vector_ip_ops"),
        (DistanceMetric.EUCLIDEAN, "<->", "vector_l2_ops"),
    ],
)
def test_each_metric_has_its_operator_and_index_opclass(
    service: PgVectorSearchService, metric: str, operator: str, opclass: str
) -> None:
    assert service._metric(metric)[:2] == (operator, opclass)


@pytest.mark.parametrize("metric", ["manhattan", "", "COSINE", "l2"])
def test_a_metric_this_backend_does_not_have_falls_back_to_cosine(
    service: PgVectorSearchService, metric: str
) -> None:
    assert service._metric(metric) == service._metric(DistanceMetric.COSINE)
    assert service._metric_name(metric) == DistanceMetric.COSINE


def test_a_known_metric_keeps_its_name(service: PgVectorSearchService) -> None:
    assert service._metric_name(DistanceMetric.DOT) == DistanceMetric.DOT


def test_cosine_distance_is_turned_into_a_score_where_higher_is_closer(
    service: PgVectorSearchService,
) -> None:
    assert service._metric(DistanceMetric.COSINE)[2].format(distance="d") == "1 - (d)"


@pytest.mark.parametrize("metric", [DistanceMetric.DOT, DistanceMetric.EUCLIDEAN])
def test_the_other_metrics_score_by_negating_the_distance(
    service: PgVectorSearchService, metric: str
) -> None:
    assert service._metric(metric)[2].format(distance="d") == "-(d)"


@pytest.mark.parametrize(
    ("opclass", "metric"),
    [
        ("vector_cosine_ops", DistanceMetric.COSINE),
        ("vector_ip_ops", DistanceMetric.DOT),
        ("vector_l2_ops", DistanceMetric.EUCLIDEAN),
    ],
)
def test_the_opclass_an_index_carries_names_the_metric_it_was_built_with(
    service: PgVectorSearchService, opclass: str, metric: str
) -> None:
    assert service._resolve_metric("chunks", opclass) == metric


@pytest.mark.parametrize("opclass", [None, "", "bm25_ops"])
def test_an_index_this_backend_cannot_read_is_treated_as_cosine(
    service: PgVectorSearchService, opclass: str | None
) -> None:
    assert service._resolve_metric("chunks", opclass) == DistanceMetric.COSINE


def test_resolving_a_metric_remembers_it_for_the_collection(
    service: PgVectorSearchService,
) -> None:
    service._resolve_metric("chunks", "vector_ip_ops")
    assert service._metrics["chunks"] == DistanceMetric.DOT


# Filters, which become comparisons on the metadata column


def test_no_filters_build_no_clause(service: PgVectorSearchService) -> None:
    assert service._where(None) == ("", {})
    assert service._where({}) == ("", {})


def test_a_scalar_filter_compares_the_metadata_as_text(
    service: PgVectorSearchService,
) -> None:
    clause, params = service._where({"tenant": "acme"})
    assert clause == "metadata ->> :f0_field = :f0"
    assert params == {"f0_field": "tenant", "f0": "acme"}


def test_the_field_name_is_bound_rather_than_written_into_the_clause(
    service: PgVectorSearchService,
) -> None:
    clause, params = service._where({"a' OR '1'='1": "x"})
    assert "OR" not in clause
    assert params["f0_field"] == "a' OR '1'='1"


@pytest.mark.parametrize("values", [["a", "b"], ("a", "b")])
def test_a_sequence_filter_becomes_an_any_of_comparison(
    service: PgVectorSearchService, values: Any
) -> None:
    clause, params = service._where({"tag": values})
    assert clause == "metadata ->> :f0_field = ANY(:f0)"
    assert params["f0"] == ["a", "b"]


def test_a_set_filter_is_an_any_of_comparison_too(
    service: PgVectorSearchService,
) -> None:
    clause, params = service._where({"tag": {"a"}})
    assert clause == "metadata ->> :f0_field = ANY(:f0)"
    assert params["f0"] == ["a"]


def test_an_empty_sequence_still_builds_a_clause_that_matches_nothing(
    service: PgVectorSearchService,
) -> None:
    clause, params = service._where({"tag": []})
    assert clause == "metadata ->> :f0_field = ANY(:f0)"
    assert params["f0"] == []


def test_several_filters_are_all_required(service: PgVectorSearchService) -> None:
    clause, params = service._where({"tenant": "acme", "year": 2026})
    assert clause == ("metadata ->> :f0_field = :f0 AND metadata ->> :f1_field = :f1")
    assert params == {
        "f0_field": "tenant",
        "f0": "acme",
        "f1_field": "year",
        "f1": "2026",
    }


@pytest.mark.parametrize(
    ("value", "bound"),
    [
        (2026, "2026"),
        (1.5, "1.5"),
        ("acme", "acme"),
        (True, "true"),
        (False, "false"),
        (None, "None"),
    ],
)
def test_a_value_is_bound_as_the_text_the_metadata_renders(
    service: PgVectorSearchService, value: Any, bound: str
) -> None:
    """`->>` returns JSON's own text, and JSON writes its booleans lower case."""
    assert service._where({"k": value})[1]["f0"] == bound


def test_a_boolean_inside_a_sequence_is_rendered_the_same_way(
    service: PgVectorSearchService,
) -> None:
    assert service._where({"k": [True, False]})[1]["f0"] == ["true", "false"]


# Creating a collection


def test_creating_a_collection_installs_the_extension_first(
    service: PgVectorSearchService,
) -> None:
    assert service._create_sql("chunks", 3, "cosine")[0] == (
        "CREATE EXTENSION IF NOT EXISTS vector"
    )


def test_the_table_carries_an_id_an_embedding_and_metadata(
    service: PgVectorSearchService,
) -> None:
    table = service._create_sql("chunks", 1536, "cosine")[1]
    assert "id TEXT PRIMARY KEY" in table
    assert "embedding vector(1536) NOT NULL" in table
    assert "metadata JSONB NOT NULL DEFAULT '{}'::jsonb" in table


def test_a_vector_size_that_arrives_as_a_float_is_written_whole(
    service: PgVectorSearchService,
) -> None:
    assert "vector(8)" in service._create_sql("chunks", 8.9, "cosine")[1]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("metric", "opclass"),
    [
        (DistanceMetric.COSINE, "vector_cosine_ops"),
        (DistanceMetric.DOT, "vector_ip_ops"),
        (DistanceMetric.EUCLIDEAN, "vector_l2_ops"),
        ("manhattan", "vector_cosine_ops"),
    ],
)
def test_the_vector_index_is_built_for_the_metric_it_will_be_queried_with(
    service: PgVectorSearchService, metric: str, opclass: str
) -> None:
    index = service._create_sql("chunks", 3, metric)[2]
    assert f"USING hnsw (embedding {opclass})" in index


def test_the_metadata_is_indexed_so_a_filter_does_not_scan(
    service: PgVectorSearchService,
) -> None:
    assert "USING gin (metadata)" in service._create_sql("chunks", 3, "cosine")[3]


def test_the_passage_is_indexed_for_full_text_search(
    service: PgVectorSearchService,
) -> None:
    assert (
        "USING gin (to_tsvector('simple', coalesce(metadata ->> 'text', '')))"
        in (service._create_sql("chunks", 3, "cosine")[4])
    )


def test_every_index_is_named_after_the_collection(
    service: PgVectorSearchService,
) -> None:
    statements = service._create_sql("chunks", 3, "cosine")
    assert [s.split('"')[1] for s in statements[2:]] == [
        "chunks_embedding_idx",
        "chunks_metadata_idx",
        "chunks_text_idx",
    ]


def test_creating_a_collection_twice_is_safe(service: PgVectorSearchService) -> None:
    """Every statement is guarded, which is what makes the method retryable."""
    assert all(
        "IF NOT EXISTS" in statement
        for statement in service._create_sql("chunks", 3, "cosine")
    )


# Upsert


def test_an_upsert_replaces_the_row_that_already_holds_the_id(
    service: PgVectorSearchService,
) -> None:
    sql = service._upsert_sql("chunks")
    assert "ON CONFLICT (id) DO UPDATE SET" in sql
    assert "embedding = EXCLUDED.embedding" in sql
    assert "metadata = EXCLUDED.metadata" in sql


def test_the_vector_and_the_metadata_are_cast_from_bound_text(
    service: PgVectorSearchService,
) -> None:
    sql = service._upsert_sql("chunks")
    assert "CAST(:embedding AS vector)" in sql
    assert "CAST(:metadata AS jsonb)" in sql


def test_a_record_becomes_one_parameter_set(service: PgVectorSearchService) -> None:
    record = VectorRecord(id="a", vector=[1.0, 0.0], metadata={"text": "hi"})
    assert service._upsert_params([record]) == [
        {"id": "a", "embedding": "[1.0,0.0]", "metadata": '{"text": "hi"}'}
    ]


def test_a_record_without_metadata_binds_an_empty_object(
    service: PgVectorSearchService,
) -> None:
    record = VectorRecord(id="a", vector=[1.0])
    assert service._upsert_params([record])[0]["metadata"] == "{}"


def test_no_records_bind_no_parameters(service: PgVectorSearchService) -> None:
    assert service._upsert_params([]) == []


def test_metadata_is_bound_as_json_rather_than_written_into_the_statement(
    service: PgVectorSearchService,
) -> None:
    record = VectorRecord(id="a", vector=[1.0], metadata={"q": "o'; DROP TABLE x; --"})
    assert service._upsert_params([record])[0]["metadata"] == (
        '{"q": "o\'; DROP TABLE x; --"}'
    )


# Search


def test_a_search_orders_by_distance_and_takes_a_bound_limit(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._search_sql("chunks", DistanceMetric.COSINE, 10, None, None)
    assert sql.endswith("ORDER BY embedding <=> CAST(:query AS vector) LIMIT :limit")
    assert params == {"limit": 10}


def test_a_search_without_filters_has_no_where_clause(
    service: PgVectorSearchService,
) -> None:
    sql, _ = service._search_sql("chunks", DistanceMetric.COSINE, 10, None, None)
    assert "WHERE" not in sql


def test_a_search_selects_the_score_the_metric_defines(
    service: PgVectorSearchService,
) -> None:
    sql, _ = service._search_sql("chunks", DistanceMetric.COSINE, 10, None, None)
    assert "1 - (embedding <=> CAST(:query AS vector)) AS score" in sql


@pytest.mark.parametrize(
    ("metric", "operator"),
    [
        (DistanceMetric.COSINE, "<=>"),
        (DistanceMetric.DOT, "<#>"),
        (DistanceMetric.EUCLIDEAN, "<->"),
    ],
)
def test_a_search_uses_the_operator_the_collection_was_indexed_for(
    service: PgVectorSearchService, metric: str, operator: str
) -> None:
    sql, _ = service._search_sql("chunks", metric, 10, None, None)
    assert f"ORDER BY embedding {operator} CAST(:query AS vector)" in sql


def test_a_filtered_search_narrows_on_the_metadata(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._search_sql(
        "chunks", DistanceMetric.COSINE, 10, {"tenant": "acme"}, None
    )
    assert "WHERE metadata ->> :f0_field = :f0 ORDER BY" in sql
    assert params["f0"] == "acme"


def test_a_score_threshold_becomes_a_condition_on_the_score(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._search_sql("chunks", DistanceMetric.COSINE, 10, None, 0.75)
    assert "WHERE 1 - (embedding <=> CAST(:query AS vector)) >= :score_threshold" in sql
    assert params["score_threshold"] == 0.75


def test_a_threshold_and_a_filter_are_both_required(
    service: PgVectorSearchService,
) -> None:
    sql, _ = service._search_sql(
        "chunks", DistanceMetric.COSINE, 10, {"tenant": "acme"}, 0.5
    )
    assert "metadata ->> :f0_field = :f0 AND 1 - (" in sql


def test_a_threshold_of_zero_is_still_a_condition(
    service: PgVectorSearchService,
) -> None:
    """`0.0` is falsy, and dropping it would widen the caller's search."""
    _, params = service._search_sql("chunks", DistanceMetric.COSINE, 10, None, 0.0)
    assert params["score_threshold"] == 0.0


def test_the_query_vector_is_bound_rather_than_written_into_the_statement(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._search_sql("chunks", DistanceMetric.COSINE, 10, None, None)
    assert ":query" in sql
    assert "query" not in params


# Get by id


def test_get_selects_only_the_id_and_the_metadata(
    service: PgVectorSearchService,
) -> None:
    assert service._get_sql("chunks", False) == (
        'SELECT id, metadata FROM "chunks" WHERE id = ANY(:ids)'
    )


def test_get_asks_for_the_embedding_as_text_when_vectors_are_wanted(
    service: PgVectorSearchService,
) -> None:
    assert "embedding::text AS embedding" in service._get_sql("chunks", True)


# Scroll


def test_scroll_orders_by_id_so_the_next_page_follows_the_last(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._scroll_sql("chunks", 100, None, None, False)
    assert sql == 'SELECT id, metadata FROM "chunks" ORDER BY id LIMIT :limit'
    assert params == {"limit": 100}


def test_scroll_resumes_after_the_offset_it_was_given(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._scroll_sql("chunks", 10, "doc-3", None, False)
    assert "WHERE id > :offset" in sql
    assert params["offset"] == "doc-3"


def test_scroll_can_both_filter_and_resume(service: PgVectorSearchService) -> None:
    sql, params = service._scroll_sql("chunks", 7, "doc-3", {"t": "a"}, True)
    assert "WHERE metadata ->> :f0_field = :f0 AND id > :offset" in sql
    assert "embedding::text AS embedding" in sql
    assert params == {"f0_field": "t", "f0": "a", "offset": "doc-3", "limit": 7}


def test_scroll_leaves_the_vectors_out_unless_asked(
    service: PgVectorSearchService,
) -> None:
    sql, _ = service._scroll_sql("chunks", 10, None, None, False)
    assert "embedding" not in sql


# Lexical search


def test_lexical_search_ranks_what_matches_the_stored_passage(
    service: PgVectorSearchService,
) -> None:
    sql, _ = service._lexical_sql("chunks", None)
    assert "ts_rank_cd(to_tsvector('simple', coalesce(metadata ->> 'text', ''))" in sql
    assert "@@ plainto_tsquery('simple', :query)" in sql
    assert sql.endswith("ORDER BY score DESC LIMIT :limit")


def test_lexical_search_does_not_stem(service: PgVectorSearchService) -> None:
    """The `simple` configuration, so a passage reads the same in any language."""
    sql, _ = service._lexical_sql("chunks", None)
    assert "'english'" not in sql
    assert sql.count("'simple'") == 4


def test_lexical_search_without_filters_asks_only_for_the_match(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._lexical_sql("chunks", None)
    assert " AND " not in sql
    assert params == {}


def test_lexical_search_adds_its_filters_to_the_match(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._lexical_sql("chunks", {"tenant": "acme"})
    assert ":query) AND metadata ->> :f0_field = :f0 ORDER BY" in sql
    assert params == {"f0_field": "tenant", "f0": "acme"}


def test_the_query_text_is_bound_rather_than_written_into_the_statement(
    service: PgVectorSearchService,
) -> None:
    sql, params = service._lexical_sql("chunks", None)
    assert ":query" in sql
    assert params == {}


# Rows on the way back


def test_a_row_becomes_a_result_carrying_its_score_as_a_float(
    service: PgVectorSearchService,
) -> None:
    result = service._to_result({"id": "a", "metadata": {"t": "x"}, "score": 1})
    assert (result.id, result.score, result.metadata) == ("a", 1.0, {"t": "x"})


@pytest.mark.parametrize("metadata", ['{"t": "x"}', b'{"t": "x"}'.decode()])
def test_metadata_that_arrives_as_json_text_is_parsed(
    service: PgVectorSearchService, metadata: str
) -> None:
    result = service._to_result({"id": "a", "metadata": metadata, "score": 0.5})
    assert result.metadata == {"t": "x"}


@pytest.mark.parametrize("metadata", [None, "", {}])
def test_metadata_that_is_missing_becomes_an_empty_mapping(
    service: PgVectorSearchService, metadata: Any
) -> None:
    result = service._to_result({"id": "a", "metadata": metadata, "score": 0.5})
    assert result.metadata == {}


def test_a_result_carries_no_vector(service: PgVectorSearchService) -> None:
    """A search reads the score, not the embedding; `get` is what returns one."""
    assert service._to_result({"id": "a", "metadata": {}, "score": 0.5}).vector is None


def test_a_row_becomes_a_record_without_a_vector_by_default(
    service: PgVectorSearchService,
) -> None:
    record = service._to_record(Row("a", {"t": "x"}), with_vectors=False)
    assert (record.id, record.vector, record.metadata) == ("a", [], {"t": "x"})


def test_a_record_parses_the_embedding_text_when_vectors_were_asked_for(
    service: PgVectorSearchService,
) -> None:
    record = service._to_record(Row("a", {}, "[1.0,0.5]"), with_vectors=True)
    assert record.vector == [1.0, 0.5]


def test_a_record_parses_metadata_that_arrives_as_json_text(
    service: PgVectorSearchService,
) -> None:
    assert service._to_record(Row("a", '{"t": "x"}'), with_vectors=False).metadata == {
        "t": "x"
    }


# Keyset paging


def test_a_full_page_reports_its_last_id_as_the_next_offset(
    service: PgVectorSearchService,
) -> None:
    records, offset = service._page([Row("a"), Row("b")], limit=2, with_vectors=False)
    assert [r.id for r in records] == ["a", "b"]
    assert offset == "b"


def test_a_short_page_is_the_last_one(service: PgVectorSearchService) -> None:
    _, offset = service._page([Row("a")], limit=2, with_vectors=False)
    assert offset is None


def test_an_empty_page_is_the_last_one(service: PgVectorSearchService) -> None:
    records, offset = service._page([], limit=2, with_vectors=False)
    assert (records, offset) == ([], None)


def test_a_page_of_nothing_because_nothing_was_asked_for_has_no_next_offset(
    service: PgVectorSearchService,
) -> None:
    """A zero limit returns no rows, which is not the same as a full page."""
    assert service._page([], limit=0, with_vectors=False) == ([], None)


# Identity


def test_the_service_names_itself_pgvector(service: PgVectorSearchService) -> None:
    assert service.get_service_name() == "pgvector"


def test_the_existence_check_binds_the_name_and_quotes_it_as_an_identifier(
    service: PgVectorSearchService,
) -> None:
    """`to_regclass` parses, so an unquoted name would fold to lower case."""
    assert service._EXISTS == (
        "SELECT to_regclass(quote_ident(:name)) IS NOT NULL AS present"
    )


def test_the_metric_lookup_binds_the_relation_name(
    service: PgVectorSearchService,
) -> None:
    assert "rel.relname = :name" in service._METRIC_SQL
