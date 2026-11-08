"""Both halves of retrieval on a real Postgres: the vector and the text.

pgvector is the only backend here besides Azure AI Search that answers a
full-text question, so this is where the `simple` text configuration, the
ranking and the filters that narrow both halves are pinned. The service and
collection fixtures are the sibling module's, re-exposed here because the
directory's conftest belongs to Qdrant.
"""

from typing import Any

import pytest

from base.models.definitions.vector_search import DistanceMetric
from base.models.view_models.exceptions.vector_search import VectorSearchError
from base.models.view_models.vector_search import VectorRecord
from base.services.vector_search.pgvector import PgVectorSearchService

from . import test_pgvector

pgvector = test_pgvector.pgvector
collection = test_pgvector.collection
record = test_pgvector.record

pytestmark = pytest.mark.integration

NORTH: list[float] = [1.0, 0.0, 0.0]
EAST: list[float] = [0.0, 1.0, 0.0]


@pytest.fixture
async def populated(pgvector: PgVectorSearchService, collection: str) -> str:
    """Two records a metre apart, each with a passage and a tenant."""
    await pgvector.aupsert(
        collection,
        [
            VectorRecord(
                id="north",
                vector=NORTH,
                metadata={"text": "a quiet robot vacuum", "tenant": "a", "year": 2026},
            ),
            VectorRecord(
                id="east",
                vector=EAST,
                metadata={"text": "a loud kettle", "tenant": "b", "year": 2025},
            ),
        ],
    )
    return collection


@pytest.fixture
async def dot_collection(pgvector: PgVectorSearchService, unique: str) -> Any:
    """A collection indexed for inner product, dropped when the test ends."""
    name = f"pgv_dot_{unique}"
    await pgvector.acreate_collection(name, vector_size=3, distance=DistanceMetric.DOT)
    yield name
    await pgvector.adelete_collection(name)


@pytest.fixture
async def euclidean_collection(pgvector: PgVectorSearchService, unique: str) -> Any:
    """A collection indexed for L2 distance, dropped when the test ends."""
    name = f"pgv_l2_{unique}"
    await pgvector.acreate_collection(
        name, vector_size=3, distance=DistanceMetric.EUCLIDEAN
    )
    yield name
    await pgvector.adelete_collection(name)


# Vector search


async def test_the_nearest_vector_comes_first(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert [r.id for r in await pgvector.asearch(populated, NORTH)] == ["north", "east"]


async def test_an_identical_vector_scores_one_under_cosine(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert (await pgvector.asearch(populated, NORTH))[0].score == pytest.approx(1.0)


async def test_an_orthogonal_vector_scores_zero_under_cosine(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert (await pgvector.asearch(populated, NORTH))[1].score == pytest.approx(0.0)


async def test_a_result_carries_the_metadata_that_was_stored(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    result = (await pgvector.asearch(populated, NORTH))[0]
    assert result.metadata["tenant"] == "a"


async def test_a_limit_caps_what_comes_back(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert len(await pgvector.asearch(populated, NORTH, limit=1)) == 1


async def test_a_search_of_an_empty_collection_returns_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    assert await pgvector.asearch(collection, NORTH) == []


async def test_searching_a_collection_that_is_not_there_fails(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.asearch(f"pgv_absent_{unique}", NORTH)


async def test_searching_with_a_vector_of_the_wrong_width_fails(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.asearch(populated, [1.0, 0.0])


def test_the_sync_path_searches_the_same_way(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert [r.id for r in pgvector.search(populated, NORTH)] == ["north", "east"]


# Filters


async def test_a_filter_narrows_the_search(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.asearch(populated, NORTH, filters={"tenant": "b"})
    assert [r.id for r in found] == ["east"]


async def test_a_list_filter_matches_any_of_its_values(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.asearch(populated, NORTH, filters={"tenant": ["a", "b"]})
    assert {r.id for r in found} == {"north", "east"}


async def test_an_empty_list_filter_matches_nothing(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert await pgvector.asearch(populated, NORTH, filters={"tenant": []}) == []


async def test_a_number_filter_matches_what_was_stored(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.asearch(populated, NORTH, filters={"year": 2026})
    assert [r.id for r in found] == ["north"]


async def test_several_filters_are_all_required(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.asearch(
        populated, NORTH, filters={"tenant": "a", "year": 2025}
    )
    assert found == []


async def test_a_boolean_filter_matches_what_was_stored(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    """JSON writes its booleans lower case, which `str(True)` does not."""
    await pgvector.aupsert(
        collection, [record("yes", archived=True), record("no", archived=False)]
    )
    found = await pgvector.asearch(collection, NORTH, filters={"archived": True})
    assert [r.id for r in found] == ["yes"]


async def test_a_boolean_filter_excludes_the_other_value(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(
        collection, [record("yes", archived=True), record("no", archived=False)]
    )
    found = await pgvector.asearch(collection, NORTH, filters={"archived": False})
    assert [r.id for r in found] == ["no"]


async def test_a_null_filter_matches_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    """`->>` reports a JSON null as SQL NULL, which equals nothing at all."""
    await pgvector.aupsert(collection, [record("a", note=None)])
    assert await pgvector.asearch(collection, NORTH, filters={"note": None}) == []


async def test_a_filter_on_a_field_no_record_has_matches_nothing(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert await pgvector.asearch(populated, NORTH, filters={"absent": "x"}) == []


@pytest.mark.parametrize(
    "field", ["tenant' OR '1'='1", 'tenant"; DROP TABLE x; --', "tenant) --"]
)
async def test_a_filter_field_carrying_sql_matches_nothing_and_breaks_nothing(
    pgvector: PgVectorSearchService, populated: str, field: str
) -> None:
    assert await pgvector.asearch(populated, NORTH, filters={field: "a"}) == []
    assert await pgvector.acount(populated) == 2


async def test_a_filter_value_carrying_sql_matches_nothing_and_breaks_nothing(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    value = "a'; DROP TABLE students; --"
    assert await pgvector.asearch(populated, NORTH, filters={"tenant": value}) == []
    assert await pgvector.acount(populated) == 2


# Score thresholds


async def test_a_score_threshold_drops_what_is_further_away(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.asearch(populated, NORTH, score_threshold=0.5)
    assert [r.id for r in found] == ["north"]


async def test_a_threshold_above_every_score_returns_nothing(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert await pgvector.asearch(populated, NORTH, score_threshold=1.5) == []


async def test_a_threshold_of_zero_still_narrows(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    """Opposite vectors score below zero under cosine, so `0.0` is not `None`."""
    await pgvector.aupsert(
        collection, [record("north", NORTH), record("south", [-1.0, 0.0, 0.0])]
    )
    found = await pgvector.asearch(collection, NORTH, score_threshold=0.0)
    assert [r.id for r in found] == ["north"]


async def test_a_threshold_and_a_filter_both_apply(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.asearch(
        populated, NORTH, filters={"tenant": "b"}, score_threshold=0.5
    )
    assert found == []


# Metrics


async def test_a_dot_product_collection_scores_the_inner_product(
    pgvector: PgVectorSearchService, dot_collection: str
) -> None:
    await pgvector.aupsert(dot_collection, [record("a", [1.0, 2.0, 3.0])])
    found = await pgvector.asearch(dot_collection, [1.0, 2.0, 3.0])
    assert found[0].score == pytest.approx(14.0)


async def test_a_dot_product_collection_prefers_the_longer_vector(
    pgvector: PgVectorSearchService, dot_collection: str
) -> None:
    await pgvector.aupsert(
        dot_collection, [record("short", NORTH), record("long", [5.0, 0.0, 0.0])]
    )
    found = await pgvector.asearch(dot_collection, NORTH)
    assert [r.id for r in found] == ["long", "short"]


async def test_a_euclidean_collection_scores_an_identical_vector_as_zero(
    pgvector: PgVectorSearchService, euclidean_collection: str
) -> None:
    await pgvector.aupsert(euclidean_collection, [record("a", [1.0, 2.0, 3.0])])
    found = await pgvector.asearch(euclidean_collection, [1.0, 2.0, 3.0])
    assert found[0].score == pytest.approx(0.0)


async def test_a_euclidean_collection_scores_a_distant_vector_lower(
    pgvector: PgVectorSearchService, euclidean_collection: str
) -> None:
    await pgvector.aupsert(
        euclidean_collection, [record("near", NORTH), record("far", [9.0, 0.0, 0.0])]
    )
    found = await pgvector.asearch(euclidean_collection, NORTH)
    assert [r.id for r in found] == ["near", "far"]


async def test_the_metric_is_recovered_from_the_index_not_assumed(
    pgvector: PgVectorSearchService, dot_collection: str
) -> None:
    """A process that did not create the collection must still query it right."""
    await pgvector.aupsert(dot_collection, [record("a", [1.0, 2.0, 3.0])])
    pgvector._metrics.clear()
    found = await pgvector.asearch(dot_collection, [1.0, 2.0, 3.0])
    assert found[0].score == pytest.approx(14.0)


def test_the_sync_path_recovers_the_metric_too(
    pgvector: PgVectorSearchService, dot_collection: str
) -> None:
    pgvector.upsert(dot_collection, [record("a", [1.0, 2.0, 3.0])])
    pgvector._metrics.clear()
    assert pgvector.search(dot_collection, [1.0, 2.0, 3.0])[0].score == pytest.approx(
        14.0
    )


# Lexical search


async def test_a_passage_is_found_by_a_word_in_it(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = await pgvector.alexical_search(populated, "kettle")
    assert [r.id for r in found] == ["east"]


async def test_every_word_in_the_question_must_appear(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert await pgvector.alexical_search(populated, "quiet kettle") == []


async def test_a_word_that_is_in_no_passage_finds_nothing(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert await pgvector.alexical_search(populated, "helicopter") == []


async def test_matching_ignores_case(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert [r.id for r in await pgvector.alexical_search(populated, "KETTLE")] == [
        "east"
    ]


async def test_a_match_is_scored_above_zero(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert (await pgvector.alexical_search(populated, "kettle"))[0].score > 0


async def test_the_denser_match_ranks_first(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(
        collection,
        [
            record("dense", text="robot robot robot"),
            record("sparse", text="robot and a long tail of other unrelated words"),
        ],
    )
    found = await pgvector.alexical_search(collection, "robot")
    assert [r.id for r in found] == ["dense", "sparse"]


async def test_lexical_search_does_not_stem(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    """The `simple` configuration, so a passage reads the same in any language."""
    await pgvector.aupsert(collection, [record("a", text="vacuums")])
    assert await pgvector.alexical_search(collection, "vacuum") == []
    assert len(await pgvector.alexical_search(collection, "vacuums")) == 1


async def test_a_stop_word_is_still_a_word(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    """`simple` strips nothing, so `the` matches rather than disappearing."""
    await pgvector.aupsert(collection, [record("a", text="the kettle")])
    assert len(await pgvector.alexical_search(collection, "the")) == 1


async def test_a_record_with_no_text_is_never_matched(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("a", tenant="a")])
    assert await pgvector.alexical_search(collection, "anything") == []


async def test_a_limit_caps_what_lexical_search_returns(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(
        collection, [record(f"doc-{i}", text="robot") for i in range(3)]
    )
    assert len(await pgvector.alexical_search(collection, "robot", limit=2)) == 2


@pytest.mark.parametrize("query", ["", "   ", "!!! ???", "&|!<>"])
async def test_a_question_with_no_words_in_it_matches_nothing(
    pgvector: PgVectorSearchService, populated: str, query: str
) -> None:
    assert await pgvector.alexical_search(populated, query) == []


@pytest.mark.parametrize(
    "query", ["kettle'; DROP TABLE students; --", "kettle & robot", "kettle:*"]
)
async def test_a_question_carrying_sql_or_tsquery_syntax_is_only_text(
    pgvector: PgVectorSearchService, populated: str, query: str
) -> None:
    """`plainto_tsquery` takes the words and ignores the punctuation."""
    await pgvector.alexical_search(populated, query)
    assert await pgvector.acount(populated) == 2


async def test_lexical_search_narrows_with_a_filter(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(
        collection,
        [
            record("mine", text="robot vacuum", tenant="a"),
            record("theirs", text="robot vacuum", tenant="b"),
        ],
    )
    found = await pgvector.alexical_search(collection, "robot", filters={"tenant": "a"})
    assert [r.id for r in found] == ["mine"]


async def test_lexical_search_takes_a_boolean_filter_too(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(
        collection,
        [
            record("live", text="robot", archived=False),
            record("gone", text="robot", archived=True),
        ],
    )
    found = await pgvector.alexical_search(
        collection, "robot", filters={"archived": False}
    )
    assert [r.id for r in found] == ["live"]


async def test_lexical_search_on_a_collection_that_is_not_there_fails(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.alexical_search(f"pgv_absent_{unique}", "robot")


async def test_both_halves_of_retrieval_find_the_same_record(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    dense = await pgvector.asearch(populated, NORTH, limit=1)
    lexical = await pgvector.alexical_search(populated, "robot vacuum", limit=1)
    assert dense[0].id == lexical[0].id == "north"


def test_the_sync_path_searches_text_the_same_way(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    assert [r.id for r in pgvector.lexical_search(populated, "kettle")] == ["east"]


def test_the_sync_path_filters_text_the_same_way(
    pgvector: PgVectorSearchService, populated: str
) -> None:
    found = pgvector.lexical_search(populated, "robot", filters={"tenant": "b"})
    assert found == []
