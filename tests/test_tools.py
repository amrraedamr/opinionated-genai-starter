"""The small helpers in base.tools, which everything else leans on."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from base.models.view_models.database import PaginationParams
from base.tools import pagination, resources
from base.tools.misc import parse_json_pairs, parse_uuid


class TestParseUuid:
    def test_it_accepts_a_uuid_string(self):
        value = uuid4()
        assert parse_uuid(str(value)) == value

    def test_it_passes_a_uuid_through(self):
        value = uuid4()
        assert parse_uuid(value) is value

    @pytest.mark.parametrize("value", ["", "not-a-uuid", None, 12345])
    def test_it_returns_none_for_anything_else(self, value):
        assert parse_uuid(value) is None


class TestParseJsonPairs:
    def test_it_decodes_json_values(self):
        assert parse_json_pairs(("a",), ['{"x": 1}']) == {"a": {"x": 1}}

    def test_it_keeps_a_value_that_is_not_json(self):
        assert parse_json_pairs(("a",), ["plain"]) == {"a": "plain"}

    def test_it_preserves_missing_values_as_none(self):
        assert parse_json_pairs(("a", "b"), ["1", None]) == {"a": 1, "b": None}


class TestPagination:
    def test_an_empty_total_does_not_call_fetch(self):
        called = False

        def fetch(skip, limit):
            nonlocal called
            called = True
            return []

        page = pagination.paginate(0, PaginationParams(page=1, size=10), fetch)
        assert page.data == []
        assert page.total_pages == 0
        assert called is False

    def test_it_asks_for_the_right_window(self):
        page = pagination.paginate(
            25, PaginationParams(page=3, size=10), lambda skip, limit: [(skip, limit)]
        )
        assert page.data == [(20, 10)]
        assert page.total_pages == 3
        assert page.total == 25

    def test_a_page_past_the_end_is_empty(self):
        page = pagination.paginate(
            5, PaginationParams(page=9, size=10), lambda skip, limit: [1]
        )
        assert page.data == []

    async def test_the_async_counterpart_matches(self):
        async def fetch(skip, limit):
            return [(skip, limit)]

        page = await pagination.apaginate(25, PaginationParams(page=2, size=10), fetch)
        assert page.data == [(10, 10)]
        assert page.total_pages == 3


class TestResources:
    def test_it_closes_what_exists_and_skips_what_does_not(self):
        class Closable:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        first, second = Closable(), Closable()
        resources.close_all(first, None, second)
        assert first.closed and second.closed

    async def test_the_async_counterpart_matches(self):
        class AsyncClosable:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        first, second = AsyncClosable(), AsyncClosable()
        await resources.aclose_all(first, None, second)
        assert first.closed and second.closed


class TestOpenAIBaseUrl:
    def test_azure_gets_the_v1_surface_appended(self):
        from base.models.definitions.model import ProviderType
        from base.tools.openai_clients import resolve_base_url

        assert (
            resolve_base_url("https://x.openai.azure.com", ProviderType.AZURE_OPENAI)
            == "https://x.openai.azure.com/openai/v1/"
        )

    def test_azure_without_a_host_is_rejected(self):
        from base.models.definitions.model import ProviderType
        from base.models.view_models.llm_exception import LLMConfigError
        from base.tools.openai_clients import resolve_base_url

        with pytest.raises(LLMConfigError):
            resolve_base_url(None, ProviderType.AZURE_OPENAI)

    def test_openai_keeps_the_default_host(self):
        from base.models.definitions.model import ProviderType
        from base.tools.openai_clients import resolve_base_url

        assert resolve_base_url(None, ProviderType.OPENAI) is None


def test_uuid_helpers_agree_with_the_stdlib():
    value = uuid4()
    assert isinstance(parse_uuid(str(value)), UUID)
    assert asyncio.iscoroutinefunction(pagination.apaginate)
