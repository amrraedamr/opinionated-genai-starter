"""Proves the integration marker reaches the real containers."""

import pytest


@pytest.mark.integration
async def test_redis_is_reachable(infra_settings):
    from base.clients.redis_client import RedisClient

    assert await RedisClient().aping() is True


@pytest.mark.integration
async def test_postgres_round_trips(infra_settings):
    from sqlalchemy import text

    from base.services.sql.service import SQLService

    async with SQLService()._aget_session() as session:
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
