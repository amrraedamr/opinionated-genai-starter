"""Shared fixtures for identity tests."""

import pathlib
from collections.abc import Iterator

import pytest

from base.testing import service_environment

SERVICE_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def app_environment() -> Iterator[pathlib.Path]:
    with service_environment("identity", SERVICE_ROOT) as private_key:
        yield private_key


@pytest.fixture(scope="session")
def client(app_environment: pathlib.Path):
    from fastapi.testclient import TestClient

    from identity.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def api_prefix() -> str:
    from identity.settings import settings

    return settings.api_prefix


@pytest.fixture(scope="session")
def configured_settings(app_environment: pathlib.Path) -> None:
    """Register identity settings so stores can be built outside the app."""
    from base.services.configuration import configure
    from identity.settings import settings

    configure(settings)
