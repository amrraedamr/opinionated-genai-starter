"""Shared fixtures for chat_api tests."""

import pathlib
from collections.abc import Iterator

import pytest

from base.testing import service_environment

SERVICE_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def app_environment() -> Iterator[pathlib.Path]:
    with service_environment(
        "chat_api",
        SERVICE_ROOT,
        {"OPENAI_LLM_API_KEY": "sk-test", "OPENAI_API_KEY": "sk-test"},
    ) as private_key:
        yield private_key


@pytest.fixture(scope="session")
def signing_key(app_environment: pathlib.Path) -> str:
    """Private key of the issuer chat_api trusts.

    chat_api composes TokenVerificationMixin only and cannot mint tokens, so a
    test stands in for the identity service and signs them itself.
    """
    return app_environment.read_text()


@pytest.fixture(scope="session")
def client(app_environment: pathlib.Path):
    from fastapi.testclient import TestClient

    from chat_api.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def api_prefix() -> str:
    from chat_api.settings import settings

    return settings.api_prefix


@pytest.fixture(scope="session")
def configured_settings(app_environment: pathlib.Path) -> None:
    """Register chat_api settings so stores can be built outside the app."""
    from base.services.configuration import configure
    from chat_api.settings import settings

    configure(settings)
