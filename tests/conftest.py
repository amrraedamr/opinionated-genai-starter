"""Shared fixtures: a migrated throwaway database and a signing keypair.

Settings are constructed when `base.services.configuration` is first imported,
and pytest imports test modules before any fixture runs. So the environment is
set here at conftest import, which is earlier than any of that.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _write_keypair(directory: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    private = directory / "private.pem"
    public = directory / "public.pem"
    subprocess.run(
        ["openssl", "genrsa", "-out", str(private), "2048"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "rsa", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    return private, public


_DIRECTORY = pathlib.Path(tempfile.mkdtemp(prefix="starter-tests-"))
atexit.register(shutil.rmtree, _DIRECTORY, True)

PRIVATE_KEY, PUBLIC_KEY = _write_keypair(_DIRECTORY)

os.environ.update(
    {
        "APP_NAME": "starter-tests",
        "ENVIRONMENT": "local",
        "SQL_BACKEND": "sqlite",
        "SQLITE_DB_PATH": str(_DIRECTORY / "test.db"),
        "AUTH_PRIVATE_KEY_PATH": str(PRIVATE_KEY),
        "AUTH_PUBLIC_KEY_PATH": str(PUBLIC_KEY),
        "OPENAI_LLM_API_KEY": "sk-test",
        "OPENAI_API_KEY": "sk-test",
        "AZURE_OPENAI_LLM_API_KEY": "sk-test",
        "AZURE_OPENAI_LLM_ENDPOINT": "https://example.openai.azure.com",
        "AZURE_OPENAI_LLM_DEPLOYMENT_NAME": "gpt-test",
    }
)


@pytest.fixture(scope="session", autouse=True)
def app_environment() -> Iterator[pathlib.Path]:
    """Bring the throwaway database up to head before anything touches it."""
    subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    yield PRIVATE_KEY


@pytest.fixture(scope="session")
def client(app_environment: pathlib.Path):
    from fastapi.testclient import TestClient

    from base.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def api_prefix() -> str:
    return "/api/7r"
