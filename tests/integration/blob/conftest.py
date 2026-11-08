"""A real Azure Blob endpoint: the Azurite the repository's compose file starts.

The emulator authenticates on a shared key rather than an identity, which is
the branch `DefaultAzureCredential` could never reach, so these tests are also
the proof that the key path signs a SAS the server then honours. Every test
names its container uniquely and deletes it afterwards, so reruns never
collide.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from base.services.blob.azure_blob import AzureBlobService
from base.services.configuration.mixins.blob.azure import AzureBlobSettingsMixin

ACCOUNT_URL = "http://127.0.0.1:10100/devstoreaccount1"
"""Azurite on the port the compose file maps, 10000 being taken here."""

ACCOUNT_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw=="
)
"""Azurite's development key. Microsoft publishes it; it unlocks nothing."""


@pytest.fixture
def azurite(make_settings: Any) -> Any:
    """Settings pointed at the local emulator, on a shared key."""
    return make_settings(
        AzureBlobSettingsMixin,
        azure_blob_account_url=ACCOUNT_URL,
        azure_blob_account_key=ACCOUNT_KEY,
    )


@pytest.fixture
async def blob(azurite: Any) -> AsyncIterator[AzureBlobService]:
    """A service pointed at the local container, closed afterwards."""
    service = AzureBlobService()
    yield service
    await service.adispose()
    service.dispose()


@pytest.fixture
async def container(blob: AzureBlobService, unique: str) -> AsyncIterator[str]:
    """An empty container nothing else uses, deleted when the test ends."""
    name = f"t{unique}"
    await blob.acreate_container(name)
    yield name
    if await blob.acontainer_exists(name):
        await blob.adelete_container(name)
