"""`AzureBlobService`: the requests it builds and the errors it translates.

There are no credentials and no emulator here, so the Azure SDK client is a
fake and the assertions are on what would have gone out. The one method that
is deliberately *not* retry-decorated is a generator, and that is pinned too.
"""

import inspect
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseTimeoutError,
)

from base.models.view_models.blob import BlobProperties
from base.models.view_models.exceptions.blob import (
    BlobAuthError,
    BlobConflictError,
    BlobError,
    BlobNotFoundError,
    BlobRateLimitError,
    BlobServerError,
    BlobTimeoutError,
    BlobUnexpectedError,
)
from base.services.blob import AbstractBlobService
from base.services.blob.azure_blob import AzureBlobService
from base.services.common.meta import purge_all
from base.services.configuration.mixins import AzureBlobSettingsMixin

ACCOUNT_URL = "https://account.blob.core.windows.net/"


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """The service and its client are singletons."""
    purge_all()
    yield
    purge_all()


class FakeBlobClient:
    """One blob, recording what the service asked of it."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.uploads: list[dict[str, Any]] = []
        self.deletes = 0
        self.exists_value = True

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    def upload_blob(self, data: bytes, **kwargs: Any) -> None:
        self._check()
        self.uploads.append({"data": data, **kwargs})

    async def _aupload(self, data: bytes, **kwargs: Any) -> None:
        self.upload_blob(data, **kwargs)

    def download_blob(self) -> Any:
        self._check()
        return SimpleNamespace(readall=lambda: b"stored bytes")

    def delete_blob(self) -> None:
        self._check()
        self.deletes += 1

    def exists(self) -> bool:
        self._check()
        return self.exists_value

    def get_blob_properties(self) -> Any:
        self._check()
        return SimpleNamespace(
            name="report.pdf",
            size=2048,
            content_settings=SimpleNamespace(content_type="application/pdf"),
            etag='"abc"',
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            metadata={"owner": "ada"},
        )


class FakeContainerClient:
    def __init__(self, blobs: list[Any] | None = None) -> None:
        self.blobs = blobs or []
        self.prefixes: list[str | None] = []
        self.created = 0
        self.exists_value = True

    def exists(self) -> bool:
        return self.exists_value

    def create_container(self) -> None:
        self.created += 1

    def list_blobs(self, name_starts_with: str | None = None) -> Iterator[Any]:
        self.prefixes.append(name_starts_with)
        yield from self.blobs


class FakeServiceClient:
    def __init__(
        self, blob: FakeBlobClient, container: FakeContainerClient | None = None
    ) -> None:
        self.blob = blob
        self.container = container or FakeContainerClient()
        self.blob_requests: list[tuple[str, str]] = []
        self.container_requests: list[str] = []
        self.deleted_containers: list[str] = []

    def get_blob_client(self, container: str, blob_name: str) -> FakeBlobClient:
        self.blob_requests.append((container, blob_name))
        return self.blob

    def get_container_client(self, container: str) -> FakeContainerClient:
        self.container_requests.append(container)
        return self.container

    def delete_container(self, container: str) -> None:
        self.deleted_containers.append(container)


class FakeClientHolder:
    """Stands in for `AzureBlobClient`, which owns both SDK clients."""

    def __init__(self, service_client: Any) -> None:
        self.sync_client = service_client
        self.async_client = service_client
        self.closed = 0
        self.aclosed = 0

    def close(self) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.aclosed += 1


class FakeHttpResponse:
    """The minimum `HttpResponseError` reads off a response."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.reason = "fake"
        self.content = b""


def http_error(status: int, headers: dict[str, str] | None = None) -> HttpResponseError:
    """An `HttpResponseError` Azure would have raised for `status`."""
    error = HttpResponseError(response=FakeHttpResponse(status, headers))  # type: ignore[arg-type]
    error.status_code = status
    return error


def build(
    make_settings: Any,
    blob: FakeBlobClient | None = None,
    container: FakeContainerClient | None = None,
) -> tuple[AzureBlobService, FakeServiceClient]:
    """A service whose SDK client is a fake, with settings already registered."""
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    service_client = FakeServiceClient(blob or FakeBlobClient(), container)
    service = AzureBlobService()
    service._client = FakeClientHolder(service_client)  # type: ignore[assignment]
    return service, service_client


def test_the_service_implements_the_neutral_contract(make_settings: Any) -> None:
    service, _ = build(make_settings)
    assert isinstance(service, AbstractBlobService)
    assert service.get_service_name() == "Azure Blob Storage"


def test_uploading_sends_the_bytes_the_content_type_and_the_metadata(
    make_settings: Any,
) -> None:
    blob = FakeBlobClient()
    service, client = build(make_settings, blob)
    service.upload(
        "reports",
        "q1.pdf",
        b"%PDF",
        content_type="application/pdf",
        metadata={"owner": "ada"},
    )
    assert client.blob_requests == [("reports", "q1.pdf")]
    sent = blob.uploads[0]
    assert sent["data"] == b"%PDF"
    assert sent["overwrite"] is True
    assert sent["content_settings"].content_type == "application/pdf"
    assert sent["metadata"] == {"owner": "ada"}


def test_uploading_defaults_to_overwriting_with_an_opaque_content_type(
    make_settings: Any,
) -> None:
    blob = FakeBlobClient()
    service, _ = build(make_settings, blob)
    service.upload("reports", "raw.bin", b"\x00")
    sent = blob.uploads[0]
    assert sent["overwrite"] is True
    assert sent["content_settings"].content_type == "application/octet-stream"
    assert sent["metadata"] is None


def test_uploading_can_be_told_not_to_overwrite(make_settings: Any) -> None:
    blob = FakeBlobClient()
    service, _ = build(make_settings, blob)
    service.upload("reports", "q1.pdf", b"x", overwrite=False)
    assert blob.uploads[0]["overwrite"] is False


def test_downloading_reads_the_whole_blob(make_settings: Any) -> None:
    service, client = build(make_settings)
    assert service.download("reports", "q1.pdf") == b"stored bytes"
    assert client.blob_requests == [("reports", "q1.pdf")]


def test_deleting_addresses_the_named_blob(make_settings: Any) -> None:
    blob = FakeBlobClient()
    service, client = build(make_settings, blob)
    service.delete("reports", "q1.pdf")
    assert blob.deletes == 1
    assert client.blob_requests == [("reports", "q1.pdf")]


def test_existence_is_reported_as_the_sdk_answered_it(make_settings: Any) -> None:
    blob = FakeBlobClient()
    service, _ = build(make_settings, blob)
    assert service.exists("reports", "q1.pdf") is True
    blob.exists_value = False
    assert service.exists("reports", "q1.pdf") is False


def test_properties_are_returned_in_the_neutral_shape(make_settings: Any) -> None:
    service, _ = build(make_settings)
    props = service.get_properties("reports", "q1.pdf")
    assert isinstance(props, BlobProperties)
    assert props.name == "report.pdf"
    assert props.size == 2048
    assert props.content_type == "application/pdf"
    assert props.etag == '"abc"'
    assert props.last_modified == "2026-01-01T00:00:00+00:00"
    assert props.metadata == {"owner": "ada"}


def test_properties_fall_back_when_the_sdk_reports_nothing(make_settings: Any) -> None:
    service, _ = build(make_settings)
    props = AzureBlobService._to_blob_properties(SimpleNamespace())
    assert props.name == ""
    assert props.size == 0
    assert props.content_type == "application/octet-stream"
    assert props.last_modified is None
    assert props.metadata == {}


def test_creating_a_container_that_exists_does_nothing(make_settings: Any) -> None:
    container = FakeContainerClient()
    container.exists_value = True
    service, _ = build(make_settings, container=container)
    service.create_container("reports")
    assert container.created == 0


def test_creating_a_container_that_is_absent_creates_it(make_settings: Any) -> None:
    container = FakeContainerClient()
    container.exists_value = False
    service, _ = build(make_settings, container=container)
    service.create_container("reports")
    assert container.created == 1


def test_deleting_a_container_addresses_it_by_name(make_settings: Any) -> None:
    service, client = build(make_settings)
    service.delete_container("reports")
    assert client.deleted_containers == ["reports"]


def test_listing_passes_the_prefix_through_as_a_name_filter(
    make_settings: Any,
) -> None:
    container = FakeContainerClient(
        blobs=[
            SimpleNamespace(
                name="2026/q1.pdf",
                size=10,
                content_settings=SimpleNamespace(content_type="application/pdf"),
                etag='"e"',
                last_modified=datetime(2026, 1, 1, tzinfo=UTC),
                metadata={"a": "b"},
            )
        ]
    )
    service, _ = build(make_settings, container=container)
    listed = list(service.list_blobs("reports", prefix="2026/"))
    assert container.prefixes == ["2026/"]
    assert [b.name for b in listed] == ["2026/q1.pdf"]
    assert listed[0].metadata == {"a": "b"}


def test_a_listed_blob_missing_its_settings_gets_the_opaque_content_type(
    make_settings: Any,
) -> None:
    container = FakeContainerClient(
        blobs=[
            SimpleNamespace(
                name="x",
                size=None,
                content_settings=None,
                etag=None,
                last_modified=None,
                metadata=None,
            )
        ]
    )
    service, _ = build(make_settings, container=container)
    listed = list(service.list_blobs("reports"))[0]
    assert listed.size == 0
    assert listed.content_type == "application/octet-stream"
    assert listed.last_modified is None
    assert listed.metadata == {}


def test_listing_is_a_generator_and_so_is_not_retry_decorated() -> None:
    """Calling a generator does no work, so a retry wrapper could never see
    the failure. The decorator's own docstring forbids it; this pins it."""
    assert inspect.isgeneratorfunction(AzureBlobService.list_blobs)
    assert inspect.isasyncgenfunction(AzureBlobService.alist_blobs)
    assert not hasattr(AzureBlobService.list_blobs, "__wrapped__")
    assert not hasattr(AzureBlobService.alist_blobs, "__wrapped__")


def test_every_single_shot_method_is_retry_decorated() -> None:
    for name in (
        "upload",
        "aupload",
        "download",
        "adownload",
        "delete",
        "adelete",
        "exists",
        "aexists",
        "get_properties",
        "aget_properties",
        "create_container",
        "delete_container",
        "container_exists",
    ):
        assert hasattr(getattr(AzureBlobService, name), "__wrapped__"), name


def test_a_failure_while_listing_escapes_untranslated(make_settings: Any) -> None:
    """The consequence of not decorating a generator, pinned as it stands."""

    class Exploding:
        def list_blobs(self, name_starts_with: str | None = None) -> Iterator[Any]:
            raise ResourceNotFoundError("no such container")

    service, _ = build(make_settings)
    service._client = FakeClientHolder(  # type: ignore[assignment]
        FakeServiceClient(FakeBlobClient(), Exploding())  # type: ignore[arg-type]
    )
    generator = service.list_blobs("missing")
    with pytest.raises(ResourceNotFoundError):
        list(generator)


@pytest.mark.parametrize(
    ("sdk_error", "domain_error"),
    [
        (ClientAuthenticationError("bad key"), BlobAuthError),
        (ResourceNotFoundError("gone"), BlobNotFoundError),
        (ResourceExistsError("already there"), BlobConflictError),
    ],
    ids=lambda value: (
        type(value).__name__ if isinstance(value, Exception) else value.__name__
    ),
)
def test_an_sdk_error_becomes_the_matching_blob_error(
    make_settings: Any, sdk_error: AzureError, domain_error: type[BlobError]
) -> None:
    service, _ = build(make_settings, FakeBlobClient(error=sdk_error))
    with pytest.raises(domain_error) as caught:
        service.download("reports", "q1.pdf")
    assert caught.value.provider_error is sdk_error


@pytest.mark.parametrize(
    ("sdk_error", "domain_error"),
    [
        (ServiceResponseTimeoutError("slow"), BlobTimeoutError),
        (ServiceRequestError("no route"), BlobError),
    ],
)
def test_a_transient_sdk_error_ends_as_a_retry_exhausted_blob_error(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Any,
    sdk_error: AzureError,
    domain_error: type[BlobError],
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    service, _ = build(make_settings, FakeBlobClient(error=sdk_error))
    with pytest.raises(domain_error) as caught:
        service.download("reports", "q1.pdf")
    assert caught.value.attempts > 1  # type: ignore[attr-defined]
    assert caught.value.provider_error is sdk_error


def test_a_throttled_account_ends_as_a_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch, make_settings: Any
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    error = http_error(429, {"Retry-After": "0", "x-ms-request-id": "req-9"})
    service, _ = build(make_settings, FakeBlobClient(error=error))
    with pytest.raises(BlobRateLimitError) as caught:
        service.exists("reports", "q1.pdf")
    assert caught.value.status_code == 429
    assert caught.value.request_id == "req-9"


def test_a_failed_server_ends_as_a_server_error(
    monkeypatch: pytest.MonkeyPatch, make_settings: Any
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    service, _ = build(make_settings, FakeBlobClient(error=http_error(503)))
    with pytest.raises(BlobServerError):
        service.download("reports", "q1.pdf")


def test_something_the_sdk_never_promised_becomes_an_unexpected_blob_error(
    make_settings: Any,
) -> None:
    service, _ = build(make_settings, FakeBlobClient(error=KeyError("surprise")))
    with pytest.raises(BlobUnexpectedError) as caught:
        service.download("reports", "q1.pdf")
    assert isinstance(caught.value.provider_error, KeyError)


def test_a_retryable_failure_is_tried_the_handlers_full_budget(
    monkeypatch: pytest.MonkeyPatch, make_settings: Any
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    attempts = {"count": 0}

    class Counting(FakeBlobClient):
        def download_blob(self) -> Any:
            attempts["count"] += 1
            raise ServiceRequestError("no route")

    service, _ = build(make_settings, Counting())
    with pytest.raises(BlobError):
        service.download("reports", "q1.pdf")
    assert attempts["count"] == service._error_handler.DEFAULT_RETRIES + 1


def test_an_unretryable_failure_is_tried_once(make_settings: Any) -> None:
    attempts = {"count": 0}

    class Counting(FakeBlobClient):
        def download_blob(self) -> Any:
            attempts["count"] += 1
            raise ResourceNotFoundError("gone")

    service, _ = build(make_settings, Counting())
    with pytest.raises(BlobNotFoundError):
        service.download("reports", "q1.pdf")
    assert attempts["count"] == 1


def test_disposing_closes_the_client_it_owns(make_settings: Any) -> None:
    service, _ = build(make_settings)
    service.dispose()
    assert service._client.closed == 1  # type: ignore[attr-defined]


async def test_disposing_asynchronously_closes_the_async_client(
    make_settings: Any,
) -> None:
    service, _ = build(make_settings)
    await service.adispose()
    assert service._client.aclosed == 1  # type: ignore[attr-defined]


class AsyncFakeBlobClient:
    """The async half of the SDK, with the same recording."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.uploads: list[dict[str, Any]] = []
        self.deletes = 0

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    async def upload_blob(self, data: bytes, **kwargs: Any) -> None:
        self._check()
        self.uploads.append({"data": data, **kwargs})

    async def download_blob(self) -> Any:
        self._check()

        class Stream:
            async def readall(self) -> bytes:
                return b"stored bytes"

        return Stream()

    async def delete_blob(self) -> None:
        self._check()
        self.deletes += 1

    async def exists(self) -> bool:
        self._check()
        return True

    async def get_blob_properties(self) -> Any:
        self._check()
        return SimpleNamespace(
            name="n",
            size=1,
            content_settings=None,
            etag=None,
            last_modified=None,
            metadata=None,
        )


class AsyncFakeContainerClient:
    def __init__(self, blobs: list[Any] | None = None) -> None:
        self.blobs = blobs or []
        self.prefixes: list[str | None] = []

    def list_blobs(self, name_starts_with: str | None = None) -> AsyncIterator[Any]:
        self.prefixes.append(name_starts_with)

        async def gen() -> AsyncIterator[Any]:
            for blob in self.blobs:
                yield blob

        return gen()


class AsyncFakeServiceClient:
    def __init__(self, blob: Any, container: Any = None) -> None:
        self.blob = blob
        self.container = container or AsyncFakeContainerClient()
        self.blob_requests: list[tuple[str, str]] = []

    def get_blob_client(self, container: str, blob_name: str) -> Any:
        self.blob_requests.append((container, blob_name))
        return self.blob

    def get_container_client(self, container: str) -> Any:
        return self.container


def build_async(
    make_settings: Any, blob: Any = None, container: Any = None
) -> tuple[AzureBlobService, AsyncFakeServiceClient]:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    client = AsyncFakeServiceClient(blob or AsyncFakeBlobClient(), container)
    service = AzureBlobService()
    service._client = FakeClientHolder(client)  # type: ignore[assignment]
    return service, client


async def test_uploading_asynchronously_sends_the_same_request(
    make_settings: Any,
) -> None:
    blob = AsyncFakeBlobClient()
    service, client = build_async(make_settings, blob)
    await service.aupload("reports", "q1.pdf", b"%PDF", content_type="application/pdf")
    assert client.blob_requests == [("reports", "q1.pdf")]
    assert blob.uploads[0]["content_settings"].content_type == "application/pdf"


async def test_downloading_asynchronously_reads_the_whole_blob(
    make_settings: Any,
) -> None:
    service, _ = build_async(make_settings)
    assert await service.adownload("reports", "q1.pdf") == b"stored bytes"


async def test_deleting_and_checking_asynchronously_reach_the_same_blob(
    make_settings: Any,
) -> None:
    blob = AsyncFakeBlobClient()
    service, _ = build_async(make_settings, blob)
    await service.adelete("reports", "q1.pdf")
    assert blob.deletes == 1
    assert await service.aexists("reports", "q1.pdf") is True


async def test_listing_asynchronously_passes_the_prefix_through(
    make_settings: Any,
) -> None:
    container = AsyncFakeContainerClient(
        blobs=[
            SimpleNamespace(
                name="2026/q1.pdf",
                size=10,
                content_settings=SimpleNamespace(content_type="application/pdf"),
                etag=None,
                last_modified=None,
                metadata=None,
            )
        ]
    )
    service, _ = build_async(make_settings, container=container)
    listed = [blob async for blob in service.alist_blobs("reports", prefix="2026/")]
    assert container.prefixes == ["2026/"]
    assert [b.name for b in listed] == ["2026/q1.pdf"]


async def test_an_async_sdk_error_becomes_a_blob_error(make_settings: Any) -> None:
    error = ResourceNotFoundError("gone")
    service, _ = build_async(make_settings, AsyncFakeBlobClient(error=error))
    with pytest.raises(BlobNotFoundError) as caught:
        await service.adownload("reports", "q1.pdf")
    assert caught.value.provider_error is error
