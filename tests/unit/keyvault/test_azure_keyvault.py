"""`AzureKeyVaultService`: the names it builds, and the errors it translates.

There is no vault here, so the SDK client is a fake and the assertions are on
the secret name that would have gone out. `format_secret_name` is checked for
what it is *not* wrapped in as much as for what it returns: it makes no network
call, so a bad name must surface as a plain `ValueError`.
"""

import inspect
from collections.abc import Iterator
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

from base.models.view_models.exceptions.keyvault import (
    KeyVaultAuthError,
    KeyVaultConflictError,
    KeyVaultError,
    KeyVaultNotFoundError,
    KeyVaultRateLimitError,
    KeyVaultServerError,
    KeyVaultUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.models.view_models.keyvault import SecretInfo
from base.services.common.meta import purge_all
from base.services.configuration.mixins import KeyVaultSettingsMixin
from base.services.keyvault import AbstractKeyVaultService
from base.services.keyvault.azure_keyvault import AzureKeyVaultService
from tests.conftest import MakeSettings

VAULT_URL = "https://contoso.vault.azure.net/"


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """The service is a singleton, so one test's fake must not reach the next."""
    purge_all()
    yield
    purge_all()


class FakeHttpResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.reason = "fake"
        self.content = b""


def http_error(status: int, headers: dict[str, str] | None = None) -> HttpResponseError:
    error = HttpResponseError(response=FakeHttpResponse(status, headers))  # type: ignore[arg-type]
    error.status_code = status
    return error


class FakeSecretClient:
    """A vault that records the names it was asked for."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requested: list[str] = []
        self.written: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    def get_secret(self, name: str) -> Any:
        self.requested.append(name)
        self._check()
        return SimpleNamespace(value=f"value-of-{name}")

    def set_secret(self, key: str, value: str) -> Any:
        self.written.append((key, value))
        self._check()
        return SimpleNamespace(properties=SimpleNamespace(version="v7"))

    def begin_delete_secret(self, name: str) -> None:
        self.deleted.append(name)
        self._check()

    def list_properties_of_secrets(self) -> Iterator[Any]:
        self._check()
        yield SimpleNamespace(
            name="appconfig-db-password",
            version="v1",
            enabled=True,
            created_on=datetime(2026, 1, 1, tzinfo=UTC),
            updated_on=datetime(2026, 2, 1, tzinfo=UTC),
        )
        yield SimpleNamespace(
            name="appconfig-legacy",
            version=None,
            enabled=None,
            created_on=None,
            updated_on=None,
        )


class AsyncFakeSecretClient(FakeSecretClient):
    async def close(self) -> None:  # type: ignore[override]
        self.closed += 1

    async def get_secret(self, name: str) -> Any:  # type: ignore[override]
        return FakeSecretClient.get_secret(self, name)

    async def set_secret(self, key: str, value: str) -> Any:  # type: ignore[override]
        return FakeSecretClient.set_secret(self, key, value)

    async def delete_secret(self, name: str) -> None:
        self.deleted.append(name)
        self._check()

    def list_properties_of_secrets(self) -> Any:  # type: ignore[override]
        async def gen() -> Any:
            for item in FakeSecretClient.list_properties_of_secrets(self):
                yield item

        return gen()


def build(
    make_settings: MakeSettings,
    error: Exception | None = None,
    **settings: Any,
) -> tuple[AzureKeyVaultService, FakeSecretClient]:
    """A service whose SDK clients are fakes, with settings already registered."""
    settings.setdefault("keyvault_url", VAULT_URL)
    make_settings(KeyVaultSettingsMixin, **settings)
    service = AzureKeyVaultService()
    client = FakeSecretClient(error)
    service._client = client  # type: ignore[assignment]
    service._async_client = AsyncFakeSecretClient(error)  # type: ignore[assignment]
    return service, client


def test_the_service_implements_the_neutral_contract(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert isinstance(service, AbstractKeyVaultService)
    assert service.get_service_name() == "Azure Key Vault"


def test_a_bare_name_is_prefixed_with_the_system_tag(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert service.format_secret_name("db-password") == "appconfig-db-password"


def test_a_name_that_already_carries_the_system_tag_is_left_alone(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert service.format_secret_name("appconfig-db") == "appconfig-db"


def test_a_name_that_already_carries_the_model_tag_is_left_alone(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert service.format_secret_name("modelconfig-gpt5") == "modelconfig-gpt5"


def test_the_tags_come_from_settings_rather_than_being_hard_coded(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(
        make_settings, system_keyvault_tag="sys", model_keyvault_tag="mdl"
    )
    assert service.format_secret_name("db") == "sys-db"
    assert service.format_secret_name("mdl-x") == "mdl-x"


def test_a_tagged_name_outside_the_allowed_prefix_is_refused(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    with pytest.raises(ValueError, match="does not start with the allowed prefix"):
        service.format_secret_name("appconfig-db", allowed_prefix="modelconfig")


def test_that_refusal_is_a_plain_value_error_not_a_provider_error(
    make_settings: MakeSettings,
) -> None:
    """It makes no network call, so wrapping it in the vault family would say
    the vault failed when the caller's own argument was wrong."""
    service, _ = build(make_settings)
    with pytest.raises(ValueError) as caught:
        service.format_secret_name("appconfig-db", allowed_prefix="modelconfig")
    assert not isinstance(caught.value, ProviderError)
    assert not isinstance(caught.value, KeyVaultError)


def test_name_formatting_is_not_retry_decorated(make_settings: MakeSettings) -> None:
    """Nothing to retry, so repeating it would only repeat the same refusal."""
    assert not hasattr(AzureKeyVaultService.format_secret_name, "__wrapped__")
    assert not inspect.iscoroutinefunction(AzureKeyVaultService.format_secret_name)


def test_an_allowed_prefix_that_matches_lets_the_name_through(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert service.format_secret_name("appconfig-db", allowed_prefix="appconfig") == (
        "appconfig-db"
    )


def test_reading_a_secret_asks_the_vault_for_the_formatted_name(
    make_settings: MakeSettings,
) -> None:
    service, client = build(make_settings)
    assert service.get_secret("db-password") == "value-of-appconfig-db-password"
    assert client.requested == ["appconfig-db-password"]


def test_writing_a_secret_returns_the_version_the_vault_assigned(
    make_settings: MakeSettings,
) -> None:
    service, client = build(make_settings)
    assert service.upsert_secret("db-password", "hunter2") == "v7"
    assert client.written == [("appconfig-db-password", "hunter2")]


def test_writing_can_be_told_to_use_the_name_exactly_as_given(
    make_settings: MakeSettings,
) -> None:
    service, client = build(make_settings)
    service.upsert_secret("raw-name", "value", format_name=False)
    assert client.written == [("raw-name", "value")]


def test_a_vault_that_returns_no_version_is_treated_as_a_failure(
    make_settings: MakeSettings,
) -> None:
    service, client = build(make_settings)

    def versionless(key: str, value: str) -> Any:
        return SimpleNamespace(properties=SimpleNamespace(version=None))

    client.set_secret = versionless  # type: ignore[method-assign]
    with pytest.raises(KeyVaultUnexpectedError):
        service.upsert_secret("db", "v")


def test_removing_a_secret_addresses_the_formatted_name(
    make_settings: MakeSettings,
) -> None:
    service, client = build(make_settings)
    service.remove_secret("db-password")
    assert client.deleted == ["appconfig-db-password"]


def test_listing_returns_the_neutral_shape_rather_than_the_sdk_type(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    listed = list(service.list_secrets())
    assert all(isinstance(item, SecretInfo) for item in listed)
    assert listed[0].name == "appconfig-db-password"
    assert listed[0].version == "v1"
    assert listed[0].created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert listed[0].updated_at == datetime(2026, 2, 1, tzinfo=UTC)


def test_a_secret_whose_enabled_flag_is_absent_is_treated_as_enabled(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    listed = list(service.list_secrets())
    assert listed[1].enabled is True
    assert listed[1].version is None


def test_reading_several_secrets_fetches_each_one_on_its_own(
    make_settings: MakeSettings,
) -> None:
    service, client = build(make_settings)
    assert service.get_secrets(["a", "b"]) == {
        "a": "value-of-appconfig-a",
        "b": "value-of-appconfig-b",
    }
    assert client.requested == ["appconfig-a", "appconfig-b"]


@pytest.mark.parametrize(
    ("sdk_error", "domain_error"),
    [
        (ClientAuthenticationError("no access"), KeyVaultAuthError),
        (ResourceNotFoundError("no such secret"), KeyVaultNotFoundError),
        (ResourceExistsError("soft deleted"), KeyVaultConflictError),
    ],
)
def test_an_sdk_error_becomes_the_matching_vault_error(
    make_settings: MakeSettings,
    sdk_error: AzureError,
    domain_error: type[KeyVaultError],
) -> None:
    service, _ = build(make_settings, error=sdk_error)
    with pytest.raises(domain_error) as caught:
        service.get_secret("db")
    assert caught.value.provider_error is sdk_error


def test_a_transient_failure_ends_as_a_retry_exhausted_vault_error(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    service, _ = build(make_settings, error=ServiceResponseTimeoutError("slow"))
    with pytest.raises(KeyVaultError) as caught:
        service.get_secret("db")
    assert caught.value.attempts > 1  # type: ignore[attr-defined]


def test_a_throttled_vault_ends_as_a_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    service, _ = build(
        make_settings,
        error=http_error(429, {"Retry-After": "0", "x-ms-request-id": "rid"}),
    )
    with pytest.raises(KeyVaultRateLimitError) as caught:
        service.get_secret("db")
    assert caught.value.status_code == 429
    assert caught.value.request_id == "rid"


def test_a_failed_vault_ends_as_a_server_error(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    service, _ = build(make_settings, error=http_error(500))
    with pytest.raises(KeyVaultServerError):
        service.get_secret("db")


def test_something_the_sdk_never_promised_becomes_an_unexpected_vault_error(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings, error=TypeError("surprise"))
    with pytest.raises(KeyVaultUnexpectedError) as caught:
        service.get_secret("db")
    assert isinstance(caught.value.provider_error, TypeError)


def test_a_retryable_failure_is_tried_the_handlers_full_budget(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    from tests.fakes import no_sleep

    no_sleep(monkeypatch)
    service, client = build(make_settings, error=ServiceRequestError("no route"))
    with pytest.raises(KeyVaultError):
        service.get_secret("db")
    assert len(client.requested) == service._error_handler.DEFAULT_RETRIES + 1


def test_an_unretryable_failure_is_tried_once(make_settings: MakeSettings) -> None:
    service, client = build(make_settings, error=ResourceNotFoundError("gone"))
    with pytest.raises(KeyVaultNotFoundError):
        service.get_secret("db")
    assert len(client.requested) == 1


def test_every_vault_call_is_retry_decorated_because_each_is_one_request() -> None:
    for name in (
        "get_secret",
        "aget_secret",
        "upsert_secret",
        "aupsert_secret",
        "remove_secret",
        "aremove_secret",
    ):
        assert hasattr(getattr(AzureKeyVaultService, name), "__wrapped__"), name


def test_listing_is_a_generator_expression_and_so_is_not_retry_decorated() -> None:
    assert not hasattr(AzureKeyVaultService.list_secrets, "__wrapped__")
    assert not hasattr(AzureKeyVaultService.alist_secrets, "__wrapped__")


async def test_reading_a_secret_asynchronously_formats_the_name_too(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert await service.aget_secret("db") == "value-of-appconfig-db"


async def test_writing_and_removing_asynchronously_reach_the_same_vault(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert await service.aupsert_secret("db", "v") == "v7"
    await service.aremove_secret("db")
    assert service._async_client.deleted == ["appconfig-db"]  # type: ignore[union-attr]


async def test_reading_several_secrets_concurrently_keeps_them_in_order(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    assert await service.aget_secrets(["a", "b", "c"]) == {
        "a": "value-of-appconfig-a",
        "b": "value-of-appconfig-b",
        "c": "value-of-appconfig-c",
    }


async def test_reading_several_secrets_one_at_a_time_is_also_offered(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    result = await service.aget_secrets(["a", "b"], concurrent=False)
    assert list(result) == ["a", "b"]
    assert service._async_client.requested == [  # type: ignore[union-attr]
        "appconfig-a",
        "appconfig-b",
    ]


async def test_listing_asynchronously_returns_the_neutral_shape(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    listed = [item async for item in service.alist_secrets()]
    assert [item.name for item in listed] == [
        "appconfig-db-password",
        "appconfig-legacy",
    ]


async def test_an_async_sdk_error_becomes_a_vault_error(
    make_settings: MakeSettings,
) -> None:
    error = ResourceNotFoundError("gone")
    service, _ = build(make_settings, error=error)
    with pytest.raises(KeyVaultNotFoundError) as caught:
        await service.aget_secret("db")
    assert caught.value.provider_error is error


def test_closing_releases_the_client_and_the_credential(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    closed: list[str] = []

    class Closeable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    service._client = Closeable("client")  # type: ignore[assignment]
    service._credential = Closeable("credential")  # type: ignore[assignment]
    service.close()
    assert closed == ["client", "credential"]
    assert service._client is None


async def test_closing_asynchronously_releases_both_async_halves(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    closed: list[str] = []

    class AsyncCloseable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

    service._async_client = AsyncCloseable("client")  # type: ignore[assignment]
    service._async_credential = AsyncCloseable("credential")  # type: ignore[assignment]
    await service.aclose()
    assert closed == ["client", "credential"]
    assert service._async_client is None


def test_the_context_manager_closes_on_the_way_out(make_settings: MakeSettings) -> None:
    service, _ = build(make_settings)
    with service as entered:
        assert entered is service
    assert service._client is None


async def test_the_async_context_manager_closes_on_the_way_out(
    make_settings: MakeSettings,
) -> None:
    service, _ = build(make_settings)
    async with service as entered:
        assert entered is service
    assert service._async_client is None
