"""A slice manifest and a service's settings have to say the same thing.

`tools/new_service.py --from <slice>` composes a new service out of these,
reading `[tool.starter.slice] mixins` to decide what the composed settings
class declares. A mixin a service's own `settings.py` uses but its manifest
omits produces a composed service missing that configuration, which has
already happened here once.
"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil
from collections.abc import Iterator
from typing import Any

import pytest
import tomllib

from base.services.configuration import configure, reset

ROOT = pathlib.Path(__file__).resolve().parents[3]
SERVICES = ROOT / "services"

SETTINGS_CLASS = {
    "identity": ("identity.settings", "IdentitySettings"),
    "chat_api": ("chat_api.settings", "ChatApiSettings"),
    "ingestion_worker": ("ingestion_worker.settings", "IngestionWorkerSettings"),
}

INFRASTRUCTURE = {
    "CoreSettings",
    "ServerSettingsMixin",
    "SQLSettingsMixin",
    "PostgresSettingsMixin",
    "MySQLSettingsMixin",
    "MongoSettingsMixin",
    "RedisSettingsMixin",
    "BrokerSettingsMixin",
    "TokenVerificationSettingsMixin",
}

SLICE_OWNED = {
    # Every mixin a slice's settings composes, manifest-declared or not. The
    # entries beyond the manifest are the per-provider companions its umbrella
    # mixin implies — `OAuthSettingsMixin` plus the five providers identity
    # offers, `LLMSettingsMixin` plus the SDK chat_api is configured for.
    "identity": {
        "TokenIssuingSettingsMixin",
        "OAuthSettingsMixin",
        "IdpSettingsMixin",
        "GoogleOAuthSettingsMixin",
        "FacebookOAuthSettingsMixin",
        "MicrosoftOAuthSettingsMixin",
        "GitHubOAuthSettingsMixin",
        "AppleOAuthSettingsMixin",
    },
    "chat_api": {
        "LLMSettingsMixin",
        "AzureOpenAILLMSettingsMixin",
        "PromptSettingsMixin",
        "RateLimitSettingsMixin",
        "RealtimeSettingsMixin",
    },
    "ingestion_worker": set(),
}


def manifest(service: str) -> dict[str, Any]:
    data = tomllib.loads((SERVICES / service / "pyproject.toml").read_text())
    return data["tool"]["starter"]["slice"]


def settings_class(service: str) -> type:
    module_name, class_name = SETTINGS_CLASS[service]
    return getattr(importlib.import_module(module_name), class_name)


def composed_mixins(service: str) -> set[str]:
    """Every settings mixin named directly in the class statement."""
    return {base.__name__ for base in settings_class(service).__bases__}


@pytest.fixture
def configured(request: pytest.FixtureRequest) -> Iterator[Any]:
    """Register a service's own settings, as its composition root would."""
    service = request.param
    module_name, _ = SETTINGS_CLASS[service]
    settings = importlib.import_module(module_name).settings
    configure(settings)
    yield settings
    reset()


# Manifest agreement


@pytest.mark.parametrize("service", sorted(SETTINGS_CLASS))
def test_every_mixin_the_manifest_declares_is_composed_by_the_settings(
    service: str,
) -> None:
    declared = set(manifest(service)["mixins"])
    composed = composed_mixins(service)

    assert declared <= composed, f"{service} manifest declares {declared - composed}"


@pytest.mark.parametrize("service", sorted(SETTINGS_CLASS))
def test_a_settings_mixin_outside_the_manifest_must_be_a_known_companion(
    service: str,
) -> None:
    declared = set(manifest(service)["mixins"])
    composed = composed_mixins(service)
    slice_owned = composed - INFRASTRUCTURE

    undeclared = slice_owned - declared - SLICE_OWNED[service]
    assert undeclared == set(), f"{service} composes undeclared {undeclared}"


@pytest.mark.parametrize("service", sorted(SETTINGS_CLASS))
def test_a_service_composes_nothing_belonging_to_another_slice(
    service: str,
) -> None:
    composed = composed_mixins(service)
    others = set().union(
        *(owned for name, owned in SLICE_OWNED.items() if name != service)
    )

    assert composed & others == set()


def test_identity_composes_exactly_its_declared_bases() -> None:
    assert (
        composed_mixins("identity")
        == INFRASTRUCTURE
        - {
            "MySQLSettingsMixin",
            "MongoSettingsMixin",
            "TokenVerificationSettingsMixin",
        }
        | SLICE_OWNED["identity"]
    )


def test_chat_api_composes_exactly_its_declared_bases() -> None:
    assert (
        composed_mixins("chat_api")
        == INFRASTRUCTURE
        - {
            "MySQLSettingsMixin",
            "MongoSettingsMixin",
        }
        | SLICE_OWNED["chat_api"]
    )


def test_the_worker_composes_only_infrastructure() -> None:
    assert composed_mixins("ingestion_worker") == {
        "CoreSettings",
        "SQLSettingsMixin",
        "PostgresSettingsMixin",
        "RedisSettingsMixin",
        "BrokerSettingsMixin",
    }


@pytest.mark.parametrize(
    ("service", "publishes"),
    [("identity", True), ("chat_api", True), ("ingestion_worker", False)],
)
def test_a_slice_that_publishes_events_composes_the_broker_settings(
    service: str, publishes: bool
) -> None:
    assert manifest(service)["publishes_events"] is publishes
    assert ("BrokerSettingsMixin" in composed_mixins(service)) is True
    assert ("RedisSettingsMixin" in composed_mixins(service)) is True


def test_issuing_and_verification_are_never_composed_together() -> None:
    composed = composed_mixins("identity")

    assert "TokenIssuingSettingsMixin" in composed
    assert "TokenVerificationSettingsMixin" not in composed


def test_a_slices_extras_name_real_extras_of_base() -> None:
    base_manifest = tomllib.loads(
        (ROOT / "packages" / "base" / "pyproject.toml").read_text()
    )
    available = set(base_manifest["project"]["optional-dependencies"])

    for service in SETTINGS_CLASS:
        assert set(manifest(service)["extras"]) <= available


def test_a_slices_mixins_name_real_mixins_of_base() -> None:
    import base.services.configuration.mixins as mixins

    for service in SETTINGS_CLASS:
        for name in manifest(service)["mixins"]:
            assert hasattr(mixins, name), f"{service} declares unknown {name}"


# Settings identity


@pytest.mark.parametrize(
    ("service", "service_name"),
    [
        ("identity", "identity"),
        ("chat_api", "chat_api"),
        ("ingestion_worker", "ingestion_worker"),
    ],
)
def test_each_service_names_itself(service: str, service_name: str) -> None:
    assert settings_class(service).model_fields["service_name"].default == service_name


@pytest.mark.parametrize("service", ["identity", "chat_api"])
def test_an_http_service_prefixes_its_own_api(service: str) -> None:
    prefix = settings_class(service).model_fields["api_prefix"].default

    assert prefix.startswith("/api")


def test_the_worker_declares_no_http_surface() -> None:
    assert manifest("ingestion_worker")["http"] is False
    assert "ServerSettingsMixin" not in composed_mixins("ingestion_worker")


# Routers


@pytest.mark.parametrize(
    ("configured", "package"),
    [("identity", "identity.routers"), ("chat_api", "chat_api.routers")],
    indirect=["configured"],
)
def test_every_router_module_imports_and_exposes_a_router(
    configured: Any, package: str
) -> None:
    routers = importlib.import_module(package)
    names = [name for _, name, is_pkg in pkgutil.iter_modules(routers.__path__)]

    assert names
    for name in names:
        module = importlib.import_module(f"{package}.{name}")
        assert hasattr(module, "router"), f"{package}.{name} has no router"


@pytest.mark.parametrize("configured", ["identity"], indirect=True)
def test_identity_exposes_auth_user_and_health_routers(configured: Any) -> None:
    routers = importlib.import_module("identity.routers")
    names = {name for _, name, _ in pkgutil.iter_modules(routers.__path__)}

    assert {"auth", "user", "health"} <= names


@pytest.mark.parametrize("configured", ["chat_api"], indirect=True)
def test_chat_api_exposes_a_conversation_router(configured: Any) -> None:
    routers = importlib.import_module("chat_api.routers")
    names = {name for _, name, _ in pkgutil.iter_modules(routers.__path__)}

    assert "conversation" in names


@pytest.mark.parametrize("configured", ["chat_api"], indirect=True)
def test_a_routers_routes_are_registered_under_the_services_prefix(
    configured: Any,
) -> None:
    from fastapi import FastAPI

    from base.tools.routers import init_routers

    app = FastAPI()
    init_routers(app, importlib.import_module("chat_api.routers"), prefix="/api/chat")

    paths = [route.path for route in app.routes]  # type: ignore[attr-defined]
    assert any(path.startswith("/api/chat") for path in paths)


# ORM registration


@pytest.mark.parametrize("configured", ["identity"], indirect=True)
def test_identity_tables_register_on_its_declarative_root(configured: Any) -> None:
    from identity.models.definitions.database import DatabaseTable

    for module in ("user", "user_identity", "outbox"):
        importlib.import_module(f"identity.models.orm.{module}")

    from identity.models.orm import IdentityBase

    registered = set(IdentityBase.metadata.tables)
    assert {t.value for t in DatabaseTable} <= registered
    assert "outbox_event" in registered


@pytest.mark.parametrize("configured", ["chat_api"], indirect=True)
def test_chat_api_tables_register_on_its_declarative_root(configured: Any) -> None:
    from chat_api.models.definitions.database import DatabaseTable

    for module in ("conversation", "chat_history_message", "prompt", "outbox"):
        importlib.import_module(f"chat_api.models.orm.{module}")

    from chat_api.models.orm import ChatBase

    registered = set(ChatBase.metadata.tables)
    assert {t.value for t in DatabaseTable} <= registered
    assert "outbox_event" in registered


@pytest.mark.parametrize("configured", ["ingestion_worker"], indirect=True)
def test_the_workers_inbox_registers_on_its_declarative_root(configured: Any) -> None:
    importlib.import_module("ingestion_worker.models.inbox")

    from ingestion_worker.models import WorkerBase

    assert "inbox_event" in WorkerBase.metadata.tables


@pytest.mark.parametrize("configured", ["chat_api"], indirect=True)
def test_two_services_keep_separate_metadata(configured: Any) -> None:
    from chat_api.models.orm import ChatBase
    from identity.models.orm import IdentityBase

    assert ChatBase.metadata is not IdentityBase.metadata


@pytest.mark.parametrize("configured", ["chat_api"], indirect=True)
def test_chat_api_does_not_own_the_identity_tables(configured: Any) -> None:
    from chat_api.models.orm import ChatBase

    assert "user" not in ChatBase.metadata.tables
    assert "user_identity" not in ChatBase.metadata.tables
