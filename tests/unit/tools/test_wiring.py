"""`tools/database.py`, `tools/routers.py`, `tools/webserver.py`, `tools/migrations.py`.

Thin wiring, each with one decision worth pinning: which SQLAlchemy accessor a
format name maps to, that a broken router module is skipped rather than
crashing startup, and that an app path or a settings object is rejected before
anything is served or migrated.
"""

import sys
import types
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI

from base.tools.database import SQLResultFormats
from base.tools.migrations import run_migrations
from base.tools.routers import init_routers
from base.tools.webserver import _import_app


class FakeResult:
    """Stands in for a SQLAlchemy `Result`, recording which accessor was used."""

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.used: list[str] = []

    def first(self) -> Any:
        self.used.append("first")
        return self.rows[0] if self.rows else None

    def scalar_one_or_none(self) -> Any:
        self.used.append("scalar_one_or_none")
        return self.rows[0] if self.rows else None

    def scalars(self) -> "FakeResult":
        self.used.append("scalars")
        return self

    def all(self) -> list[Any]:
        self.used.append("all")
        return self.rows

    def mappings(self) -> "FakeResult":
        self.used.append("mappings")
        return self


def test_one_or_none_takes_the_first_row() -> None:
    result = FakeResult(["row"])
    assert SQLResultFormats.one_or_none(result) == "row"  # type: ignore[arg-type]
    assert result.used == ["first"]


def test_one_or_none_answers_none_for_an_empty_result() -> None:
    assert SQLResultFormats.one_or_none(FakeResult([])) is None  # type: ignore[arg-type]


def test_one_or_none_scalar_unwraps_the_single_column() -> None:
    result = FakeResult([7])
    assert SQLResultFormats.one_or_none_scalar(result) == 7  # type: ignore[arg-type]
    assert result.used == ["scalar_one_or_none"]


def test_all_scalar_sequence_unwraps_every_row() -> None:
    result = FakeResult([1, 2])
    assert SQLResultFormats.all_scalar_sequence(result) == [1, 2]  # type: ignore[arg-type]
    assert result.used == ["scalars", "all"]


def test_all_scalar_list_hands_back_a_real_list() -> None:
    assert SQLResultFormats.all_scalar_list(FakeResult((1, 2))) == [1, 2]  # type: ignore[arg-type]


def test_all_returns_the_rows_without_unwrapping_them() -> None:
    result = FakeResult([("a", 1)])
    assert SQLResultFormats.all(result) == [("a", 1)]  # type: ignore[arg-type]
    assert "scalars" not in result.used


def test_all_mappings_goes_through_the_mapping_accessor() -> None:
    result = FakeResult([{"a": 1}])
    assert SQLResultFormats.all_mappings(result) == [{"a": 1}]  # type: ignore[arg-type]
    assert result.used == ["mappings", "all"]


def router_package(modules: dict[str, types.ModuleType]) -> types.ModuleType:
    """A throwaway package whose submodules are already importable."""
    name = "fake_routers_pkg"
    package = types.ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = package
    for submodule_name, module in modules.items():
        sys.modules[f"{name}.{submodule_name}"] = module
    return package


@pytest.fixture
def cleanup_modules() -> Any:
    registered: list[str] = []
    yield registered.append
    for name in list(sys.modules):
        if name.startswith("fake_routers_pkg"):
            del sys.modules[name]


def test_every_router_module_in_the_package_is_mounted(
    monkeypatch: pytest.MonkeyPatch, cleanup_modules: Any
) -> None:
    first = types.ModuleType("first")
    first.router = APIRouter(prefix="/first")  # type: ignore[attr-defined]
    first.router.add_api_route("/ping", lambda: {"ok": True})  # type: ignore[attr-defined]
    second = types.ModuleType("second")
    second.router = APIRouter(prefix="/second")  # type: ignore[attr-defined]
    second.router.add_api_route("/ping", lambda: {"ok": True})  # type: ignore[attr-defined]

    package = router_package({"first": first, "second": second})
    monkeypatch.setattr(
        "pkgutil.iter_modules",
        lambda path: [(None, "first", False), (None, "second", False)],
    )

    app = FastAPI()
    init_routers(app, package, prefix="/api")
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/first/ping" in paths
    assert "/api/second/ping" in paths


def test_a_subpackage_is_not_treated_as_a_router_module(
    monkeypatch: pytest.MonkeyPatch, cleanup_modules: Any
) -> None:
    package = router_package({})
    monkeypatch.setattr("pkgutil.iter_modules", lambda path: [(None, "nested", True)])
    app = FastAPI()
    before = len(app.routes)
    init_routers(app, package)
    assert len(app.routes) == before


def test_a_module_with_no_router_is_skipped_rather_than_failing_startup(
    monkeypatch: pytest.MonkeyPatch, cleanup_modules: Any
) -> None:
    good = types.ModuleType("good")
    good.router = APIRouter(prefix="/good")  # type: ignore[attr-defined]
    good.router.add_api_route("/ping", lambda: {"ok": True})  # type: ignore[attr-defined]
    package = router_package({"good": good, "bad": types.ModuleType("bad")})
    monkeypatch.setattr(
        "pkgutil.iter_modules",
        lambda path: [(None, "bad", False), (None, "good", False)],
    )

    app = FastAPI()
    init_routers(app, package)
    assert "/good/ping" in {route.path for route in app.routes}  # type: ignore[attr-defined]


def test_an_app_path_resolves_to_the_named_attribute() -> None:
    assert _import_app("base.tools.webserver:run_webserver").__name__ == "run_webserver"


@pytest.mark.parametrize("path", ["base.tools.webserver", "base.tools.webserver:", ""])
def test_an_app_path_without_both_halves_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="Expected 'package.module:attribute'"):
        _import_app(path)


def test_an_app_path_with_no_module_half_fails_on_the_import_instead() -> None:
    """Pinned as it stands: the guard checks the separator, not the module name."""
    with pytest.raises(ValueError, match="Empty module name"):
        _import_app(":run_webserver")


def test_an_app_path_naming_a_missing_attribute_is_refused() -> None:
    with pytest.raises(AttributeError):
        _import_app("base.tools.webserver:not_a_real_attribute")


def test_migrations_refuse_settings_that_cannot_reach_a_database(
    make_settings: Any,
) -> None:
    from sqlalchemy import MetaData

    settings = make_settings()
    with pytest.raises(TypeError, match="compose SQLMixin"):
        run_migrations(MetaData(), settings)
