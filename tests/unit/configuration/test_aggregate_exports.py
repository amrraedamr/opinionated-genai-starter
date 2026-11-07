"""The two aggregate `__init__.py` files are the public surface of this area.

A name in `__all__` that does not resolve, or an import the list forgot, is
only noticed by whoever imports it next. Walking both lists here means a
half-finished rename fails at test time instead.
"""

import importlib
import pkgutil
from types import ModuleType
from typing import Protocol

import pytest
from pydantic import BaseModel as PydanticBaseModel

from base.services.configuration import mixins as mixins_package
from base.services.configuration import protocols as protocols_package

AGGREGATES: tuple[ModuleType, ...] = (mixins_package, protocols_package)


@pytest.mark.parametrize("package", AGGREGATES, ids=lambda m: m.__name__)
def test_all_is_sorted(package: ModuleType) -> None:
    assert list(package.__all__) == sorted(package.__all__)


@pytest.mark.parametrize("package", AGGREGATES, ids=lambda m: m.__name__)
def test_all_has_no_duplicates(package: ModuleType) -> None:
    assert len(package.__all__) == len(set(package.__all__))


@pytest.mark.parametrize("package", AGGREGATES, ids=lambda m: m.__name__)
def test_every_exported_name_resolves(package: ModuleType) -> None:
    unresolved = [name for name in package.__all__ if not hasattr(package, name)]
    assert unresolved == []


@pytest.mark.parametrize("package", AGGREGATES, ids=lambda m: m.__name__)
def test_every_exported_name_is_a_class(package: ModuleType) -> None:
    assert [
        name for name in package.__all__ if not isinstance(getattr(package, name), type)
    ] == []


def test_every_mixin_discoverable_in_the_package_is_exported() -> None:
    assert _declared_classes(mixins_package, _is_settings_mixin) == set(
        mixins_package.__all__
    )


def test_every_protocol_discoverable_in_the_package_is_exported() -> None:
    assert _declared_classes(protocols_package, _is_settings_protocol) == set(
        protocols_package.__all__
    )


def test_every_mixin_is_importable_from_the_aggregate_and_its_own_module() -> None:
    for name in mixins_package.__all__:
        exported = getattr(mixins_package, name)
        own_module = importlib.import_module(exported.__module__)
        assert getattr(own_module, name) is exported


def test_every_protocol_is_importable_from_the_aggregate_and_its_own_module() -> None:
    for name in protocols_package.__all__:
        exported = getattr(protocols_package, name)
        own_module = importlib.import_module(exported.__module__)
        assert getattr(own_module, name) is exported


def _is_settings_mixin(obj: object) -> bool:
    return (
        isinstance(obj, type)
        and issubclass(obj, PydanticBaseModel)
        and obj.__name__.endswith("SettingsMixin")
    )


def _is_settings_protocol(obj: object) -> bool:
    return (
        isinstance(obj, type)
        and obj.__name__.startswith("Supports")
        and obj.__name__.endswith("Config")
        and Protocol in obj.__bases__
    )


def _declared_classes(package: ModuleType, predicate: object) -> set[str]:
    """Every matching class defined anywhere under a package, by name."""
    found: set[str] = set()
    modules = [package]
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        modules.append(importlib.import_module(info.name))
    for module in modules:
        for name, obj in vars(module).items():
            if predicate(obj) and obj.__module__.startswith(package.__name__):  # type: ignore[operator]
                found.add(name)
    return found
