"""`NonInitialize`: a class meant to be used, never instantiated.

Pins that calling such a class hands back the class itself, so its classmethods
keep working and no per-call state is created.
"""

import pytest

from base.services.common.meta import NonInitialize


class Helpers(metaclass=NonInitialize):
    """A utility namespace with no instance state."""

    registry: dict[str, int] = {}

    @classmethod
    def double(cls, value: int) -> int:
        return value * 2


def test_calling_the_class_returns_the_class_rather_than_an_instance() -> None:
    assert Helpers() is Helpers
    assert not isinstance(Helpers(), Helpers)


def test_the_returned_class_still_answers_its_classmethods() -> None:
    assert Helpers().double(4) == 8


def test_every_call_hands_back_the_same_namespace() -> None:
    Helpers().registry["hits"] = 1
    assert Helpers().registry == {"hits": 1}
    Helpers.registry.clear()


def test_constructor_arguments_are_refused() -> None:
    with pytest.raises(TypeError):
        Helpers("unexpected")  # type: ignore[call-arg]


def test_the_password_hashing_service_is_used_without_instantiation() -> None:
    """The one shipped user of `NonInitialize`, so the rule is not theoretical."""
    from base.tools.security import PasswordHashingService

    assert PasswordHashingService() is PasswordHashingService
