"""Who gets to see a traceback.

The gate used to read `environment != "production"`, which disclosed for
`prod`, `Production`, `staging` and an unset variable alike.
"""

from typing import Any

import pydantic
import pytest

from base.models.definitions.environment import DEBUG_ENVIRONMENTS, Environment
from base.tools.exception_handlers import AppExceptionHandler


@pytest.mark.parametrize("environment", ["local", "development"])
def test_a_developer_environment_may_see_the_detail(
    make_settings: Any, environment: str
) -> None:
    make_settings(environment=environment)
    assert AppExceptionHandler._is_debug() is True


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_a_deployed_environment_sees_nothing(
    make_settings: Any, environment: str
) -> None:
    make_settings(environment=environment)
    assert AppExceptionHandler._is_debug() is False


def test_the_default_environment_is_a_developer_one(make_settings: Any) -> None:
    """Unset means a laptop, which is the only safe thing it can mean."""
    assert make_settings().environment is Environment.LOCAL


@pytest.mark.parametrize(
    "environment", ["prod", "Production", "PRODUCTION", "prd", "uat", ""]
)
def test_a_name_the_library_does_not_know_is_refused_at_startup(
    make_settings: Any, environment: str
) -> None:
    """A misspelling fails loudly instead of picking an environment by accident.

    `prod` and `Production` both used to mean "not production", and therefore
    "show the caller everything".
    """
    with pytest.raises(pydantic.ValidationError):
        make_settings(environment=environment)


def test_the_gate_names_what_may_see_rather_than_what_may_not() -> None:
    """An allowlist is what makes an unforeseen value fail closed."""
    assert DEBUG_ENVIRONMENTS == frozenset(
        {Environment.LOCAL, Environment.DEVELOPMENT}
    )
    assert Environment.PRODUCTION not in DEBUG_ENVIRONMENTS
    assert Environment.STAGING not in DEBUG_ENVIRONMENTS


def test_everything_that_reads_the_environment_is_typed_as_the_enum() -> None:
    """A `str` here lets a typo through; the enum is what closes the set."""
    import inspect

    from base.app import _resolve_cors_origins
    from base.services.configuration.protocols import SupportsServerConfig
    from base.services.logging.config import logging_config
    from base.tools.webserver import run_webserver

    signatures = {
        "_resolve_cors_origins": _resolve_cors_origins,
        "logging_config": logging_config,
        "run_webserver": run_webserver,
    }
    for name, function in signatures.items():
        annotation = inspect.signature(function).parameters["environment"].annotation
        assert "Environment" in str(annotation), f"{name} takes {annotation}"

    assert "Environment" in str(
        inspect.get_annotations(SupportsServerConfig)["environment"]
    )
