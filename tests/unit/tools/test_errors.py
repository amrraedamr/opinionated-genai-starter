"""`tools/errors.py`: the errors that tell a developer what to change.

Each of these is read by a person rather than caught by code, so the message
is the behaviour: it has to name the extra, the mixin or the contract that
would make the failure go away.
"""

from enum import Enum
from http import HTTPStatus

import pytest

from base.models.definitions.errors import AuthErrors, DatabaseErrors, GenericErrors
from base.tools.errors import (
    ErrorsException,
    InvalidSignedPayload,
    MissingExtraError,
    MissingSettingsError,
    MissingSettingsMixinError,
    ServiceHealthCheckError,
    StateDecodeError,
    UnsupportedBackendError,
    raise_error,
)


def test_a_missing_extra_names_both_the_extra_and_the_distribution() -> None:
    error = MissingExtraError("azure", "azure-storage-blob")
    message = str(error)
    assert "'azure'" in message
    assert "azure-storage-blob" in message
    assert "base[azure]" in message
    assert error.extra == "azure"
    assert error.distribution == "azure-storage-blob"


def test_a_missing_extra_names_the_feature_that_needed_it() -> None:
    assert str(
        MissingExtraError("tokenizer", "tiktoken", feature="TokenCounter")
    ).startswith("TokenCounter requires the 'tokenizer' extra")


def test_a_missing_extra_falls_back_to_naming_the_integration() -> None:
    assert str(MissingExtraError("redis", "redis")).startswith(
        "The 'redis' integration requires"
    )


def test_a_missing_extra_is_an_import_error_so_it_catches_as_one() -> None:
    assert isinstance(MissingExtraError("a", "b"), ImportError)


def test_an_unsupported_backend_tells_an_operator_what_this_build_does_support() -> (
    None
):
    error = UnsupportedBackendError(
        "postgresql", "postgres", "asyncpg / psycopg2-binary", ["asyncpg"], ["sqlite"]
    )
    message = str(error)
    assert "SQL_BACKEND='postgresql'" in message
    assert "asyncpg" in message
    assert "This build supports: sqlite" in message
    assert "base[postgres]" in message


def test_an_unsupported_backend_says_so_plainly_when_nothing_is_supported() -> None:
    error = UnsupportedBackendError("mysql", "mysql", "aiomysql", ["aiomysql"], [])
    assert "This build supports: none" in str(error)


def test_an_unsupported_backend_keeps_the_lists_it_was_given() -> None:
    error = UnsupportedBackendError(
        "mysql", "mysql", "aiomysql", ["aiomysql"], ["sqlite"]
    )
    assert error.backend == "mysql"
    assert error.missing == ("aiomysql",)
    assert error.available == ("sqlite",)


def test_an_unsupported_backend_is_still_a_missing_extra() -> None:
    error = UnsupportedBackendError("mysql", "mysql", "aiomysql", [], [])
    assert isinstance(error, MissingExtraError)
    assert isinstance(error, ImportError)


def test_missing_settings_names_the_contract_that_went_unmet() -> None:
    error = MissingSettingsError("ChatSettings", "SupportsSQLConfig")
    assert "ChatSettings does not satisfy SupportsSQLConfig" in str(error)
    assert error.settings_class == "ChatSettings"
    assert error.contract == "SupportsSQLConfig"
    assert isinstance(error, RuntimeError)


def test_a_missing_settings_mixin_names_the_mixin_and_the_setting() -> None:
    error = MissingSettingsMixinError("redis", "RedisSettingsMixin", "BROKER")
    message = str(error)
    assert "BROKER enables 'redis'" in message
    assert "Add RedisSettingsMixin" in message
    assert error.selected == "redis"
    assert error.mixin == "RedisSettingsMixin"
    assert error.setting == "BROKER"


def test_an_invalid_signed_payload_distinguishes_expired_from_tampered() -> None:
    expired = InvalidSignedPayload(expired=True)
    invalid = InvalidSignedPayload(expired=False)
    assert expired.expired is True and str(expired) == "expired"
    assert invalid.expired is False and str(invalid) == "invalid"


def test_an_invalid_signed_payload_insists_the_caller_says_which() -> None:
    with pytest.raises(TypeError):
        InvalidSignedPayload()  # type: ignore[call-arg]


def test_a_state_decode_error_is_a_value_error() -> None:
    assert isinstance(StateDecodeError("bad"), ValueError)


def test_a_health_check_error_carries_only_its_message() -> None:
    assert str(ServiceHealthCheckError("redis unreachable")) == "redis unreachable"


def test_an_errors_exception_built_from_an_enum_reads_its_message() -> None:
    error = ErrorsException(error_enum=AuthErrors.USER_UNAUTHORIZED)
    assert str(error) == "Error: Unauthorized."
    assert error.error_enum is AuthErrors.USER_UNAUTHORIZED
    assert error.message is None


def test_an_errors_exception_built_from_a_message_keeps_its_status() -> None:
    error = ErrorsException(message="nope", status_code=418)
    assert str(error) == "nope"
    assert error.status_code == 418


def test_an_errors_exception_with_nothing_at_all_still_says_something() -> None:
    assert str(ErrorsException()) == "Unknown Error"


def test_raise_error_raises_the_enum_it_was_given() -> None:
    with pytest.raises(ErrorsException) as caught:
        raise_error(DatabaseErrors.RECORD_NOT_FOUND)
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND
    assert "Record not found." in str(caught.value)


def test_raise_error_raises_a_message_when_given_no_enum() -> None:
    with pytest.raises(ErrorsException) as caught:
        raise_error(message="something broke", status_code=500)
    assert caught.value.message == "something broke"
    assert caught.value.status_code == 500
    assert caught.value.error_enum is None


def test_raise_error_prefers_the_enum_over_a_message() -> None:
    with pytest.raises(ErrorsException) as caught:
        raise_error(GenericErrors.UNEXPECTED_ERROR, message="ignored")
    assert caught.value.message is None


def test_raise_error_never_returns() -> None:
    with pytest.raises(ErrorsException):
        raise_error()


@pytest.mark.parametrize(
    "enum_cls", [AuthErrors, DatabaseErrors, GenericErrors], ids=lambda c: c.__name__
)
def test_every_catalogued_error_carries_a_message_and_a_real_status(
    enum_cls: type[Enum],
) -> None:
    for member in enum_cls:
        assert member.value["message"]
        assert isinstance(member.value["status_code"], HTTPStatus)
