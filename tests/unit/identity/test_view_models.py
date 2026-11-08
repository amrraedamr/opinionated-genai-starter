"""The request and response shapes identity accepts and returns.

A validator that lets something through is the interesting direction, so the
refusals are listed one per rule. Optional fields matter as much: a client
that serialises an absent name as `null` is the common case, not an attack.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from identity.models.view_models.auth import (
    LoginRequest,
    LoginResponse,
    OAuthExchangeRequest,
    OAuthProvidersResponse,
    RefreshRequest,
    RefreshResponse,
    RegisterRequest,
)
from identity.models.view_models.user import (
    CreateUserResponse,
    DeleteUserResponse,
    EditUserRequest,
    EditUserResponse,
    GetUserResponse,
)

PASSWORD = "Str0ng!pass"


# Registering


def test_a_registration_needs_no_names() -> None:
    request = RegisterRequest(email="person@example.com", password=PASSWORD)
    assert request.first_name is None


def test_a_registration_may_send_its_names_as_null() -> None:
    """A client serialising an absent field as `null` is not a malformed one."""
    request = RegisterRequest(
        email="person@example.com", password=PASSWORD, first_name=None, last_name=None
    )
    assert (request.first_name, request.last_name) == (None, None)


def test_a_registration_keeps_a_hyphenated_name() -> None:
    request = RegisterRequest(
        email="person@example.com", password=PASSWORD, first_name="Mary-Jane"
    )
    assert request.first_name == "Mary-Jane"


def test_a_registration_keeps_an_apostrophe_in_a_name() -> None:
    request = RegisterRequest(
        email="person@example.com", password=PASSWORD, last_name="O'Neill"
    )
    assert request.last_name == "O'Neill"


@pytest.mark.parametrize("name", ["Ada2", "", "Ada Lovelace", "-Ada", "<script>"])
def test_a_name_that_is_not_a_name_is_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="person@example.com", password=PASSWORD, first_name=name)


@pytest.mark.parametrize(
    "password", ["short", "alllowercase1!", "ALLUPPERCASE1!", "NoDigits!!", "NoPunct1a"]
)
def test_a_password_that_breaks_a_rule_is_refused(password: str) -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="person@example.com", password=password)


def test_an_address_that_is_not_an_address_is_refused() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="not-an-address", password=PASSWORD)


# Logging in


def test_a_login_needs_a_password_of_some_length() -> None:
    with pytest.raises(ValidationError):
        LoginRequest(email="person@example.com", password="")


def test_a_login_password_longer_than_any_password_is_refused() -> None:
    with pytest.raises(ValidationError):
        LoginRequest(email="person@example.com", password="x" * 257)


def test_a_login_password_is_not_held_to_the_registration_rules() -> None:
    """An old account's password must still be checkable after the rules change."""
    assert LoginRequest(email="person@example.com", password="weak").password == "weak"


# Tokens in and out


def test_a_refresh_request_refuses_a_token_longer_than_any_token() -> None:
    with pytest.raises(ValidationError):
        RefreshRequest(refresh_token="x" * 8193)


def test_a_token_response_carries_both_halves() -> None:
    body = LoginResponse(access_token="a", refresh_token="r").serialize()
    assert body == {"accessToken": "a", "refreshToken": "r"}


def test_a_refresh_response_looks_the_same_as_a_login_response() -> None:
    assert RefreshResponse(access_token="a", refresh_token="r").serialize() == (
        LoginResponse(access_token="a", refresh_token="r").serialize()
    )


@pytest.mark.parametrize("code", ["", "x" * 2049])
def test_an_exchange_code_of_an_impossible_length_is_refused(code: str) -> None:
    with pytest.raises(ValidationError):
        OAuthExchangeRequest(code=code)


def test_the_provider_list_may_be_empty() -> None:
    assert OAuthProvidersResponse(providers=[]).serialize() == {"providers": []}


# Editing an account


def test_an_edit_may_change_nothing() -> None:
    assert EditUserRequest().serialize(exclude_unset=True) == {}


def test_an_edit_may_send_its_names_as_null() -> None:
    assert EditUserRequest(first_name=None).first_name is None


@pytest.mark.parametrize("name", ["Ada2", "", "Ada Lovelace"])
def test_an_edited_name_that_is_not_a_name_is_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        EditUserRequest(first_name=name)


def test_an_edited_password_is_held_to_the_registration_rules() -> None:
    with pytest.raises(ValidationError):
        EditUserRequest(current_password=PASSWORD, new_password="weak")


def test_a_new_password_without_the_current_one_is_refused() -> None:
    with pytest.raises(ValidationError) as caught:
        EditUserRequest(new_password="Ev3n!stronger")
    assert "together" in str(caught.value)


def test_a_current_password_without_a_new_one_is_refused() -> None:
    with pytest.raises(ValidationError):
        EditUserRequest(current_password=PASSWORD)


def test_the_two_password_fields_are_accepted_together() -> None:
    request = EditUserRequest(current_password=PASSWORD, new_password="Ev3n!stronger")
    assert request.new_password == "Ev3n!stronger"


def test_an_edit_hides_both_password_fields_from_what_it_applies() -> None:
    """The router patches the user from this, and a hash is not a column patch."""
    applied = EditUserRequest(
        first_name="Grace", current_password=PASSWORD, new_password="Ev3n!stronger"
    ).serialize(
        exclude_unset=True,
        exclude_none=True,
        by_alias=False,
        exclude={"current_password", "new_password"},
    )
    assert applied == {"first_name": "Grace"}


# Responses


def test_a_created_user_is_announced_by_id_alone() -> None:
    user_id = uuid.uuid4()
    assert CreateUserResponse(user_id=user_id).serialize() == {"userId": str(user_id)}


def test_a_created_user_response_needs_a_real_id() -> None:
    with pytest.raises(ValidationError):
        CreateUserResponse(user_id="not-a-uuid")


def test_a_profile_is_returned_in_the_casing_a_client_reads() -> None:
    user_id = uuid.uuid4()
    body = GetUserResponse(
        user_id=user_id, email="person@example.com", first_name="Ada"
    ).serialize()
    assert body == {
        "userId": str(user_id),
        "email": "person@example.com",
        "firstName": "Ada",
        "lastName": None,
    }


def test_a_profile_needs_an_address() -> None:
    with pytest.raises(ValidationError):
        GetUserResponse(user_id=uuid.uuid4())


def test_an_edit_and_a_delete_answer_with_nothing_to_say() -> None:
    assert EditUserResponse().serialize() == {}
    assert DeleteUserResponse().serialize() == {}


def test_an_edit_may_send_its_new_password_as_null() -> None:
    request = EditUserRequest(current_password=None, new_password=None)
    assert request.new_password is None
