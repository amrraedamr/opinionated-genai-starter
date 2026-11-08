"""The token types both halves of authentication share.

`TokenResponse` parses what an OAuth2 token endpoint sends, so it must stay on
plain pydantic: the repository's shared `BaseModel` camelises aliases, and
`accessToken` is not what RFC 6749 puts on the wire. `TokenPair` is this
API's own response and does camelise — the two are pinned side by side.
"""

import pytest
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError

from base.models.definitions.tokens import TokenType
from base.models.view_models import BaseModel as CamelisingBaseModel
from base.services.auth.models import (
    AccessToken,
    Principal,
    RefreshToken,
    TokenPair,
    TokenResponse,
)

WIRE_RESPONSE: dict[str, object] = {
    "access_token": "at",
    "token_type": "Bearer",
    "expires_in": 3599,
    "refresh_token": "rt",
    "id_token": "it",
    "scope": "openid email",
}


def test_token_response_is_plain_pydantic() -> None:
    assert issubclass(TokenResponse, PydanticBaseModel)
    assert not issubclass(TokenResponse, CamelisingBaseModel)
    assert TokenResponse.model_config.get("alias_generator") is None


def test_token_response_parses_the_wire_format_a_provider_sends() -> None:
    parsed = TokenResponse.model_validate(WIRE_RESPONSE)
    assert parsed.access_token == "at"
    assert parsed.token_type == "Bearer"
    assert parsed.expires_in == 3599
    assert parsed.refresh_token == "rt"
    assert parsed.id_token == "it"
    assert parsed.scope == "openid email"


def test_token_response_dumps_the_snake_case_keys_it_parsed() -> None:
    dumped = TokenResponse.model_validate(WIRE_RESPONSE).model_dump()
    assert dumped == WIRE_RESPONSE
    assert "accessToken" not in dumped


def test_token_response_rejects_camel_case_keys() -> None:
    """Proves the shared camelising base would have broken parsing."""
    with pytest.raises(ValidationError):
        TokenResponse.model_validate({"accessToken": "at"})


def test_token_response_needs_only_an_access_token() -> None:
    """A client-credentials reply carries no id token and often no expiry."""
    parsed = TokenResponse.model_validate({"access_token": "at"})
    assert parsed.access_token == "at"
    assert parsed.token_type == "Bearer"
    assert parsed.expires_in is None
    assert parsed.refresh_token is None
    assert parsed.id_token is None
    assert parsed.scope is None


def test_token_response_without_an_access_token_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        TokenResponse.model_validate({"token_type": "Bearer"})
    assert "access_token" in str(caught.value)


def test_token_response_keeps_a_providers_own_token_type() -> None:
    parsed = TokenResponse.model_validate({"access_token": "at", "token_type": "DPoP"})
    assert parsed.token_type == "DPoP"


def test_token_response_coerces_a_string_expires_in() -> None:
    """Some providers send `expires_in` as a JSON string."""
    assert (
        TokenResponse.model_validate(
            {"access_token": "at", "expires_in": "120"}
        ).expires_in
        == 120
    )


def test_token_pair_camelises_because_it_is_this_api_s_own_response() -> None:
    pair = TokenPair(access_token=AccessToken("at"), refresh_token=RefreshToken("rt"))
    assert issubclass(TokenPair, CamelisingBaseModel)
    assert pair.serialize() == {"accessToken": "at", "refreshToken": "rt"}
    assert pair.model_dump() == {"access_token": "at", "refresh_token": "rt"}


def test_token_pair_accepts_either_spelling_on_the_way_in() -> None:
    assert TokenPair.model_validate({"accessToken": "at", "refreshToken": "rt"}) == (
        TokenPair.model_validate({"access_token": "at", "refresh_token": "rt"})
    )


def test_the_token_new_types_erase_to_str_at_runtime() -> None:
    assert AccessToken.__supertype__ is str
    assert RefreshToken.__supertype__ is str
    assert AccessToken is not RefreshToken
    assert AccessToken("x") == "x"


def test_a_token_new_type_does_not_narrow_validation() -> None:
    """The distinction is for a reader and a type checker, not pydantic."""
    pair = TokenPair.model_validate({"access_token": "at", "refresh_token": "rt"})
    assert type(pair.access_token) is str


def test_a_principal_defaults_to_an_anonymous_shape() -> None:
    principal = Principal(subject="s")
    assert principal.user_id is None
    assert principal.email is None
    assert principal.roles == frozenset()
    assert principal.scopes == frozenset()
    assert principal.issuer is None
    assert principal.token_type is None
    assert principal.claims == {}


def test_a_principal_is_hashable_free_of_mutable_role_sets() -> None:
    principal = Principal(subject="s", roles=["admin"], scopes=["read"])
    assert principal.roles == frozenset({"admin"})
    assert isinstance(principal.roles, frozenset)
    assert isinstance(principal.scopes, frozenset)


def test_the_auth_package_re_exports_the_shared_token_types() -> None:
    """They live outside `services.auth` so outbound code avoids the extra."""
    from base.models.view_models import tokens as shared

    assert AccessToken is shared.AccessToken
    assert RefreshToken is shared.RefreshToken
    assert TokenPair is shared.TokenPair
    assert TokenResponse is shared.TokenResponse


# The carrier and the claim


def test_the_carrier_types_are_erased_to_plain_strings_at_runtime() -> None:
    """`AccessToken` names which string you hold; it is not a value to compare."""
    assert AccessToken("abc") == "abc"
    assert type(AccessToken("abc")) is str
    assert AccessToken("abc") == RefreshToken("abc")


def test_the_claim_vocabulary_is_a_separate_runtime_value() -> None:
    """`TokenType` is what goes inside the token, which the carrier cannot say."""
    assert TokenType.ACCESS == "access"
    assert TokenType.ACCESS != TokenType.REFRESH


def test_the_oauth_scheme_is_not_the_claim_vocabulary() -> None:
    """`TokenResponse.token_type` is RFC 6749's scheme, so it stays a plain string."""
    assert TokenResponse(access_token=AccessToken("a")).token_type == "Bearer"
    assert "Bearer" not in set(TokenType)
