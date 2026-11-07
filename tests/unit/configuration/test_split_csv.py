"""`split_csv` is how every list-valued setting reaches code.

Environment variables carry one string, so `OAUTH_PROVIDERS`, `IDP_ALGORITHMS`
and `TRUSTED_HOSTS` are all split at the point of reading. An operator's
trailing comma or stray space must not become an empty provider name.
"""

import pytest

from base.tools.misc import split_csv


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a,b,c", ["a", "b", "c"]),
        ("a", ["a"]),
        ("  a  ,  b  ", ["a", "b"]),
        ("a,,b", ["a", "b"]),
        ("a,   ,b", ["a", "b"]),
        (",a,", ["a"]),
        (",,,", []),
        ("   ", []),
        ("", []),
        (None, []),
    ],
)
def test_blanks_and_spacing_are_dropped(value: str | None, expected: list[str]) -> None:
    assert split_csv(value) == expected


def test_case_is_preserved_by_default() -> None:
    assert split_csv("Google,MICROSOFT") == ["Google", "MICROSOFT"]


def test_lower_normalises_values_compared_against_known_names() -> None:
    assert split_csv(" GOOGLE , Microsoft ", lower=True) == ["google", "microsoft"]


def test_lower_does_not_deduplicate() -> None:
    assert split_csv("google,GOOGLE", lower=True) == ["google", "google"]


def test_order_is_the_order_the_operator_wrote() -> None:
    assert split_csv("github,apple,google") == ["github", "apple", "google"]


def test_inner_spacing_inside_one_item_survives() -> None:
    assert split_csv("name email, openid") == ["name email", "openid"]


def test_the_result_is_a_new_list_each_call() -> None:
    first = split_csv("a,b")
    first.append("c")
    assert split_csv("a,b") == ["a", "b"]


def test_an_empty_setting_falls_back_where_a_caller_supplies_one() -> None:
    """The `split_csv(...) or [default]` idiom `TokenVerifier` relies on."""
    assert (split_csv("") or ["RS256"]) == ["RS256"]
    assert (split_csv("ES256,RS512") or ["RS256"]) == ["ES256", "RS512"]
