"""`validators.py`: the two rules the identity service applies to a signup.

Both are reached straight from a pydantic field validator, so whatever they
accept is what lands in the database and whatever they reject is what a
stranger sees as an error message. The interesting half is therefore what they
refuse, and what a password report says when several rules fail at once.
"""

import pytest

from base.tools.validators import validate_name, validate_password


@pytest.mark.parametrize(
    "name",
    ["Amr", "amr", "A", "MaryJane", "Mary-Jane", "O'Brien", "Anne-Marie-Claire"],
)
def test_a_well_formed_name_is_returned_unchanged(name: str) -> None:
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "Amr Ahmed",
        "Amr ",
        " Amr",
        "-Amr",
        "Amr-",
        "'Amr",
        "Amr'",
        "Jean--Luc",
        "Amr123",
        "Amr_Ahmed",
        "Amr.Ahmed",
        "amr@example.com",
    ],
)
def test_a_name_outside_the_rule_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="only letters, hyphens, or apostrophes"):
        validate_name(name)


@pytest.mark.parametrize("name", ["عمرو", "Amré", "Ünal", "Иван", "李"])
def test_a_name_in_another_script_is_refused(name: str) -> None:
    """Pinned as it stands: the rule is ASCII letters, which excludes most names."""
    with pytest.raises(ValueError):
        validate_name(name)


@pytest.mark.parametrize("name", ["Amr\n", "Amr\n\n", "Mary-Jane\n"])
def test_a_trailing_newline_does_not_sneak_a_name_past_the_rule(name: str) -> None:
    """`$` matches before a final newline, so an anchored `match` accepted this."""
    with pytest.raises(ValueError):
        validate_name(name)


@pytest.mark.parametrize(
    "password",
    ["Passw0rd!", "aB3!aB3!", "P@ssword1", "Pässwörd1!", "LongEnough9#AndMore"],
)
def test_a_password_meeting_every_rule_is_returned_unchanged(password: str) -> None:
    assert validate_password(password) == password


def test_a_password_of_exactly_eight_characters_is_long_enough() -> None:
    assert validate_password("Ab3!cdef") == "Ab3!cdef"


@pytest.mark.parametrize(
    ("password", "message"),
    [
        ("Ab3!cde", "at least 8 characters long"),
        ("ab3!cdefg", "at least one uppercase letter"),
        ("AB3!CDEFG", "at least one lowercase letter"),
        ("Abc!defgh", "at least one digit"),
        ("Abc1defgh", "at least one special character"),
    ],
)
def test_each_rule_names_itself_when_it_is_the_one_that_failed(
    password: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_password(password)


def test_every_rule_a_password_breaks_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as error:
        validate_password("")

    assert len(str(error.value).splitlines()) == 5


def test_the_violations_are_reported_one_per_line_in_rule_order() -> None:
    with pytest.raises(ValueError) as error:
        validate_password("abc")

    assert str(error.value).splitlines() == [
        "Password must be at least 8 characters long.",
        "Password must contain at least one uppercase letter.",
        "Password must contain at least one digit.",
        "Password must contain at least one special character.",
    ]


@pytest.mark.parametrize("special", list('!@#$%^&*(),.?":{}|<>'))
def test_each_character_the_rule_calls_special_satisfies_it(special: str) -> None:
    assert validate_password(f"Abcdefg1{special}") is not None


@pytest.mark.parametrize("special", list("-_+=[];'/\\~`"))
def test_a_punctuation_mark_outside_the_list_does_not_count_as_special(
    special: str,
) -> None:
    """Pinned as it stands: `-` and `_` are the ones a user reaches for first."""
    with pytest.raises(ValueError, match="at least one special character"):
        validate_password(f"Abcdefg1{special}")


def test_an_arabic_indic_digit_counts_as_a_digit() -> None:
    """`\\d` is unicode-aware, so a non-ASCII numeral satisfies the rule."""
    assert validate_password("Abcdefg١!") == "Abcdefg١!"


def test_whitespace_inside_a_password_is_allowed() -> None:
    assert validate_password("Ab3! cdef") == "Ab3! cdef"
