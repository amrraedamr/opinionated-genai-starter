"""`locale.py`: which language a piece of text is, by counting letters.

Routing, prompt selection and stored message metadata all read this, and it
answers for a single message rather than a conversation — so a question in
English about an Arabic document is English. What matters is the tie, the
fallbacks, and the characters that belong to neither alphabet.
"""

import re

import pytest

from base.models.definitions.database import Locale
from base.tools.locale import _determine_locale, get_text_locale

ENGLISH = re.compile(r"[a-zA-Z]")
ARABIC = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


@pytest.mark.parametrize("text", ["hello", "Capital of France?", "a"])
def test_latin_text_reads_as_english(text: str) -> None:
    assert get_text_locale(text) == Locale.EN


@pytest.mark.parametrize("text", ["مرحبا", "ما هي عاصمة فرنسا؟", "ا"])
def test_arabic_text_reads_as_arabic(text: str) -> None:
    assert get_text_locale(text) == Locale.AR


def test_the_alphabet_with_more_letters_wins_a_mixed_message() -> None:
    assert get_text_locale("مرحبا hi") == Locale.AR
    assert get_text_locale("hello مر") == Locale.EN


def test_a_tie_is_broken_towards_english() -> None:
    assert get_text_locale("ab مر") == Locale.EN


@pytest.mark.parametrize("text", ["", "123", "   ", "!@#", "🙂", "٢٠٢٦-٠١"])
def test_text_with_no_letters_falls_back(text: str) -> None:
    """Arabic-Indic digits are inside the Arabic block, so `٢٠٢٦` is not neutral."""
    expected = Locale.AR if any(ARABIC.findall(text)) else Locale.EN
    assert get_text_locale(text) == expected


def test_a_message_of_digits_alone_reads_the_previous_message_instead() -> None:
    assert get_text_locale("42", fallback_text="ما رأيك؟") == Locale.AR
    assert get_text_locale("42", fallback_text="what do you think?") == Locale.EN


def test_the_fallback_is_only_consulted_when_the_text_decides_nothing() -> None:
    assert get_text_locale("hello", fallback_text="مرحبا") == Locale.EN


def test_a_fallback_with_no_letters_either_leaves_the_default_to_decide() -> None:
    assert get_text_locale("42", fallback_text="99", default_fallback=Locale.AR) == (
        Locale.AR
    )


@pytest.mark.parametrize("fallback", [None, "", "   "])
def test_an_absent_fallback_is_skipped_rather_than_examined(
    fallback: str | None,
) -> None:
    assert get_text_locale("42", fallback, Locale.AR) == Locale.AR


def test_english_is_the_answer_when_nothing_at_all_decides() -> None:
    assert get_text_locale("42") == Locale.EN
    assert get_text_locale("", None, None) == Locale.EN


def test_a_default_fallback_of_other_is_honoured() -> None:
    assert get_text_locale("42", default_fallback=Locale.OTHER) == Locale.OTHER


def test_arabic_presentation_forms_still_read_as_arabic() -> None:
    assert get_text_locale("\ufedf\ufe8e") == Locale.AR
    assert get_text_locale("\ufb51") == Locale.AR


def test_an_arabic_supplement_letter_reads_as_arabic() -> None:
    assert get_text_locale("\u0750\u08a0") == Locale.AR


def test_a_latin_letter_with_a_diacritic_counts_as_neither() -> None:
    """`é` is outside `[a-zA-Z]`, so a word spelt with it decides nothing."""
    assert get_text_locale("éé", default_fallback=Locale.OTHER) == Locale.OTHER


def test_counting_reports_nothing_when_the_text_holds_no_letters() -> None:
    assert _determine_locale(ENGLISH, ARABIC, "123 !?") is None


def test_counting_answers_the_alphabet_that_appears_most() -> None:
    assert _determine_locale(ENGLISH, ARABIC, "abc م") == Locale.EN
    assert _determine_locale(ENGLISH, ARABIC, "a مرح") == Locale.AR
