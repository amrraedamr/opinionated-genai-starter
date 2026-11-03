import re

from base.models.definitions.database import Locale


def _determine_locale(
    english_pattern: re.Pattern, arabic_pattern: re.Pattern, text_to_check: str
) -> Locale | None:
    """Helper function to determine the locale from the given text."""
    en_count = len(re.findall(english_pattern, text_to_check))
    ar_count = len(re.findall(arabic_pattern, text_to_check))

    # if the text contains only numbers/special characters, return None
    if en_count == 0 and ar_count == 0:
        return None

    # Default to English if counts are equal,
    # or return the language with the higher count
    return Locale.EN if en_count >= ar_count else Locale.AR


def get_text_locale(
    text: str,
    fallback_text: str | None = None,
    default_fallback: Locale | None = None,
) -> Locale:
    """Returns the locale of a text string. Either en or ar."""
    english_pattern = re.compile(r"[a-zA-Z]")
    arabic_pattern = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
    )

    # first check if the text is in English or Arabic
    locale = _determine_locale(english_pattern, arabic_pattern, text)
    if locale is not None:
        return locale

    # second check if the fallback_text is provided. (last message)
    if fallback_text:
        locale = _determine_locale(english_pattern, arabic_pattern, fallback_text)
        if locale is not None:
            return locale

    # if no locale is determined, return the default_fallback if provided.
    if default_fallback:
        return default_fallback

    return Locale.EN  # Default to English if no locale is determined
