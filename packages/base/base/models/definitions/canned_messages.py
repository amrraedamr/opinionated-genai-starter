from enum import Enum

from base.models.definitions.database import Locale
from base.models.view_models.llm_exception import (
    LLMContentFilterError,
    LLMError,
    LLMIncompleteError,
    LLMRateLimitError,
    LLMStreamError,
)


class CannedMessages(Enum):
    """User-facing strings used as the answer when a flow aborts.

    Each value maps :class:`Locale` directly to its localized text.
    Access via ``CannedMessages.X.value[locale]``.

    Python 3.12+ allows non-hashable enum values via ``EnumType``'s
    ``_unhashable_values_`` fallback; we never call
    ``CannedMessages({...})`` so the linear-scan path doesn't matter at
    runtime.
    """

    ERROR_FALLBACK = {
        Locale.EN: "Sorry, something went wrong. Please try again.",
        Locale.AR: "عذرًا، حدث خطأ ما. يرجى المحاولة مرة أخرى.",
    }
    LLM_GENERIC_ERROR = {
        Locale.EN: (
            "We couldn't process your request right now. Please try again shortly."
        ),
        Locale.AR: "تعذّر معالجة طلبك في الوقت الحالي. يرجى المحاولة بعد قليل.",
    }
    LLM_RATE_LIMITED = {
        Locale.EN: "The service is busy. Please try again in a moment.",
        Locale.AR: "الخدمة مشغولة حاليًا. يرجى المحاولة بعد لحظات.",
    }
    LLM_INCOMPLETE = {
        Locale.EN: (
            "Your request was too long for a complete answer. "
            "Please rephrase or narrow the question."
        ),
        Locale.AR: (
            "طلبك أطول من أن نُجيب عنه بالكامل. يرجى إعادة الصياغة أو تضييق السؤال."
        ),
    }
    LLM_STREAM_FAILED = {
        Locale.EN: "The response was interrupted. Please try again.",
        Locale.AR: "انقطع الرد قبل اكتماله. يرجى المحاولة مرة أخرى.",
    }
    JAILBREAK_ATTEMPT = {
        Locale.EN: "Your request was blocked by our safety filters.",
        Locale.AR: "تم حجب طلبك بواسطة مرشحات السلامة لدينا.",
    }
    HATE_SPEECH_FILTERED = {
        Locale.EN: (
            "Your request was blocked because the response would have "
            "contained hate speech."
        ),
        Locale.AR: "تم حجب طلبك لأن الرد كان سيحتوي على خطاب كراهية.",
    }
    VIOLENCE_FILTERED = {
        Locale.EN: (
            "Your request was blocked because the response would have "
            "contained violent content."
        ),
        Locale.AR: "تم حجب طلبك لأن الرد كان سيحتوي على محتوى عنيف.",
    }
    SEXUAL_CONTENT_FILTERED = {
        Locale.EN: (
            "Your request was blocked because the response would have "
            "contained sexual content."
        ),
        Locale.AR: "تم حجب طلبك لأن الرد كان سيحتوي على محتوى جنسي.",
    }
    SELF_HARM_FILTERED = {
        Locale.EN: (
            "Your request was blocked because the response would have "
            "contained self-harm content."
        ),
        Locale.AR: ("تم حجب طلبك لأن الرد كان سيحتوي على محتوى يتعلق بإيذاء النفس."),
    }

    @classmethod
    def from_llm_error(cls, exc: LLMError) -> "CannedMessages":
        """Map an :class:`LLMError` variant to the matching canned
        message. Falls back to :attr:`LLM_GENERIC_ERROR` for unknown
        subclasses."""
        if isinstance(exc, LLMContentFilterError):
            return cls._from_content_filter(exc)
        if isinstance(exc, LLMRateLimitError):
            return cls.LLM_RATE_LIMITED
        if isinstance(exc, LLMIncompleteError):
            return cls.LLM_INCOMPLETE
        if isinstance(exc, LLMStreamError):
            return cls.LLM_STREAM_FAILED
        return cls.LLM_GENERIC_ERROR

    @classmethod
    def _from_content_filter(cls, exc: LLMContentFilterError) -> "CannedMessages":
        """Inspect Azure-style ``content_filter_results`` on
        ``incomplete_details`` and pick the matching category. Falls
        back to :attr:`JAILBREAK_ATTEMPT` when no specific category is
        flagged."""
        details = getattr(exc, "incomplete_details", None)
        results = getattr(details, "content_filter_results", None) or {}
        for category, canned in (
            ("hate", cls.HATE_SPEECH_FILTERED),
            ("violence", cls.VIOLENCE_FILTERED),
            ("sexual", cls.SEXUAL_CONTENT_FILTERED),
            ("self_harm", cls.SELF_HARM_FILTERED),
            ("jailbreak", cls.JAILBREAK_ATTEMPT),
        ):
            entry = results.get(category) if isinstance(results, dict) else None
            if isinstance(entry, dict) and entry.get("filtered"):
                return canned
        return cls.JAILBREAK_ATTEMPT
