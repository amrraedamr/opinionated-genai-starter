from base.tools.errors import MissingExtraError

try:
    import tiktoken
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "tiktoken", feature="TokenCounter") from exc

from base.models.view_models.models import OpenAIModels
from base.services.common.meta import KeyedSingleton
from base.services.logging.console import ConsoleLoggerService


class TokenCounter(metaclass=KeyedSingleton):
    """Tiktoken-backed token counter, keyed per model.

    Wraps tiktoken.encoding_for_model() with a fallback to cl100k_base for
    unknown models. Instances are cached by KeyedSingleton keyed on the
    model argument, so all callers requesting the same model share one
    encoder — important because tiktoken lazily loads encoding data files
    on first use.

    Accepts OpenAIModels enum values for autocomplete and known-good models,
    or plain strings for snapshot versions (e.g. "gpt-5.4-2026-03-15"),
    fine-tuned models (e.g. "ft:gpt-4o:acme::abc123"), and Azure deployment
    names. StrEnum instances are passed directly to tiktoken since they are
    real strings at runtime.

    Token counting is sync because tiktoken is CPU-bound with no I/O.
    """

    FALLBACK_ENCODING: str = "cl100k_base"

    def __init__(self, model: OpenAIModels | str) -> None:
        self._model: str = model
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

        try:
            self._encoding = tiktoken.encoding_for_model(self._model)
        except KeyError:
            self._logger.warning(
                f"[TokenCounter] Unknown model {self._model!r}, "
                f"falling back to {self.FALLBACK_ENCODING} encoding"
            )
            self._encoding = tiktoken.get_encoding(self.FALLBACK_ENCODING)

    def count(self, text: str) -> int:
        """Count tokens in a single string. Returns 0 for empty input."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_batch(self, texts: list[str]) -> int:
        """Count total tokens across multiple strings."""
        return sum(self.count(t) for t in texts)

    def encode(self, text: str) -> list[int]:
        """Tokens for a string, for callers that need to split on them."""
        return self._encoding.encode(text) if text else []

    def decode(self, tokens: list[int]) -> str:
        """Turn tokens back into text. Inverse of `encode`."""
        return self._encoding.decode(tokens) if tokens else ""
