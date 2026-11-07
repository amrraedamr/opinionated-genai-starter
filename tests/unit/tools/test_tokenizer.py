"""`TokenCounter`: one tiktoken encoder per model, shared by every caller.

tiktoken loads an encoding file on first use, so the per-model caching is the
point. The fallback matters just as much: a model newer than the installed
tiktoken must still be countable, with an estimate rather than an exception.
"""

from collections.abc import Iterator

import pytest

from base.services.common.meta import purge_all
from base.tools.tokenizer import TokenCounter


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """Counters are keyed singletons, so one test must not seed the next."""
    purge_all()
    yield
    purge_all()


def test_two_callers_asking_for_one_model_share_an_encoder() -> None:
    assert TokenCounter("gpt-4o") is TokenCounter("gpt-4o")


def test_two_models_each_get_their_own_counter() -> None:
    assert TokenCounter("gpt-4o") is not TokenCounter("gpt-4")


def test_an_empty_string_counts_as_no_tokens() -> None:
    assert TokenCounter("gpt-4o").count("") == 0


def test_text_counts_as_more_than_nothing() -> None:
    counter = TokenCounter("gpt-4o")
    assert counter.count("hello world") > 0
    assert counter.count("hello world hello world") > counter.count("hello world")


def test_counting_a_batch_sums_its_parts() -> None:
    counter = TokenCounter("gpt-4o")
    texts = ["hello", "world", ""]
    assert counter.count_batch(texts) == sum(counter.count(t) for t in texts)


def test_counting_an_empty_batch_is_zero() -> None:
    assert TokenCounter("gpt-4o").count_batch([]) == 0


def test_non_latin_text_is_counted_rather_than_rejected() -> None:
    counter = TokenCounter("gpt-4o")
    assert counter.count("مرحبا بالعالم") > 0
    assert counter.count("🎉🎊") > 0
    assert counter.count("日本語のテキスト") > 0


def test_encoding_and_decoding_are_inverses() -> None:
    counter = TokenCounter("gpt-4o")
    text = "hello مرحبا 🎉"
    assert counter.decode(counter.encode(text)) == text


def test_encoding_produces_one_token_id_per_counted_token() -> None:
    counter = TokenCounter("gpt-4o")
    assert len(counter.encode("hello world")) == counter.count("hello world")


def test_encoding_nothing_yields_no_tokens_and_decoding_none_yields_nothing() -> None:
    counter = TokenCounter("gpt-4o")
    assert counter.encode("") == []
    assert counter.decode([]) == ""


def test_an_unknown_model_falls_back_to_the_newest_encoding() -> None:
    counter = TokenCounter("a-model-tiktoken-has-never-heard-of")
    assert counter._encoding.name == TokenCounter.FALLBACK_ENCODING == "o200k_base"
    assert counter.count("hello world") > 0


def test_the_fallback_says_which_model_it_could_not_place(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The console logger owns the root handlers, so the warning lands on stdout."""
    TokenCounter("another-model-nobody-has-mapped")
    written = capsys.readouterr().out
    assert "another-model-nobody-has-mapped" in written
    assert "o200k_base" in written


def test_a_known_model_does_not_warn(capsys: pytest.CaptureFixture[str]) -> None:
    TokenCounter("gpt-4o")
    assert "falling back" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.4-2026-03-15",
        "ft:gpt-4o:acme::abc123",
        "my-azure-deployment-name",
    ],
)
def test_a_snapshot_a_fine_tune_and_a_deployment_name_are_all_countable(
    model: str,
) -> None:
    assert TokenCounter(model).count("hello") > 0


def test_a_counter_remembers_which_model_it_was_built_for() -> None:
    assert TokenCounter("gpt-4o")._model == "gpt-4o"
