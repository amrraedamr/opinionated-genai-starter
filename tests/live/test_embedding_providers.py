"""What only a real embedding provider can answer.

The unit suite proves what the adapter *builds*. These prove what the service
does with it. Two things here are only observable against a real provider: that
a deployment returns the width it was configured for, which a vector store
commits to before the first embedding exists, and that the seam's own batching
hands vectors back in the order they went in. The abstraction warns that an
adapter reimplementing batching could silently reorder; nothing but a live call
can show that it does not.

Every test costs money, so the texts are a few tokens and the model is the
cheap tier.
"""

from __future__ import annotations

import math

import pytest

from base.models.definitions.model import ProviderType
from base.models.view_models.model import EmbeddingConfigurations
from base.services.configuration.mixins import (
    AzureOpenAIEmbeddingSettingsMixin,
    OpenAIEmbeddingSettingsMixin,
)
from base.services.text_embedding.openai import OpenAIEmbeddingService
from base.services.usage import UsageTracker
from tests.conftest import MakeSettings, credential

pytestmark = pytest.mark.live

SMALL_MODEL = "azure.text-embedding-3-small"
SMALL_WIDTH = 1536
LARGE_MODEL = "azure.text-embedding-3-large"
LARGE_WIDTH = 3072


def configure_all(make_settings: MakeSettings) -> None:
    """One gateway and one key serve both embedding surfaces here."""
    make_settings(OpenAIEmbeddingSettingsMixin, AzureOpenAIEmbeddingSettingsMixin)


def config_for(
    model: str,
    *,
    provider: ProviderType = ProviderType.OPENAI,
    dimensions: int | None = None,
    batch_size: int | None = None,
) -> EmbeddingConfigurations:
    """The shared gateway, pointed at one embedding deployment."""
    return EmbeddingConfigurations(
        api_host=credential("OPENAI_EMBEDDING_ENDPOINT"),
        api_key=credential("OPENAI_EMBEDDING_API_KEY"),
        model_name=model,
        api_type=provider,
        dimensions=dimensions,
        batch_size=batch_size,
    )


def service_for(
    model: str,
    *,
    provider: ProviderType = ProviderType.OPENAI,
    dimensions: int | None = None,
    batch_size: int | None = None,
) -> OpenAIEmbeddingService:
    return OpenAIEmbeddingService(
        config_for(
            model, provider=provider, dimensions=dimensions, batch_size=batch_size
        )
    )


def cosine(left: list[float], right: list[float]) -> float:
    """Closeness of two vectors, 1.0 being identical."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


async def test_a_single_text_comes_back_as_one_vector(
    make_settings: MakeSettings,
) -> None:
    """The narrow path a query takes, end to end."""
    configure_all(make_settings)
    vector = service_for(SMALL_MODEL, dimensions=SMALL_WIDTH).embed_query("hello")
    assert len(vector) == SMALL_WIDTH
    assert any(value != 0.0 for value in vector), "an all-zero vector is not an answer"


async def test_both_embedding_surfaces_reach_the_gateway(
    make_settings: MakeSettings,
) -> None:
    """OpenAI and Azure OpenAI are different clients behind the same seam."""
    configure_all(make_settings)
    for provider in (ProviderType.OPENAI, ProviderType.AZURE_OPENAI):
        service = service_for(SMALL_MODEL, provider=provider, dimensions=SMALL_WIDTH)
        vector = await service.aembed_query("hello")
        assert len(vector) == SMALL_WIDTH, f"{provider} returned the wrong width"


async def test_the_configured_width_is_what_arrives(
    make_settings: MakeSettings,
) -> None:
    """A vector store creates its collection from this before any text exists."""
    configure_all(make_settings)
    for model, width in ((SMALL_MODEL, SMALL_WIDTH), (LARGE_MODEL, LARGE_WIDTH)):
        service = service_for(model, dimensions=width)
        assert service.dimensions == width
        assert len(await service.aembed_query("hello")) == width, model


async def test_a_reduced_width_is_honoured(make_settings: MakeSettings) -> None:
    """Asking a wide model for narrow vectors is how storage cost is cut."""
    configure_all(make_settings)
    narrow = 256
    service = service_for(LARGE_MODEL, dimensions=narrow)
    assert len(await service.aembed_query("hello")) == narrow


async def test_a_split_batch_keeps_its_order(make_settings: MakeSettings) -> None:
    """The seam splits; the caller still gets one vector per input, in order.

    Each text is embedded alone and then as part of an oversized batch, and the
    two are compared position by position. A reordering survives every other
    assertion in this file, so it is pinned directly.
    """
    configure_all(make_settings)
    texts = ["a cat sleeps", "the stock market fell", "bread is baking", "rain again"]
    one_at_a_time = [
        service_for(SMALL_MODEL, dimensions=SMALL_WIDTH).embed_query(text)
        for text in texts
    ]
    batched = service_for(
        SMALL_MODEL, dimensions=SMALL_WIDTH, batch_size=2
    ).embed_documents(texts)

    assert len(batched) == len(texts)
    for index, (alone, together) in enumerate(zip(one_at_a_time, batched, strict=True)):
        assert cosine(alone, together) > 0.99, f"position {index} is not the same text"


async def test_each_batch_is_recorded_once(make_settings: MakeSettings) -> None:
    """Every batch is a billable request, so every batch is accounted for."""
    configure_all(make_settings)
    tracker = UsageTracker()
    tracker.reset()

    service = service_for(SMALL_MODEL, dimensions=SMALL_WIDTH, batch_size=2)
    await service.aembed_documents(["one", "two", "three", "four", "five"])

    records = [r for r in tracker.records if r.model == SMALL_MODEL]
    assert len(records) == 3, "five texts at two per request is three requests"
    assert all(r.input_tokens > 0 for r in records), "a billed call counted no input"
    assert all(r.error is None for r in records)


async def test_meaning_survives_the_round_trip(make_settings: MakeSettings) -> None:
    """Related text lands closer than unrelated text, which is the whole point."""
    configure_all(make_settings)
    service = service_for(SMALL_MODEL, dimensions=SMALL_WIDTH)
    dog, puppy, tax = await service.aembed_documents(
        ["a dog barks", "a puppy yaps", "quarterly tax filing deadlines"]
    )
    assert cosine(dog, puppy) > cosine(dog, tax)


async def test_both_counterparts_agree(make_settings: MakeSettings) -> None:
    """The async path is the same request, so it is the same answer."""
    configure_all(make_settings)
    service = service_for(SMALL_MODEL, dimensions=SMALL_WIDTH)
    assert (
        cosine(service.embed_query("hello"), await service.aembed_query("hello")) > 0.99
    )


async def test_unicode_survives_the_round_trip(make_settings: MakeSettings) -> None:
    """Non-Latin text must reach the provider intact rather than mangled."""
    configure_all(make_settings)
    service = service_for(SMALL_MODEL, dimensions=SMALL_WIDTH)
    arabic, same, other = await service.aembed_documents(
        ["القطة تنام على السرير", "القطة تنام على السرير", "اجتماع الميزانية السنوية"]
    )
    assert cosine(arabic, same) > 0.99, "identical text embedded differently"
    assert cosine(arabic, same) > cosine(arabic, other)


async def test_an_empty_list_never_reaches_the_provider(
    make_settings: MakeSettings,
) -> None:
    """Nothing to embed is not a billable request."""
    configure_all(make_settings)
    tracker = UsageTracker()
    tracker.reset()
    assert (
        await service_for(SMALL_MODEL, dimensions=SMALL_WIDTH).aembed_documents([])
        == []
    )
    assert tracker.records == []
