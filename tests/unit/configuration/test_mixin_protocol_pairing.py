"""The mixin/protocol naming pairing, walked rather than listed.

Every settings mixin has one protocol describing its read side, named by the
same rule. Walking both `__all__` lists catches a rename that updated only one
side — the failure mode a hand-written list of pairs cannot see.
"""

from base.services.configuration import mixins as mixins_package
from base.services.configuration import protocols as protocols_package
from base.services.configuration.mixins import (
    AzureOpenAIEmbeddingSettingsMixin,
    AzureOpenAILLMSettingsMixin,
    AzureOpenAISettingsMixin,
)
from base.services.configuration.protocols import (
    SupportsAzureOpenAIConfig,
    SupportsAzureOpenAIEmbeddingConfig,
    SupportsAzureOpenAILLMConfig,
    SupportsBrokerConfig,
    SupportsRedisBrokerConfig,
    SupportsRedisConfig,
)

MIXIN_NAMES: tuple[str, ...] = tuple(mixins_package.__all__)
PROTOCOL_NAMES: tuple[str, ...] = tuple(protocols_package.__all__)

PROTOCOLS_WITHOUT_A_MIXIN: frozenset[str] = frozenset(
    {
        # A composition of two contracts a Redis-backed broker reads together.
        # No single mixin declares it; a service composes both.
        "SupportsRedisBrokerConfig",
    }
)

MIXINS_WITHOUT_A_PROTOCOL: frozenset[str] = frozenset()


def _protocol_for(mixin_name: str) -> str:
    """`FooSettingsMixin` -> `SupportsFooConfig`."""
    return f"Supports{mixin_name.removesuffix('SettingsMixin')}Config"


def _mixin_for(protocol_name: str) -> str:
    """`SupportsFooConfig` -> `FooSettingsMixin`."""
    stem = protocol_name.removeprefix("Supports").removesuffix("Config")
    return f"{stem}SettingsMixin"


def test_every_mixin_name_ends_in_settings_mixin() -> None:
    assert [n for n in MIXIN_NAMES if not n.endswith("SettingsMixin")] == []


def test_every_protocol_name_is_supports_something_config() -> None:
    assert [
        n
        for n in PROTOCOL_NAMES
        if not (n.startswith("Supports") and n.endswith("Config"))
    ] == []


def test_every_mixin_has_the_protocol_its_name_implies() -> None:
    missing = sorted(
        _protocol_for(name)
        for name in MIXIN_NAMES
        if name not in MIXINS_WITHOUT_A_PROTOCOL
        and _protocol_for(name) not in PROTOCOL_NAMES
    )
    assert missing == []


def test_every_protocol_has_the_mixin_its_name_implies() -> None:
    missing = sorted(
        _mixin_for(name)
        for name in PROTOCOL_NAMES
        if name not in PROTOCOLS_WITHOUT_A_MIXIN and _mixin_for(name) not in MIXIN_NAMES
    )
    assert missing == []


def test_the_documented_exceptions_are_the_only_unpaired_names() -> None:
    unpaired_protocols = {
        name for name in PROTOCOL_NAMES if _mixin_for(name) not in MIXIN_NAMES
    }
    unpaired_mixins = {
        name for name in MIXIN_NAMES if _protocol_for(name) not in PROTOCOL_NAMES
    }
    assert unpaired_protocols == PROTOCOLS_WITHOUT_A_MIXIN
    assert unpaired_mixins == MIXINS_WITHOUT_A_PROTOCOL


def test_the_composed_redis_broker_contract_unions_its_two_halves() -> None:
    bases = SupportsRedisBrokerConfig.__mro__
    assert SupportsRedisConfig in bases
    assert SupportsBrokerConfig in bases


def test_the_composed_redis_broker_contract_is_still_runtime_checkable() -> None:
    """Inherited, not redeclared — so `settings_for` can `isinstance` it."""
    assert isinstance(object(), SupportsRedisBrokerConfig) is False


def test_the_shared_azure_openai_mixin_spans_generation_and_embedding() -> None:
    """The documented one-mixin-two-capabilities case, pinned both ways."""
    assert issubclass(AzureOpenAILLMSettingsMixin, AzureOpenAISettingsMixin)
    assert issubclass(AzureOpenAIEmbeddingSettingsMixin, AzureOpenAISettingsMixin)
    assert SupportsAzureOpenAIConfig in SupportsAzureOpenAILLMConfig.__mro__
    assert SupportsAzureOpenAIConfig in SupportsAzureOpenAIEmbeddingConfig.__mro__


def test_the_shared_azure_openai_api_version_is_declared_exactly_once() -> None:
    assert "azure_openai_api_version" in AzureOpenAISettingsMixin.model_fields
    assert "azure_openai_api_version" not in vars(AzureOpenAILLMSettingsMixin).get(
        "__annotations__", {}
    )
    assert "azure_openai_api_version" not in vars(
        AzureOpenAIEmbeddingSettingsMixin
    ).get("__annotations__", {})
