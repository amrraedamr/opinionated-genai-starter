"""Every enum in `models/definitions/`: its members, values and comparisons.

These are the vocabularies adapters translate into, so a renamed member or a
changed wire value breaks every provider at once. `StrEnum` members are also
compared against bare strings throughout the codebase, which only works
because they *are* strings — pinned here rather than assumed.
"""

from enum import Enum, StrEnum

import pytest

from base.models.definitions.backends import BackendSetting
from base.models.definitions.background_tasks import BackgroundTaskBackend, TaskStatus
from base.models.definitions.broker import BrokerBackend, installed_brokers
from base.models.definitions.cache import CacheChannels, CacheKeys
from base.models.definitions.database import Locale, SortingOrder, SQLBackend
from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.definitions.exceptions import ExceptionCategory
from base.models.definitions.health import ProbeStatus
from base.models.definitions.message import MessageRole
from base.models.definitions.messaging import InboxState
from base.models.definitions.model import (
    CachePersistence,
    FinishReason,
    ProviderType,
    ReasoningEffort,
)
from base.models.definitions.tokens import TokenType
from base.models.definitions.vector_search import (
    VECTOR_UPSERT_BATCH_SIZE,
    DistanceMetric,
)

STR_ENUMS: list[type[StrEnum]] = [
    BackendSetting,
    BackgroundTaskBackend,
    BrokerBackend,
    CachePersistence,
    DistanceMetric,
    ExceptionCategory,
    FailureKind,
    FinishReason,
    InboxState,
    Locale,
    MessageRole,
    ProbeStatus,
    ProviderType,
    ReasoningEffort,
    SQLBackend,
    SortingOrder,
    TaskStatus,
    TokenType,
]


@pytest.mark.parametrize("enum_cls", STR_ENUMS, ids=lambda cls: cls.__name__)
def test_every_str_enum_member_equals_its_own_string_value(
    enum_cls: type[StrEnum],
) -> None:
    for member in enum_cls:
        assert member == member.value
        assert isinstance(member, str)
        assert f"{member}" == member.value


@pytest.mark.parametrize("enum_cls", STR_ENUMS, ids=lambda cls: cls.__name__)
def test_every_str_enum_is_constructible_from_its_wire_value(
    enum_cls: type[StrEnum],
) -> None:
    for member in enum_cls:
        assert enum_cls(member.value) is member


@pytest.mark.parametrize("enum_cls", STR_ENUMS, ids=lambda cls: cls.__name__)
def test_no_str_enum_carries_two_members_with_the_same_value(
    enum_cls: type[StrEnum],
) -> None:
    values = [member.value for member in enum_cls]
    assert len(values) == len(set(values))


def test_provider_type_names_every_backend_this_repository_speaks_to() -> None:
    assert {member.value for member in ProviderType} == {
        "azure-openai",
        "openai",
        "anthropic",
        "gemini",
        "ollama",
        "vllm",
        "groq",
        "together",
        "litellm",
    }


def test_reasoning_effort_runs_from_none_to_max() -> None:
    assert [member.value for member in ReasoningEffort] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


def test_cache_persistence_offers_exactly_two_levels() -> None:
    assert [member.value for member in CachePersistence] == ["short", "long"]


def test_finish_reason_covers_every_way_a_model_stops() -> None:
    assert {member.value for member in FinishReason} == {
        "stop",
        "tool_calls",
        "length",
        "content_filter",
        "other",
    }


def test_failure_kind_names_every_classification_a_handler_can_reach() -> None:
    assert {member.value for member in FailureKind} == {
        "rate_limit",
        "server",
        "timeout",
        "connection",
        "bad_request",
        "auth",
        "not_found",
        "conflict",
        "unknown",
    }


def test_only_failures_describing_the_moment_are_worth_retrying() -> None:
    assert RETRYABLE_FAILURES == frozenset(
        {
            FailureKind.RATE_LIMIT,
            FailureKind.SERVER,
            FailureKind.TIMEOUT,
            FailureKind.CONNECTION,
        }
    )


def test_a_failure_describing_the_request_itself_is_not_retryable() -> None:
    for kind in (
        FailureKind.BAD_REQUEST,
        FailureKind.AUTH,
        FailureKind.NOT_FOUND,
        FailureKind.CONFLICT,
        FailureKind.UNKNOWN,
    ):
        assert kind not in RETRYABLE_FAILURES


def test_message_role_names_the_five_roles_a_conversation_uses() -> None:
    assert {member.value for member in MessageRole} == {
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
    }


def test_inbox_state_tracks_one_consumed_event() -> None:
    assert {member.value for member in InboxState} == {
        "processing",
        "processed",
        "failed",
    }


def test_broker_backend_offers_in_process_and_redis() -> None:
    assert {member.value for member in BrokerBackend} == {"memory", "redis"}


def test_only_the_in_memory_broker_is_not_distributed() -> None:
    assert BrokerBackend.MEMORY.is_distributed is False
    assert BrokerBackend.REDIS.is_distributed is True


def test_the_in_memory_broker_needs_no_extra_and_is_always_installed() -> None:
    assert BrokerBackend.MEMORY.extra is None
    assert BrokerBackend.MEMORY.distribution is None
    assert BrokerBackend.MEMORY.required_imports == ()
    assert BrokerBackend.MEMORY.missing_imports == ()
    assert BrokerBackend.MEMORY.is_installed is True


def test_the_redis_broker_names_the_extra_that_installs_it() -> None:
    assert BrokerBackend.REDIS.extra == "redis"
    assert BrokerBackend.REDIS.distribution == "redis"
    assert BrokerBackend.REDIS.required_imports == ("redis",)


def test_installed_brokers_reports_only_what_this_build_can_run() -> None:
    installed = installed_brokers()
    assert BrokerBackend.MEMORY in installed
    assert all(backend.is_installed for backend in installed)


def test_cache_keys_and_channels_are_plain_enums_not_string_enums() -> None:
    """They are compared by member, never passed to a client as a bare string."""
    assert not issubclass(CacheKeys, StrEnum)
    assert not issubclass(CacheChannels, StrEnum)
    assert issubclass(CacheKeys, Enum)
    assert CacheKeys.TEST != "test_cache_key"
    assert CacheKeys.TEST.value == "test_cache_key"


def test_cache_channels_names_the_channels_this_repository_publishes_on() -> None:
    assert {member.value for member in CacheChannels} == {
        "test_channel",
        "keyvault_reload",
        "logs",
        "prompt_reload",
    }


def test_a_channel_is_namespaced_per_application() -> None:
    assert CacheChannels.LOGS.for_app("chat-api") == "chat-api:logs"
    assert CacheChannels.PROMPT_RELOAD.for_app("") == ":prompt_reload"


def test_locale_covers_the_two_languages_plus_an_escape_hatch() -> None:
    assert {member.value for member in Locale} == {"en", "ar", "other"}


def test_vector_search_pins_a_batch_size_both_backends_accept() -> None:
    assert VECTOR_UPSERT_BATCH_SIZE == 100
    assert {member.value for member in DistanceMetric} == {
        "cosine",
        "dot",
        "euclidean",
    }


def test_backend_setting_names_the_variables_that_select_a_backend() -> None:
    assert {member.value for member in BackendSetting} == {"SQL_BACKEND", "BROKER"}


def test_token_type_names_the_two_tokens_this_library_mints() -> None:
    assert {member.value for member in TokenType} == {"access", "refresh"}
