"""Every settings mixin: its fields, its defaults, and how a deployment sets them.

A mixin is a deployment contract. Its field names are the environment variable
names upper-cased, its defaults are what a service gets when an operator sets
nothing, and it must stay on plain pydantic: the repository's shared
`BaseModel` camelises aliases, which would rename every key on dump and break
the env-var mapping these tests pin.
"""

from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel as PydanticBaseModel

from base.models.definitions.broker import BrokerBackend
from base.models.definitions.rate_limiting import RateLimitBackend
from base.models.view_models import BaseModel as CamelisingBaseModel
from base.services.configuration import mixins as mixins_package
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    CredentialsSettingsMixin,
    MongoSettingsMixin,
    MySQLSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    SQLSettingsMixin,
)
from tests.conftest import settings_class

ALL_MIXINS: tuple[type[PydanticBaseModel], ...] = tuple(
    getattr(mixins_package, name) for name in mixins_package.__all__
)

DEFAULTS: dict[str, dict[str, Any]] = {
    "AnthropicSettingsMixin": {
        "anthropic_api_key": "",
        "anthropic_llm_endpoint": "",
        "anthropic_llm_model": "",
        "anthropic_max_output_tokens": 4096,
    },
    "AppleOAuthSettingsMixin": {
        "oauth_apple_client_id": "",
        "oauth_apple_client_secret": "",
        "oauth_apple_team_id": "",
        "oauth_apple_key_id": "",
        "oauth_apple_private_key": "",
        "oauth_apple_private_key_path": "",
        "oauth_apple_discovery_url": (
            "https://appleid.apple.com/.well-known/openid-configuration"
        ),
        "oauth_apple_scopes": "name email",
        "oauth_apple_redirect_uri": "",
    },
    "AzureBlobSettingsMixin": {
        "azure_blob_account_url": "",
        "azure_blob_account_key": "",
    },
    "AzureOpenAIEmbeddingSettingsMixin": {
        "azure_openai_api_version": "",
        "azure_openai_embedding_endpoint": "",
        "azure_openai_embedding_deployment_name": "",
        "azure_openai_embedding_api_key": "",
        "azure_openai_embedding_dimensions": 3072,
        "azure_openai_embedding_batch_size": 2048,
    },
    "AzureOpenAILLMSettingsMixin": {
        "azure_openai_api_version": "",
        "azure_openai_llm_endpoint": "",
        "azure_openai_llm_deployment_name": "",
        "azure_openai_llm_api_key": "",
    },
    "AzureOpenAISettingsMixin": {"azure_openai_api_version": ""},
    "AzureSearchSettingsMixin": {
        "azure_search_endpoint": "",
        "azure_search_vector_field": "vector",
        "azure_search_key_field": "id",
        "azure_search_api_key": "",
    },
    "BrokerSettingsMixin": {
        "broker": BrokerBackend.MEMORY,
        "broker_max_attempts": 5,
        "broker_claim_idle_ms": 30_000,
        "broker_retry_backoff_seconds": 0.25,
    },
    "CohereSettingsMixin": {
        "cohere_api_key": "",
        "cohere_rerank_model": "rerank-v4.0-fast",
    },
    "CosmosSettingsMixin": {
        "azure_cosmos_account_uri": "",
        "azure_cosmos_account_key": "",
        "azure_cosmos_database_name": "",
    },
    "CredentialsSettingsMixin": {
        "outbound_token_url": "",
        "outbound_client_id": "",
        "outbound_client_secret": "",
        "outbound_scope": "",
        "outbound_credential_placement": "basic",
        "outbound_token_refresh_skew_seconds": 60,
        "outbound_verify_tls": True,
    },
    "FacebookOAuthSettingsMixin": {
        "oauth_facebook_client_id": "",
        "oauth_facebook_client_secret": "",
        "oauth_facebook_authorize_url": "https://www.facebook.com/v21.0/dialog/oauth",
        "oauth_facebook_token_url": (
            "https://graph.facebook.com/v21.0/oauth/access_token"
        ),
        "oauth_facebook_userinfo_url": (
            "https://graph.facebook.com/me?fields=id,email,first_name,last_name"
        ),
        "oauth_facebook_scopes": "email public_profile",
        "oauth_facebook_redirect_uri": "",
    },
    "GeminiSettingsMixin": {
        "gemini_api_key": "",
        "gemini_llm_endpoint": "",
        "gemini_llm_model": "",
        "gemini_use_vertex": False,
        "gemini_project": "",
        "gemini_location": "global",
    },
    "GitHubOAuthSettingsMixin": {
        "oauth_github_client_id": "",
        "oauth_github_client_secret": "",
        "oauth_github_authorize_url": "https://github.com/login/oauth/authorize",
        "oauth_github_token_url": "https://github.com/login/oauth/access_token",
        "oauth_github_userinfo_url": "https://api.github.com/user",
        "oauth_github_scopes": "read:user user:email",
        "oauth_github_redirect_uri": "",
    },
    "GoogleOAuthSettingsMixin": {
        "oauth_google_client_id": "",
        "oauth_google_client_secret": "",
        "oauth_google_discovery_url": (
            "https://accounts.google.com/.well-known/openid-configuration"
        ),
        "oauth_google_scopes": "openid email profile",
        "oauth_google_redirect_uri": "",
    },
    "GuardrailSettingsMixin": {
        "guardrails_enabled": False,
        "pii_redaction_enabled": True,
        "moderation_enabled": False,
        "pii_extra_patterns": {},
        "moderation_model": "omni-moderation-latest",
        "moderation_threshold": 0.5,
        "moderation_categories": [],
        "injection_block_enabled": True,
        "pii_stream_tail": 48,
    },
    "RealtimeSettingsMixin": {
        "realtime_provider": "openai",
        "realtime_model": "gpt-realtime",
        "realtime_voice": "",
        "realtime_instructions": "",
        "realtime_audio_enabled": True,
    },
    "IdpSettingsMixin": {
        "idp_resource_server_enabled": False,
        "idp_issuer": "",
        "idp_audience": "",
        "idp_jwks_url": "",
        "idp_discovery_url": "",
        "idp_algorithms": "RS256",
        "idp_provider_name": "entra",
        "idp_email_claim": "email",
    },
    "KeyVaultSettingsMixin": {
        "keyvault_url": "",
        "system_keyvault_tag": "appconfig",
        "model_keyvault_tag": "modelconfig",
    },
    "LLMSettingsMixin": {"capture_raw_traces": False},
    "MicrosoftOAuthSettingsMixin": {
        "oauth_microsoft_client_id": "",
        "oauth_microsoft_client_secret": "",
        "oauth_microsoft_discovery_url": (
            "https://login.microsoftonline.com/common/v2.0/"
            ".well-known/openid-configuration"
        ),
        "oauth_microsoft_scopes": "openid email profile",
        "oauth_microsoft_redirect_uri": "",
    },
    "MongoSettingsMixin": {
        "mongodb_username": "",
        "mongodb_password": "",
        "mongodb_base_path": "mongodb",
        "mongodb_host": "localhost",
        "mongodb_database": "",
        "mongodb_params": "",
        "mongodb_min_pool_size": 5,
        "mongodb_max_pool_size": 50,
        "mongodb_connect_timeout_ms": 10_000,
        "mongodb_socket_timeout_ms": 30_000,
        "mongodb_server_selection_timeout_ms": 30_000,
    },
    "MySQLSettingsMixin": {
        "mysql_host": "",
        "mysql_database": "",
        "mysql_username": "",
        "mysql_password": "",
    },
    "OAuthSettingsMixin": {
        "oauth_providers": "",
        "oauth_success_redirect": "http://localhost:3000/auth/callback",
        "oauth_state_ttl_seconds": 600,
        "oauth_exchange_code_ttl_seconds": 60,
    },
    "OpenAIEmbeddingSettingsMixin": {
        "openai_embedding_endpoint": "",
        "openai_embedding_model": "",
        "openai_embedding_api_key": "",
        "openai_embedding_dimensions": 1536,
        "openai_embedding_batch_size": 2048,
    },
    "OpenAILLMSettingsMixin": {
        "openai_llm_endpoint": "",
        "openai_llm_model": "",
        "openai_llm_api_key": "",
    },
    "PostgresSettingsMixin": {
        "pg_host": "",
        "pg_port": "5432",
        "pg_database": "",
        "pg_ssl_root_cert": "",
        "pg_username": "",
        "pg_password": "",
    },
    "PromptSettingsMixin": {"load_prompts_from_db": False},
    "QdrantSettingsMixin": {
        "qdrant_url": "",
        "qdrant_grpc_port": 6334,
        "qdrant_prefer_grpc": True,
        "qdrant_timeout": 30,
        "qdrant_api_key": "",
    },
    "RateLimitSettingsMixin": {
        "rate_limit_backend": RateLimitBackend.MEMORY,
        "rate_limit_requests_per_minute": 300,
        "rate_limit_auth_per_minute": 20,
        "rate_limit_llm_tokens_per_minute": 100_000,
        "rate_limit_ws_connect_per_minute": 60,
        "rate_limit_ws_messages_per_minute": 600,
    },
    "RedisSettingsMixin": {
        "redis_url": "",
        "redis_ssl_cert_reqs": "",
        "redis_max_connections": 50,
        "redis_socket_timeout": 5.0,
        "redis_socket_connect_timeout": 5.0,
        "redis_retry_on_timeout": True,
        "redis_health_check_interval": 30,
        "redis_decode_responses": True,
    },
    "RerankSettingsMixin": {"rerank_overfetch": 4},
    "SQLSettingsMixin": {
        "sql_backend": "sqlite",
        "sqlite_db_path": "database/app.db",
        "db_pool_size": 10,
        "db_max_overflow": 20,
        "db_pool_timeout": 30,
        "db_pool_recycle": 3600,
        "db_ssl_enabled": True,
    },
    "ServerSettingsMixin": {
        "api_prefix": "/api",
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "max_gunicorn_workers": 2,
        "trusted_hosts": "localhost,127.0.0.1,testserver",
        "request_max_bytes": 1_048_576,
        "request_concurrency_limit": 100,
    },
    "TelemetrySettingsMixin": {
        "otel_enabled": False,
        "otel_exporter_endpoint": "http://localhost:4317",
        "otel_service_name": "",
        "otel_sample_ratio": 1.0,
        "otel_capture_content": False,
    },
    "TenancySettingsMixin": {
        "tenancy_enabled": False,
        "tenant_claim": "tid",
        "tenancy_rls_enabled": False,
    },
    "TokenIssuingSettingsMixin": {
        "auth_public_key_path": "keys/public.pem",
        "auth_private_key_path": "keys/private.pem",
        "access_token_expiry_seconds": 900,
        "refresh_token_expiry_seconds": 604800,
    },
    "TokenVerificationSettingsMixin": {"auth_public_key_path": "keys/public.pem"},
}


def _mixin_id(mixin: type) -> str:
    return mixin.__name__


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_mixin_declares_exactly_the_fields_recorded_here(mixin: type) -> None:
    assert set(mixin.model_fields) == set(DEFAULTS[mixin.__name__])


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_mixin_defaults_every_field_so_a_service_can_start(mixin: type) -> None:
    actual = {name: field.default for name, field in mixin.model_fields.items()}
    assert actual == DEFAULTS[mixin.__name__]


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_mixin_default_matches_the_type_its_field_declares(mixin: type) -> None:
    for name, field in mixin.model_fields.items():
        runtime_types = _runtime_types(field.annotation)
        assert isinstance(field.default, runtime_types), (
            f"{mixin.__name__}.{name} defaults to {field.default!r}, "
            f"which is not a {field.annotation}"
        )


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_mixin_reads_each_field_from_the_field_name_upper_cased(
    mixin: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected: dict[str, Any] = {}
    for name, field in mixin.model_fields.items():
        raw, parsed = _environment_sample(field.annotation, field.default)
        monkeypatch.setenv(name.upper(), raw)
        expected[name] = parsed

    settings = settings_class(mixin)(service_name="svc")
    actual = {name: getattr(settings, name) for name in mixin.model_fields}
    assert actual == expected


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_mixin_is_plain_pydantic_not_the_camelising_base_model(mixin: type) -> None:
    assert issubclass(mixin, PydanticBaseModel)
    assert not issubclass(mixin, CamelisingBaseModel)
    assert mixin.model_config.get("alias_generator") is None


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_mixin_never_renames_a_field_on_dump(mixin: type) -> None:
    for name, field in mixin.model_fields.items():
        assert field.alias is None, f"{mixin.__name__}.{name} carries an alias"
        assert field.serialization_alias is None


@pytest.mark.parametrize("mixin", ALL_MIXINS, ids=_mixin_id)
def test_a_validation_alias_still_accepts_the_field_name_upper_cased(
    mixin: type,
) -> None:
    for name, field in mixin.model_fields.items():
        choices = getattr(field.validation_alias, "choices", None)
        if choices is None:
            continue
        assert name.upper() in choices, (
            f"{mixin.__name__}.{name} cannot be set from {name.upper()}"
        )


def test_settings_composed_from_every_mixin_dump_in_snake_case() -> None:
    """A shared camelising base would turn `pg_host` into `pgHost` here."""
    settings = settings_class(*ALL_MIXINS)(service_name="svc")
    dumped = settings.model_dump()
    assert dumped == settings.model_dump(by_alias=True)
    assert [key for key in dumped if key != key.lower()] == []
    for key in ("pg_host", "mongodb_host", "redis_url", "auth_public_key_path"):
        assert key in dumped


def test_every_mixin_composes_into_one_settings_class() -> None:
    settings = settings_class(*ALL_MIXINS)(service_name="svc")
    for mixin in ALL_MIXINS:
        assert isinstance(settings, mixin)


def test_the_literal_backed_fields_reject_a_value_outside_their_choices() -> None:
    assert (
        get_origin(SQLSettingsMixin.model_fields["sql_backend"].annotation) is Literal
    )
    assert get_args(SQLSettingsMixin.model_fields["sql_backend"].annotation) == (
        "sqlite",
        "mysql",
        "postgresql",
    )
    assert get_args(
        CredentialsSettingsMixin.model_fields[
            "outbound_credential_placement"
        ].annotation
    ) == ("basic", "form")


def test_the_broker_field_is_the_broker_backend_enum() -> None:
    assert BrokerSettingsMixin.model_fields["broker"].annotation is BrokerBackend


def test_the_redis_timeouts_are_floats_so_sub_second_values_survive() -> None:
    for name in ("redis_socket_timeout", "redis_socket_connect_timeout"):
        assert RedisSettingsMixin.model_fields[name].annotation is float


def test_a_database_user_can_also_be_set_from_the_vendor_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`POSTGRES_USER` and `MYSQL_USER` are what the official images read."""
    monkeypatch.setenv("POSTGRES_USER", "pg-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg-password")
    monkeypatch.setenv("MYSQL_USER", "my-user")
    monkeypatch.setenv("MONGODB_USER", "mongo-user")

    settings = settings_class(
        PostgresSettingsMixin, MySQLSettingsMixin, MongoSettingsMixin
    )(service_name="svc")
    assert settings.pg_username == "pg-user"
    assert settings.pg_password == "pg-password"
    assert settings.mysql_username == "my-user"
    assert settings.mongodb_username == "mongo-user"


def _runtime_types(annotation: Any) -> tuple[type, ...]:
    """The concrete types a field's annotation permits at runtime."""
    origin = get_origin(annotation)
    if origin is Literal:
        return tuple({type(arg) for arg in get_args(annotation)})
    if origin is Union:
        return tuple(
            concrete for arg in get_args(annotation) for concrete in _runtime_types(arg)
        )
    if annotation is float:
        return (float, int)
    if origin is not None:
        return (origin,)
    assert isinstance(annotation, type)
    return (annotation,)


def _environment_sample(annotation: Any, default: Any) -> tuple[str, Any]:
    """An environment value differing from the default, and what it parses to."""
    origin = get_origin(annotation)
    if origin is Literal:
        choice = next(arg for arg in get_args(annotation) if arg != default)
        return str(choice), choice
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        member = next(m for m in annotation if m != default)
        return str(member.value), member
    if annotation is bool:
        flipped = not default
        return str(flipped).lower(), flipped
    if annotation is int:
        return "4321", 4321
    if annotation is float:
        return "1.25", 1.25
    if origin is dict:
        return '{"probe": "value"}', {"probe": "value"}
    if origin is list:
        return '["probe"]', ["probe"]
    return "set-from-the-environment", "set-from-the-environment"
