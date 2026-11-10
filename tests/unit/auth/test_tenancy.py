"""Which customer a caller acts for, and that nobody acts for one unasked.

Tenancy is opt-in, so the property that matters most is the negative one: a
service that did not turn it on never gets a tenant, whatever a token carries.
An identity provider sends `tid` on every token it issues, so a service reading
it without having asked would start partitioning data it never meant to.
"""

import asyncio
import contextvars
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from base.models.definitions.errors import AuthErrors
from base.models.definitions.tenancy import DEFAULT_TENANT_CLAIM, TenantId
from base.services.auth import (
    TokenIssuer,
    TokenVerifier,
    current_tenant_id,
    tenant_id_dependency,
)
from base.services.auth.bearer import current_principal
from base.services.configuration.mixins import (
    IdpSettingsMixin,
    TenancySettingsMixin,
    TokenIssuingSettingsMixin,
)
from base.tools.errors import ErrorsException
from base.tools.exception_handlers import AppExceptionHandler
from base.tools.tenancy import CallerContext, TenantContext, current_tenant
from tests.conftest import MakeSettings
from tests.unit.auth.conftest import KeyPair

ISSUER = "https://login.example.com/tenant/v2.0"
AUDIENCE = "api://this-service"


def _settings(
    make_settings: MakeSettings, keys: KeyPair, *mixins: type, **values: Any
) -> Any:
    return make_settings(
        TokenIssuingSettingsMixin,
        *mixins,
        auth_private_key_path=str(keys.private_path),
        auth_public_key_path=str(keys.public_path),
        **values,
    )


def _local_token(**claims: Any) -> str:
    return TokenIssuer().issue_access({"user_id": "u-1", **claims})


# The claim


def test_the_default_claim_is_the_one_entra_id_issues() -> None:
    assert DEFAULT_TENANT_CLAIM == "tid"


def test_a_service_that_did_not_opt_in_never_gets_a_tenant(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys)
    assert TokenVerifier().verify(_local_token(tid="acme")).tenant_id is None


def test_composing_the_mixin_without_enabling_it_is_still_off(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin)
    assert TokenVerifier().verify(_local_token(tid="acme")).tenant_id is None


def test_an_opted_in_service_reads_the_tenant_from_a_first_party_token(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    assert TokenVerifier().verify(_local_token(tid="acme")).tenant_id == "acme"


def test_the_claim_name_is_configurable(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(
        make_settings,
        keys,
        TenancySettingsMixin,
        tenancy_enabled=True,
        tenant_claim="org",
    )
    principal = TokenVerifier().verify(_local_token(org="acme", tid="ignored"))
    assert principal.tenant_id == "acme"


def test_a_token_naming_no_tenant_reads_as_none(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    assert TokenVerifier().verify(_local_token()).tenant_id is None


def test_an_empty_tenant_claim_reads_as_none(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    assert TokenVerifier().verify(_local_token(tid="")).tenant_id is None


def test_a_numeric_tenant_claim_is_read_as_text(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    assert TokenVerifier().verify(_local_token(tid=42)).tenant_id == TenantId("42")


# Identity provider tokens


def _idp_token(private_pem: str, **claims: Any) -> str:
    payload: dict[str, Any] = {
        "sub": "idp-subject",
        "email": "person@example.com",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": datetime.now(UTC) + timedelta(minutes=10),
        "iat": datetime.now(UTC),
        **claims,
    }
    return jwt.encode(payload, private_pem, algorithm="RS256")


def _idp_verifier(other_keys: KeyPair) -> TokenVerifier:
    verifier = TokenVerifier()
    verifier._jwks_client = SimpleNamespace(  # type: ignore[assignment]
        get_signing_key_from_jwt=lambda token: SimpleNamespace(
            key=other_keys.public_pem
        )
    )
    return verifier


def _idp_settings(make_settings: MakeSettings, keys: KeyPair, **values: Any) -> Any:
    return _settings(
        make_settings,
        keys,
        IdpSettingsMixin,
        TenancySettingsMixin,
        idp_resource_server_enabled=True,
        idp_issuer=ISSUER,
        idp_audience=AUDIENCE,
        idp_jwks_url="https://login.example.com/keys",
        **values,
    )


def test_an_opted_in_service_reads_the_tenant_from_an_idp_token(
    make_settings: MakeSettings, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys, tenancy_enabled=True)
    token = _idp_token(other_keys.private_pem, tid="contoso")
    assert _idp_verifier(other_keys).verify(token).tenant_id == "contoso"


def test_an_idp_token_carrying_tid_gives_no_tenant_unless_opted_in(
    make_settings: MakeSettings, keys: KeyPair, other_keys: KeyPair
) -> None:
    """Entra ID sends `tid` on every token, so this is the case that matters."""
    _idp_settings(make_settings, keys)
    token = _idp_token(other_keys.private_pem, tid="contoso")
    assert _idp_verifier(other_keys).verify(token).tenant_id is None


# The dependency


def _app() -> FastAPI:
    app = FastAPI()
    AppExceptionHandler().register(app)

    @app.get("/scoped")
    async def scoped(tenant: TenantId = tenant_id_dependency) -> dict[str, str | None]:
        return {"tenant": tenant, "context": TenantContext.get()}

    return app


def test_the_dependency_hands_a_route_the_callers_tenant(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    response = TestClient(_app()).get(
        "/scoped", headers={"Authorization": f"Bearer {_local_token(tid='acme')}"}
    )
    assert response.status_code == 200
    assert response.json()["tenant"] == "acme"


def test_the_dependency_sets_the_tenant_context_for_what_runs_below_it(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    response = TestClient(_app()).get(
        "/scoped", headers={"Authorization": f"Bearer {_local_token(tid='acme')}"}
    )
    assert response.json()["context"] == "acme"


def test_a_token_naming_no_tenant_is_refused_by_the_dependency(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    response = TestClient(_app()).get(
        "/scoped", headers={"Authorization": f"Bearer {_local_token()}"}
    )
    assert response.status_code == 401


def test_a_service_that_did_not_opt_in_is_refused_by_the_dependency(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    """A route asking for a tenant in a service with tenancy off is a mistake."""
    _settings(make_settings, keys)
    response = TestClient(_app()).get(
        "/scoped", headers={"Authorization": f"Bearer {_local_token(tid='acme')}"}
    )
    assert response.status_code == 401


async def test_the_dependency_refuses_with_the_incomplete_token_error(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    _settings(make_settings, keys, TenancySettingsMixin, tenancy_enabled=True)
    principal = TokenVerifier().verify(_local_token())
    with pytest.raises(ErrorsException) as caught:
        await current_tenant_id(principal)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


def test_the_dependency_is_a_single_shared_object() -> None:
    assert isinstance(tenant_id_dependency, type(Depends(current_tenant_id)))
    assert tenant_id_dependency.dependency is current_tenant_id


# The context


def test_nothing_sets_a_tenant_by_default() -> None:
    assert TenantContext.get() is None


def test_a_scope_sets_the_tenant_inside_it_and_restores_it_after() -> None:
    with TenantContext.scope(TenantId("acme")):
        assert TenantContext.get() == "acme"
    assert TenantContext.get() is None


def test_scopes_nest_and_each_restores_its_own_predecessor() -> None:
    with TenantContext.scope(TenantId("outer")):
        with TenantContext.scope(TenantId("inner")):
            assert TenantContext.get() == "inner"
        assert TenantContext.get() == "outer"


def test_a_scope_restores_the_previous_tenant_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError):
        with TenantContext.scope(TenantId("acme")):
            raise RuntimeError("boom")
    assert TenantContext.get() is None


def test_a_scope_can_explicitly_clear_the_tenant() -> None:
    with TenantContext.scope(TenantId("acme")):
        with TenantContext.scope(None):
            assert TenantContext.get() is None
        assert TenantContext.get() == "acme"


async def test_concurrent_tasks_each_see_their_own_tenant() -> None:
    import asyncio

    seen: dict[str, TenantId | None] = {}

    async def act_for(name: str) -> None:
        with TenantContext.scope(TenantId(name)):
            await asyncio.sleep(0)
            seen[name] = TenantContext.get()

    await asyncio.gather(act_for("acme"), act_for("globex"))
    assert seen == {"acme": "acme", "globex": "globex"}


def test_the_context_variable_is_the_one_the_context_reads() -> None:
    token = current_tenant.set(TenantId("direct"))
    try:
        assert TenantContext.get() == "direct"
    finally:
        current_tenant.reset(token)


async def test_an_authenticated_request_acts_for_its_user(
    make_settings: MakeSettings, keys: KeyPair
) -> None:
    """A per-user budget below the route needs the caller, not only the tenant."""
    _settings(make_settings, keys)
    user = UUID(int=42)
    token = TokenIssuer().issue_access({"user_id": str(user)})

    async def run() -> str | None:
        await current_principal(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        )
        return CallerContext.get()

    assert await asyncio.create_task(run(), context=contextvars.Context()) == str(user)
