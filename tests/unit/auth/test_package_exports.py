"""What the auth packages re-export, and why the token types live outside them.

`base.services.auth` guards on the `auth` extra, so a service that only
authenticates itself outbound must not have to import it to name a token.
Pins that the shared types really are shared, and that each `__all__`
resolves.
"""

import importlib
from types import ModuleType

import pytest

import base.services.auth as auth_package
import base.services.auth.models as models_module
import base.services.auth.oauth as oauth_package

PACKAGES: tuple[ModuleType, ...] = (auth_package, models_module, oauth_package)


@pytest.mark.parametrize("package", PACKAGES, ids=lambda m: m.__name__)
def test_every_exported_name_resolves(package: ModuleType) -> None:
    assert [name for name in package.__all__ if not hasattr(package, name)] == []


@pytest.mark.parametrize("package", PACKAGES, ids=lambda m: m.__name__)
def test_all_is_sorted_and_free_of_duplicates(package: ModuleType) -> None:
    assert list(package.__all__) == sorted(package.__all__)
    assert len(package.__all__) == len(set(package.__all__))


def test_the_auth_package_exports_both_halves_of_its_surface() -> None:
    assert set(auth_package.__all__) == {
        "AccessToken",
        "Principal",
        "RefreshToken",
        "SupportsIdpConfig",
        "SupportsTokenIssuingConfig",
        "SupportsTokenVerificationConfig",
        "TokenIssuer",
        "TokenPair",
        "TokenResponse",
        "TokenVerifier",
        "current_principal",
        "current_tenant_id",
        "current_user_id",
        "http_bearer",
        "http_bearer_dependency",
        "principal_dependency",
        "tenant_id_dependency",
        "user_id_dependency",
    }


def test_the_oauth_package_exports_its_registry_and_models() -> None:
    assert set(oauth_package.__all__) == {
        "OAuthProfile",
        "OAuthProviderConfig",
        "OAuthProviderRegistry",
        "build_apple_client_secret",
        "load_oauth_provider_configs",
    }


def test_the_token_types_are_the_ones_outbound_code_imports() -> None:
    """Imported from `base.models.view_models.tokens`, not re-declared here."""
    shared = importlib.import_module("base.models.view_models.tokens")
    credentials = importlib.import_module("base.clients.credentials")
    for name in ("AccessToken", "RefreshToken", "TokenPair", "TokenResponse"):
        assert getattr(models_module, name) is getattr(shared, name)
    assert credentials.AccessToken is shared.AccessToken


def test_the_contracts_the_auth_package_reads_are_the_shared_ones() -> None:
    protocols = importlib.import_module("base.services.configuration.protocols")
    for name in (
        "SupportsIdpConfig",
        "SupportsTokenIssuingConfig",
        "SupportsTokenVerificationConfig",
    ):
        assert getattr(auth_package, name) is getattr(protocols, name)
