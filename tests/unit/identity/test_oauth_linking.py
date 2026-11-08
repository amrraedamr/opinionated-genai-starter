"""Which local account a provider's user becomes.

This is where an OAuth login can go wrong quietly: a provider that will not
say an address is verified must not be allowed to reach an account that was
created with it. The subject is the only identifier a provider owns, so the
link is keyed on it and the email is only ever a way of finding an account
the user has already proved they hold.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from base.models.definitions.errors import AuthErrors, DatabaseErrors
from base.services.auth.oauth import OAuthProfile
from base.tools.errors import ErrorsException
from identity.models.orm.user import UserDTO
from identity.services.auth.oauth import OAuthService
from identity.stores.user import UserStore
from identity.stores.user_identity import UserIdentityStore


@pytest.fixture
def service(identity_settings: Any) -> OAuthService:
    return OAuthService()


def profile(**overrides: Any) -> OAuthProfile:
    values: dict[str, Any] = {
        "subject": "subject-1",
        "email": "person@example.com",
        "email_verified": True,
        "first_name": "Ada",
        "last_name": "Lovelace",
        **overrides,
    }
    return OAuthProfile(**values)


async def local_account(email: str = "person@example.com") -> UserDTO:
    """An account that already exists, created with a password."""
    return await UserStore().aregister(UserDTO(email=email, password_hash="argon2"))


# Incomplete profiles


async def test_a_profile_with_no_subject_is_refused(service: OAuthService) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aresolve_oauth_user("google", profile(subject=""))
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


async def test_a_profile_with_no_email_is_refused(service: OAuthService) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aresolve_oauth_user("google", profile(email=""))
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


# First login


async def test_a_first_login_creates_the_account_the_provider_describes(
    service: OAuthService,
) -> None:
    user = await service.aresolve_oauth_user("google", profile())
    assert user.email == "person@example.com"
    assert user.first_name == "Ada"
    assert user.last_name == "Lovelace"


async def test_an_account_created_by_a_provider_has_no_password(
    service: OAuthService,
) -> None:
    """Nothing to guess: this account can only be reached through the provider."""
    user = await service.aresolve_oauth_user("google", profile())
    assert user.password_hash is None


async def test_a_first_login_records_what_was_linked(service: OAuthService) -> None:
    user = await service.aresolve_oauth_user("google", profile())
    link = await UserIdentityStore().aget_by_provider_subject("google", "subject-1")
    assert link is not None
    assert link.user_id == user.id
    assert link.provider == "google"
    assert link.email_at_link == "person@example.com"


# Returning users


async def test_a_returning_user_is_recognised_by_subject(
    service: OAuthService,
) -> None:
    first = await service.aresolve_oauth_user("google", profile())
    second = await service.aresolve_oauth_user("google", profile())
    assert second.id == first.id


async def test_a_returning_user_is_not_linked_twice(service: OAuthService) -> None:
    await service.aresolve_oauth_user("google", profile())
    await service.aresolve_oauth_user("google", profile())
    links = await UserIdentityStore()._aget_many(["subject-1"], column_name="subject")
    assert len(links) == 1


async def test_a_subject_that_changed_its_address_still_reaches_its_account(
    service: OAuthService,
) -> None:
    """The subject is what the provider owns; the address is not."""
    first = await service.aresolve_oauth_user("google", profile())
    second = await service.aresolve_oauth_user(
        "google", profile(email="renamed@example.com")
    )
    assert second.id == first.id
    assert second.email == "person@example.com"


async def test_the_same_subject_at_another_provider_is_a_separate_link(
    service: OAuthService,
) -> None:
    google = await service.aresolve_oauth_user("google", profile())
    github = await service.aresolve_oauth_user("github", profile())
    assert github.id == google.id
    assert await UserIdentityStore().aget_by_provider_subject("github", "subject-1")


# Linking to an account that already exists


async def test_a_verified_address_links_to_the_account_that_holds_it(
    service: OAuthService,
) -> None:
    existing = await local_account()
    linked = await service.aresolve_oauth_user("google", profile())
    assert linked.id == existing.id


async def test_linking_leaves_the_existing_password_alone(
    service: OAuthService,
) -> None:
    await local_account()
    linked = await service.aresolve_oauth_user("google", profile())
    assert linked.password_hash == "argon2"


async def test_an_unverified_address_never_reaches_an_existing_account(
    service: OAuthService,
) -> None:
    """The takeover: a provider that will not vouch for an address it was given.

    Refusing the second account is the unique email doing its job; what
    matters is that the existing account is not handed over.
    """
    existing = await local_account()
    with pytest.raises(ErrorsException) as caught:
        await service.aresolve_oauth_user("google", profile(email_verified=False))
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION
    assert not await UserIdentityStore().aget_by_provider_subject("google", "subject-1")
    assert (await UserStore().aget(existing.id)).password_hash == "argon2"


async def test_an_unverified_address_nobody_holds_still_creates_an_account(
    service: OAuthService,
) -> None:
    """Closed defaults are about takeover, not about refusing new users."""
    user = await service.aresolve_oauth_user("google", profile(email_verified=False))
    assert user.email == "person@example.com"


async def test_two_providers_vouching_for_one_address_reach_one_account(
    service: OAuthService,
) -> None:
    google = await service.aresolve_oauth_user("google", profile())
    github = await service.aresolve_oauth_user(
        "github", profile(subject="github-subject")
    )
    assert github.id == google.id


async def test_an_address_at_another_account_is_left_alone(
    service: OAuthService,
) -> None:
    await local_account("someone.else@example.com")
    user = await service.aresolve_oauth_user("google", profile())
    assert user.email == "person@example.com"


# The IdP path, which arrives already verified


async def test_an_idp_principal_resolves_through_the_same_linking(
    service: OAuthService,
) -> None:
    existing = await local_account()
    user = await service.aresolve_idp_user(
        provider="entra",
        subject="entra-subject",
        email="person@example.com",
        first_name="Ada",
    )
    assert user.id == existing.id


async def test_an_idp_principal_is_trusted_by_default(service: OAuthService) -> None:
    """A verified IdP is the premise of accepting its tokens at all."""
    existing = await local_account()
    user = await service.aresolve_idp_user(
        provider="entra", subject="entra-subject", email="person@example.com"
    )
    assert user.id == existing.id


async def test_an_idp_that_says_an_address_is_unverified_is_not_trusted(
    service: OAuthService,
) -> None:
    await local_account()
    with pytest.raises(ErrorsException) as caught:
        await service.aresolve_idp_user(
            provider="entra",
            subject="entra-subject",
            email="person@example.com",
            email_verified=False,
        )
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_an_idp_principal_carries_its_names_through(
    service: OAuthService,
) -> None:
    user = await service.aresolve_idp_user(
        provider="entra",
        subject="entra-subject",
        email="new@example.com",
        first_name="Grace",
        last_name="Hopper",
    )
    assert (user.first_name, user.last_name) == ("Grace", "Hopper")


# Soft deletion


async def test_a_deleted_link_does_not_recognise_the_subject(
    service: OAuthService,
) -> None:
    await service.aresolve_oauth_user("google", profile())
    link = await UserIdentityStore().aget_by_provider_subject("google", "subject-1")
    assert link is not None
    await UserIdentityStore().adelete(link.id)
    assert not await UserIdentityStore().aget_by_provider_subject("google", "subject-1")


async def test_a_user_the_store_cannot_find_is_a_record_not_found(
    service: OAuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await UserStore().aget(uuid.uuid4())
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND
