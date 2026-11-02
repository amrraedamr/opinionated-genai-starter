"""UserStore held to the same contract as every other AbstractStore."""

import uuid
from typing import Any

import pytest

from base.testing import AbstractStoreConformance


@pytest.mark.usefixtures("configured_settings")
class TestUserStoreConformance(AbstractStoreConformance):
    def make_store(self) -> Any:
        from identity.stores.user import UserStore

        return UserStore()

    def make_entity(self, **overrides: Any) -> Any:
        from identity.models.orm.user import UserDTO

        return UserDTO(
            email=overrides.get("email", f"{uuid.uuid4().hex}@example.com"),
            password_hash="hash",
            first_name="Test",
            last_name="User",
        )
