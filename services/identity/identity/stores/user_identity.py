from typing import ClassVar

from base.services.common.meta import SingletonABC
from base.services.store.sql import SQLStore
from identity.models.orm.user_identity import UserIdentityDTO


class UserIdentityStore(SQLStore[UserIdentityDTO], metaclass=SingletonABC):
    protected_fields: ClassVar[frozenset[str]] = SQLStore.protected_fields | {
        "provider",
        "subject",
    }

    def __init__(self) -> None:
        super().__init__(UserIdentityDTO)

    def get_by_provider_subject(
        self, provider: str, subject: str
    ) -> UserIdentityDTO | None:
        return self._one(
            self._active(
                UserIdentityDTO.provider == provider,
                UserIdentityDTO.subject == subject,
            )
        )

    async def aget_by_provider_subject(
        self, provider: str, subject: str
    ) -> UserIdentityDTO | None:
        return await self._aone(
            self._active(
                UserIdentityDTO.provider == provider,
                UserIdentityDTO.subject == subject,
            )
        )
