from typing import ClassVar

from base.models.orm.user import UserDTO
from base.services.common.meta import SingletonABC
from base.services.store.sql import SQLStore


class UserStore(SQLStore[UserDTO], metaclass=SingletonABC):
    protected_fields: ClassVar[frozenset[str]] = SQLStore.protected_fields | {"email"}
    search_fields: ClassVar[tuple[str, ...]] = ("email",)

    def __init__(self) -> None:
        super().__init__(UserDTO)
