from typing import ClassVar

from sqlalchemy.exc import IntegrityError

from base.models.definitions.errors import DatabaseErrors
from base.services.common.meta import SingletonABC
from base.services.messaging import add_event
from base.services.store.sql import SQLStore
from base.tools.correlation import current_correlation_id
from base.tools.errors import raise_error
from contracts.events import Event
from contracts.v1.identity import USER_REGISTERED, UserRegistered
from identity.models.orm.outbox import OutboxEventDTO
from identity.models.orm.user import UserDTO


class UserStore(SQLStore[UserDTO], metaclass=SingletonABC):
    protected_fields: ClassVar[frozenset[str]] = SQLStore.protected_fields | {"email"}
    search_fields: ClassVar[tuple[str, ...]] = ("email",)

    def __init__(self) -> None:
        super().__init__(UserDTO)

    async def aregister(self, user: UserDTO) -> UserDTO:
        try:
            async with self._sql_service.atransaction() as session:
                session.add(user)
                await session.flush()
                event = Event.of(
                    USER_REGISTERED,
                    UserRegistered(
                        user_id=user.id,
                        email=user.email,
                        registered_at=user.created_at,
                    ),
                    producer="identity",
                    correlation_id=current_correlation_id(),
                )
                add_event(session, OutboxEventDTO, event)
                await session.flush()
        except IntegrityError:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)
        return user
