# Import each table so it registers on IdentityBase.metadata. Add a line per table.
import identity.models.orm.outbox  # noqa: F401
import identity.models.orm.user  # noqa: F401
import identity.models.orm.user_identity  # noqa: F401
from base.tools.migrations import run_migrations
from identity.models.orm import IdentityBase
from identity.settings import settings

run_migrations(IdentityBase.metadata, settings)
