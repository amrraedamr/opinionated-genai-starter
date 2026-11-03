import hashlib
from typing import Any

from base.tools.serializers import serialize


def hash_args(*args: Any, **kwargs: Any) -> str:
    """Deterministic SHA-256 hash of arbitrary args/kwargs.

    Uses the ExpansiveEncoder serializer to handle Pydantic models,
    DTOs, UUIDs, datetimes, and any other registered types.
    """
    payload = serialize({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
