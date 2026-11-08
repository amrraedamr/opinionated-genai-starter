"""The `type` claim this service writes into the payloads it signs."""

from enum import StrEnum


class SignedPayloadType(StrEnum):
    """What a short-lived signed payload from this service is for."""

    OAUTH_CODE = "oauth_code"
