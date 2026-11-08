"""The `type` claim a first-party token carries.

The carrier types — `AccessToken`, `RefreshToken` — live beside this in
`models/view_models/tokens.py`. Those name *which string* you are holding and
are erased at runtime; this names the value written *inside* it, which is what
verification compares against.
"""

from enum import StrEnum


class TokenType(StrEnum):
    """The `type` claim every token this library mints carries."""

    ACCESS = "access"
    REFRESH = "refresh"
