from enum import Enum

from fastapi import status


class AuthErrors(Enum):
    USER_UNAUTHORIZED = {
        "message": "Unauthorized.",
        "status_code": status.HTTP_401_UNAUTHORIZED,
    }
    USER_NOT_FOUND = {
        "message": "User not found.",
        "status_code": status.HTTP_404_NOT_FOUND,
    }
    USER_TOKEN_EXPIRED = {
        "message": "Token expired.",
        "status_code": status.HTTP_401_UNAUTHORIZED,
    }
    USER_DECODE_FAILED = {
        "message": "Invalid token.",
        "status_code": status.HTTP_401_UNAUTHORIZED,
    }
    USER_TOKEN_INCOMPLETE = {
        "message": "Token incomplete.",
        "status_code": status.HTTP_401_UNAUTHORIZED,
    }
    PASSWORD_NOT_SET = {
        "message": "Password login is not enabled for this account.",
        "status_code": status.HTTP_400_BAD_REQUEST,
    }
    OAUTH_PROVIDER_DISABLED = {
        "message": "OAuth provider is not configured.",
        "status_code": status.HTTP_400_BAD_REQUEST,
    }
    OAUTH_STATE_INVALID = {
        "message": "Invalid or expired OAuth state.",
        "status_code": status.HTTP_400_BAD_REQUEST,
    }
    OAUTH_EXCHANGE_FAILED = {
        "message": "OAuth token exchange failed.",
        "status_code": status.HTTP_502_BAD_GATEWAY,
    }
    OAUTH_PROFILE_INCOMPLETE = {
        "message": "OAuth provider did not return a usable identity.",
        "status_code": status.HTTP_400_BAD_REQUEST,
    }
    OAUTH_CODE_INVALID = {
        "message": "Invalid or expired OAuth exchange code.",
        "status_code": status.HTTP_400_BAD_REQUEST,
    }
    IDP_NOT_CONFIGURED = {
        "message": "Identity provider resource server is not configured.",
        "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
    }


class DatabaseErrors(Enum):
    RECORD_NOT_FOUND = {
        "message": "Record not found.",
        "status_code": status.HTTP_404_NOT_FOUND,
    }
    UNIQUE_CONSTRAINT_VIOLATION = {
        "message": "Record already exists.",
        "status_code": status.HTTP_409_CONFLICT,
    }


class GenericErrors(Enum):
    UNEXPECTED_ERROR = {
        "message": "Unexpected error.",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
