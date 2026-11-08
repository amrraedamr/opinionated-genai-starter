from enum import Enum
from http import HTTPStatus


class AuthErrors(Enum):
    USER_UNAUTHORIZED = {
        "message": "Unauthorized.",
        "status_code": HTTPStatus.UNAUTHORIZED,
    }
    USER_NOT_FOUND = {
        "message": "User not found.",
        "status_code": HTTPStatus.NOT_FOUND,
    }
    USER_TOKEN_EXPIRED = {
        "message": "Token expired.",
        "status_code": HTTPStatus.UNAUTHORIZED,
    }
    USER_DECODE_FAILED = {
        "message": "Invalid token.",
        "status_code": HTTPStatus.UNAUTHORIZED,
    }
    USER_TOKEN_INCOMPLETE = {
        "message": "Token incomplete.",
        "status_code": HTTPStatus.UNAUTHORIZED,
    }
    PASSWORD_NOT_SET = {
        "message": "Password login is not enabled for this account.",
        "status_code": HTTPStatus.BAD_REQUEST,
    }
    OAUTH_PROVIDER_DISABLED = {
        "message": "OAuth provider is not configured.",
        "status_code": HTTPStatus.BAD_REQUEST,
    }
    OAUTH_STATE_INVALID = {
        "message": "Invalid or expired OAuth state.",
        "status_code": HTTPStatus.BAD_REQUEST,
    }
    OAUTH_EXCHANGE_FAILED = {
        "message": "OAuth token exchange failed.",
        "status_code": HTTPStatus.BAD_GATEWAY,
    }
    OAUTH_PROFILE_INCOMPLETE = {
        "message": "OAuth provider did not return a usable identity.",
        "status_code": HTTPStatus.BAD_REQUEST,
    }
    OUTBOUND_TOKEN_FAILED = {
        "message": "Could not acquire credentials for an upstream dependency.",
        "status_code": HTTPStatus.BAD_GATEWAY,
    }
    OAUTH_CODE_INVALID = {
        "message": "Invalid or expired OAuth exchange code.",
        "status_code": HTTPStatus.BAD_REQUEST,
    }
    IDP_NOT_CONFIGURED = {
        "message": "Identity provider resource server is not configured.",
        "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
    }


class DatabaseErrors(Enum):
    RECORD_NOT_FOUND = {
        "message": "Record not found.",
        "status_code": HTTPStatus.NOT_FOUND,
    }
    UNIQUE_CONSTRAINT_VIOLATION = {
        "message": "Record already exists.",
        "status_code": HTTPStatus.CONFLICT,
    }


class GenericErrors(Enum):
    UNEXPECTED_ERROR = {
        "message": "Unexpected error.",
        "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
    }


class TenancyErrors(Enum):
    TENANT_NOT_IN_CONTEXT = {
        "message": "No tenant in context for a tenant-scoped operation.",
        "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
    }
