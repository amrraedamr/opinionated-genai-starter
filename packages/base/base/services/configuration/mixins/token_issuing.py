from base.services.configuration.mixins.token_verification import TokenVerificationMixin


class TokenIssuingMixin(TokenVerificationMixin):
    """Only a service that issues tokens should include this."""

    auth_private_key_path: str = "keys/private.pem"
    access_token_expiry_seconds: int = 900
    refresh_token_expiry_seconds: int = 604800
