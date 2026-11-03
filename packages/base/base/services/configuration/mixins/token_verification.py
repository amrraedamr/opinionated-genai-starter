from pydantic import BaseModel


class TokenVerificationMixin(BaseModel):
    """Verify tokens. Does NOT issue them."""

    auth_public_key_path: str = "keys/public.pem"
