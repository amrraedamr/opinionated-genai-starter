from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from base.models.definitions.errors import AuthErrors
from base.models.orm.user import UserDTO
from base.models.view_models import BaseResponse
from base.models.view_models.auth import (
    LoginRequest,
    LoginResponse,
    OAuthExchangeRequest,
    OAuthProvidersResponse,
    RefreshRequest,
    RefreshResponse,
    RefreshToken,
    RegisterRequest,
)
from base.models.view_models.user import CreateUserResponse
from base.services.auth import AuthService
from base.services.auth.oauth import OAuthService
from base.services.store.user import UserStore
from base.tools.errors import raise_error
from base.tools.security import PasswordHashingService

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=BaseResponse[CreateUserResponse])
async def register(
    request: RegisterRequest,
) -> BaseResponse[CreateUserResponse]:
    user = UserDTO(
        email=request.email,
        password_hash=PasswordHashingService.hash(request.password),
        first_name=request.first_name,
        last_name=request.last_name,
    )
    created = await UserStore().aadd(user)
    return BaseResponse(data=CreateUserResponse(user_id=created.id))


@router.post("/login", response_model=BaseResponse[LoginResponse])
async def login(
    request: LoginRequest,
) -> BaseResponse[LoginResponse]:
    user = await UserStore()._aget(request.email, column_name="email")
    if not user or not user.password_hash:
        raise_error(AuthErrors.USER_UNAUTHORIZED)
    if not PasswordHashingService.verify(request.password, user.password_hash):
        raise_error(AuthErrors.USER_UNAUTHORIZED)
    token_pair = AuthService().generate_token_pair(user)
    return BaseResponse(
        data=LoginResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
        )
    )


@router.post("/refresh", response_model=BaseResponse[RefreshResponse])
async def refresh(request: RefreshRequest) -> BaseResponse[RefreshResponse]:
    token_pair = await AuthService().arefresh_access_token(
        RefreshToken(request.refresh_token)
    )
    return BaseResponse(
        data=RefreshResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
        )
    )


@router.get("/oauth/providers", response_model=BaseResponse[OAuthProvidersResponse])
async def oauth_providers() -> BaseResponse[OAuthProvidersResponse]:
    return BaseResponse(
        data=OAuthProvidersResponse(providers=OAuthService().enabled_providers())
    )


@router.get("/oauth/{provider}")
async def oauth_start(provider: str) -> RedirectResponse:
    url = await OAuthService().abuild_authorize_redirect(provider)
    return RedirectResponse(url=url, status_code=302)


@router.api_route("/oauth/{provider}/callback", methods=["GET", "POST"])
async def oauth_callback(
    provider: str,
    request: Request,
    code: str | None = None,
    state: str | None = None,
) -> RedirectResponse:
    # Apple uses response_mode=form_post (POST + optional user JSON on first auth).
    apple_user_json: str | None = None
    if request.method == "POST":
        form = await request.form()
        code = str(form.get("code") or "") or code
        state = str(form.get("state") or "") or state
        user_field = form.get("user")
        if user_field is not None:
            apple_user_json = str(user_field)

    if not code or not state:
        raise_error(AuthErrors.OAUTH_STATE_INVALID)

    exchange_code = await OAuthService().ahandle_callback(
        provider,
        code=code,
        state=state,
        apple_user_json=apple_user_json,
    )
    return RedirectResponse(
        url=OAuthService().success_redirect_url(exchange_code),
        status_code=302,
    )


@router.post("/oauth/exchange", response_model=BaseResponse[LoginResponse])
async def oauth_exchange(
    request: OAuthExchangeRequest,
) -> BaseResponse[LoginResponse]:
    token_pair = await OAuthService().aexchange_code(request.code)
    return BaseResponse(
        data=LoginResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
        )
    )
