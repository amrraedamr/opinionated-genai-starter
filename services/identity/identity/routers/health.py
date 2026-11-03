from fastapi import APIRouter

router = APIRouter(prefix="/identity", tags=["identity"])


@router.get("/ping")
async def ping() -> dict[str, str]:
    return {"service": "identity"}
