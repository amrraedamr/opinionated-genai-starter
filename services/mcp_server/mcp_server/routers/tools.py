from fastapi import APIRouter

from base.models.view_models.auth import Principal
from base.services.auth.bearer import principal_dependency
from mcp_server.tools import toolbox

router = APIRouter(prefix="/mcp_server", tags=["mcp_server"])


@router.get("/tools")
async def list_tools(principal: Principal = principal_dependency) -> list[str]:
    """The tools served over MCP, for a caller checking what it can reach."""
    del principal
    return toolbox.names
