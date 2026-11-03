import importlib
import logging
import pkgutil
from types import ModuleType

from fastapi import FastAPI


def init_routers(
    app: FastAPI,
    routers: ModuleType,
    prefix: str = "",
) -> None:
    """
    Include all routers from the routers package.
    - Walk through the modules inside the routers package.
    - Include the router field from each router module in the FastAPI instance.

    To use the package walker, you need to create routers within the use-case application routers package.

    1) Create your router modules within the routers package, where each router module would contain the router field.
    router = APIRouter(prefix="/router-name", tags=["Router Name"])

    :param app: FastAPI instance
    :return: -> None
    """
    for _, name, is_pkg in pkgutil.iter_modules(routers.__path__):
        if is_pkg:
            continue
        try:
            router_full_path = f"{routers.__name__}.{name}"
            router_module = importlib.import_module(router_full_path)
            app.include_router(router_module.router, prefix=prefix)
        except (ImportError, AttributeError):
            logging.exception(f"[FastAPI] Couldn't import router [{name}].\n")
            continue
