"""`routers/health.py`: the liveness answer, which names the service."""

from httpx import Response

from identity.routers.health import router

from .conftest import client_for


def test_a_ping_names_the_service_answering_it() -> None:
    with client_for(router) as client:
        response: Response = client.get("/identity/ping")
    assert response.status_code == 200
    assert response.json() == {"service": "identity"}
