import asyncio
from typing import Any, Literal, overload

import httpx
import requests

from base.models.view_models.http import HttpResponse
from base.services.common.meta import Singleton


class HttpRequestService(metaclass=Singleton):
    _BASE_BACKOFF_SECONDS: float = 0.5
    _MAX_RETRIES: int = 3

    def __init__(self) -> None:
        self._sync_client: requests.Session = requests.Session()
        self._async_client: httpx.AsyncClient = httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30,
            ),
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        self._default_api_timeout: float = 10.0

    @property
    def sync_client(self) -> requests.Session:
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        return self._async_client

    @staticmethod
    def _is_json_content_type(content_type: str) -> bool:
        return (
            "application/json" in content_type
            or "application/graphql-response+json" in content_type
        )

    @staticmethod
    def _parse_response_body(
        content_type: str,
        json_fn: Any,
        text: str,
        url: str,
    ) -> dict[str, Any]:
        if HttpRequestService._is_json_content_type(content_type):
            return json_fn()
        return {"url": url, "text": text}

    @staticmethod
    def _response_headers(headers: Any) -> dict[str, str]:
        # Normalize to lowercase so lookups stay case-insensitive after flattening.
        return {str(key).lower(): str(value) for key, value in headers.items()}

    @staticmethod
    def _extract_error_detail(
        response: requests.Response | httpx.Response | None,
    ) -> dict[str, Any] | str | None:
        if response is None:
            return None
        try:
            return response.json()
        except (ValueError, RuntimeError):
            return response.text

    @overload
    def call_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        *,
        raw: Literal[False] = False,
    ) -> HttpResponse: ...

    @overload
    def call_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        *,
        raw: Literal[True],
    ) -> bytes: ...

    @overload
    def call_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        *,
        raw: bool = False,
    ) -> HttpResponse | bytes: ...

    def call_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        raw: bool = False,
    ) -> HttpResponse | bytes:
        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            try:
                resp: requests.Response = self._sync_client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=query_params,
                    json=body,
                    timeout=timeout or self._default_api_timeout,
                    auth=auth,
                    data=data,
                    files=files,
                )
                resp.raise_for_status()

                if raw:
                    return resp.content

                ctype: str = resp.headers.get("Content-Type", "")
                body_data: dict[str, Any] = self._parse_response_body(
                    content_type=ctype,
                    json_fn=resp.json,
                    text=resp.text,
                    url=str(resp.url),
                )
                return HttpResponse.ok(
                    status_code=resp.status_code,
                    data=body_data,
                    headers=self._response_headers(resp.headers),
                )

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < retry_count:
                    import time

                    time.sleep(self._BASE_BACKOFF_SECONDS * (attempt + 1))
                    continue
                if raw:
                    raise RuntimeError(
                        f"Request timed out after {retry_count + 1} attempts: {e}"
                    ) from e
                return HttpResponse.fail(error=e)

            except requests.exceptions.RequestException as e:
                if raw:
                    raise RuntimeError(f"Request failed: {e}") from e
                return HttpResponse.fail(
                    error=e,
                    status_code=e.response.status_code
                    if e.response is not None
                    else None,
                    error_detail=self._extract_error_detail(e.response),
                    headers=self._response_headers(e.response.headers)
                    if e.response is not None
                    else None,
                )

        if raw:
            raise RuntimeError(
                f"Request timed out after {retry_count + 1} attempts: {last_error}"
            )
        return HttpResponse.fail(error=last_error or Exception("Unknown error"))

    @overload
    async def acall_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        *,
        raw: Literal[False] = False,
    ) -> HttpResponse: ...

    @overload
    async def acall_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        *,
        raw: Literal[True],
    ) -> bytes: ...

    @overload
    async def acall_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        *,
        raw: bool = False,
    ) -> HttpResponse | bytes: ...

    async def acall_endpoint(
        self,
        url: str,
        method: str = "GET",
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_count: int = _MAX_RETRIES,
        raw: bool = False,
    ) -> HttpResponse | bytes:
        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            try:
                resp: httpx.Response = await self._async_client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=query_params,
                    json=body,
                    timeout=timeout or self._default_api_timeout,
                    auth=auth,
                    data=data,
                    files=files,
                    follow_redirects=True,
                )
                resp.raise_for_status()

                if raw:
                    return resp.content

                ctype: str = resp.headers.get("Content-Type", "")
                body_data: dict[str, Any] = self._parse_response_body(
                    content_type=ctype,
                    json_fn=resp.json,
                    text=resp.text,
                    url=str(resp.url),
                )
                return HttpResponse.ok(
                    status_code=resp.status_code,
                    data=body_data,
                    headers=self._response_headers(resp.headers),
                )

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < retry_count:
                    await asyncio.sleep(self._BASE_BACKOFF_SECONDS * (attempt + 1))
                    continue
                if raw:
                    raise RuntimeError(
                        f"Request timed out after {retry_count + 1} attempts: {e}"
                    ) from e
                return HttpResponse.fail(error=e)

            except httpx.HTTPStatusError as e:
                if raw:
                    raise RuntimeError(
                        f"Request failed with status {e.response.status_code}: {e}"
                    ) from e
                return HttpResponse.fail(
                    error=e,
                    status_code=e.response.status_code,
                    error_detail=self._extract_error_detail(e.response),
                    headers=self._response_headers(e.response.headers),
                )

            except httpx.RequestError as e:
                if raw:
                    raise RuntimeError(f"Request failed: {e}") from e
                return HttpResponse.fail(error=e)

        if raw:
            raise RuntimeError(
                f"Request timed out after {retry_count + 1} attempts: {last_error}"
            )
        return HttpResponse.fail(error=last_error or Exception("Unknown error"))

    def close(self) -> None:
        self._sync_client.close()

    async def aclose(self) -> None:
        await self._async_client.aclose()
        self._sync_client.close()
