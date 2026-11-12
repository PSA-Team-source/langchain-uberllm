"""List the models UberLLM serves."""

from __future__ import annotations

import os
from typing import Any, Literal

import httpx

from langchain_uberllm.chat_models import API_KEY_ENV, BASE_URL_ENV, DEFAULT_BASE_URL

ModelType = Literal["chat", "image", "video", "music", "audio", "all"]


def list_models(
    *,
    type: ModelType = "chat",  # noqa: A002 - matches the API's ?type= parameter
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Return UberLLM's model catalog from `GET /v1/models`.

    Each entry is the API's model object: `id`, `display_name`,
    `context_length`, `pricing` (USD per 1M tokens for chat models, with the
    official list price next to UberLLM's price), `supports_tools`,
    `supports_vision`, `endpoints` and more.

    Args:
        type: `"chat"` (default) for text models usable with `ChatUberLLM`,
            or `"image"`, `"video"`, `"music"`, `"audio"`, `"all"`.
        api_key: UberLLM key. Defaults to `UBERLLM_API_KEY`. The catalog is
            also readable without a key.
        base_url: Defaults to `UBERLLM_BASE_URL`, else
            `https://api.uberllm.dev/v1`.
        timeout: Seconds before the request is abandoned.

    Raises:
        httpx.HTTPStatusError: The API answered with an error status.

    Example:
        ```python
        from langchain_uberllm import list_models

        for m in list_models():
            print(m["id"], m["pricing"]["input_usd"], m["pricing"]["output_usd"])
        ```
    """
    url = (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.environ.get(API_KEY_ENV)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    params = {} if type == "chat" else {"type": type}
    resp = httpx.get(f"{url}/models", headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return list(resp.json().get("data", []))
