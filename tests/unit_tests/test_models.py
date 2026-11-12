"""Offline tests for list_models (HTTP is mocked)."""

from __future__ import annotations

import os
from unittest import mock

import httpx
import pytest

from langchain_uberllm import list_models


def _fake_get(captured: dict, status: int = 200):
    def fake_get(url, headers=None, params=None, timeout=None):  # noqa: ANN001, ANN202
        captured.update(url=url, headers=headers, params=params)
        request = httpx.Request("GET", url)
        body = {"object": "list", "data": [{"id": "deepseek/deepseek-v4-flash", "type": "chat"}]}
        return httpx.Response(status, json=body if status == 200 else {"error": "x"}, request=request)

    return fake_get


def test_list_models_chat_default() -> None:
    captured: dict = {}
    with mock.patch.dict(os.environ, {"UBERLLM_API_KEY": "ull_env"}), mock.patch(
        "langchain_uberllm.models.httpx.get", _fake_get(captured)
    ):
        models = list_models()
    assert models == [{"id": "deepseek/deepseek-v4-flash", "type": "chat"}]
    assert captured["url"] == "https://api.uberllm.dev/v1/models"
    assert captured["params"] == {}
    assert captured["headers"] == {"Authorization": "Bearer ull_env"}


def test_list_models_type_and_base_url() -> None:
    captured: dict = {}
    with mock.patch("langchain_uberllm.models.httpx.get", _fake_get(captured)):
        list_models(type="image", api_key="ull_k", base_url="https://x.test/v1/")
    assert captured["url"] == "https://x.test/v1/models"
    assert captured["params"] == {"type": "image"}


def test_list_models_keyless() -> None:
    captured: dict = {}
    env = {k: v for k, v in os.environ.items() if k != "UBERLLM_API_KEY"}
    with mock.patch.dict(os.environ, env, clear=True), mock.patch(
        "langchain_uberllm.models.httpx.get", _fake_get(captured)
    ):
        list_models()
    assert captured["headers"] == {}


def test_list_models_raises_on_error() -> None:
    with mock.patch("langchain_uberllm.models.httpx.get", _fake_get({}, status=401)):
        with pytest.raises(httpx.HTTPStatusError):
            list_models(api_key="bad")
