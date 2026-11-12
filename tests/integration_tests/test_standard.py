"""LangChain's standard integration tests, run live against UberLLM.

Skipped without UBERLLM_API_KEY. About 40 small requests to a low-cost model.
"""

from __future__ import annotations

import os

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_uberllm import ChatUberLLM

pytestmark = pytest.mark.skipif(
    not os.environ.get("UBERLLM_API_KEY"), reason="UBERLLM_API_KEY not set"
)


class TestChatUberLLMStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatUberLLM]:
        return ChatUberLLM

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": os.environ.get("UBERLLM_TEST_MODEL", "deepseek/deepseek-v4-flash"),
            "temperature": 0,
        }

    @property
    def has_tool_choice(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return False  # the default test model is text-only

    @property
    def supports_model_override(self) -> bool:
        return True

    @property
    def model_override_value(self) -> str:
        return "google/gemini-3.1-flash-lite"
