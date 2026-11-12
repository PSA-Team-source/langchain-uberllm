"""Offline unit tests: no network, no key needed."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd, load
from langchain_tests.unit_tests import ChatModelUnitTests
from syrupy.assertion import SnapshotAssertion

from langchain_uberllm import ChatUberLLM
from langchain_uberllm.chat_models import DEFAULT_BASE_URL

MODEL = "deepseek/deepseek-v4-flash"


class TestChatUberLLMStandard(ChatModelUnitTests):
    """LangChain's standard unit tests for chat model integrations."""

    @property
    def chat_model_class(self) -> type[ChatUberLLM]:
        return ChatUberLLM

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL, "api_key": "ull_test"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"UBERLLM_API_KEY": "ull_env", "UBERLLM_BASE_URL": "https://example.test/v1"},
            {"model": MODEL},
            {"openai_api_key": "ull_env", "openai_api_base": "https://example.test/v1"},
        )

    @pytest.mark.xfail(
        reason=(
            "The standard test loads with allowed_objects='all', which only covers "
            "classes in langchain-core's own mapping; see test_serdes_allowlisted."
        ),
        strict=True,
    )
    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        super().test_serdes(model, snapshot)

    def test_serdes_allowlisted(
        self, model: BaseChatModel, snapshot: SnapshotAssertion
    ) -> None:
        """The standard serdes check, with ChatUberLLM on the load allowlist."""
        env_params, _, _ = self.init_from_env_params
        with mock.patch.dict(os.environ, env_params):
            ser = dumpd(model)
            assert ser == snapshot(name="serialized")
            restored = load(
                ser,
                valid_namespaces=model.get_lc_namespace()[:1],
                allowed_objects=[ChatUberLLM],
                secrets_from_env=True,
            )
        assert isinstance(restored, ChatUberLLM)
        assert restored.openai_api_key.get_secret_value() == "ull_env"
        assert model.model_dump(exclude={"openai_api_key"}) == restored.model_dump(
            exclude={"openai_api_key"}
        )


def _clean_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {"UBERLLM_API_KEY", "UBERLLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"}
    }
    env.update(extra)
    return env


def test_defaults_to_uberllm_base_url() -> None:
    with mock.patch.dict(os.environ, _clean_env(), clear=True):
        llm = ChatUberLLM(model=MODEL, api_key="ull_x")
    assert llm.openai_api_base == DEFAULT_BASE_URL
    assert str(llm.root_client.base_url).rstrip("/") == DEFAULT_BASE_URL


def test_openai_env_is_ignored() -> None:
    """An OpenAI key or base URL in the environment must never be used."""
    env = _clean_env(
        UBERLLM_API_KEY="ull_env",
        OPENAI_API_KEY="sk-openai",
        OPENAI_BASE_URL="https://api.openai.com/v1",
        OPENAI_API_BASE="https://api.openai.com/v1",
    )
    with mock.patch.dict(os.environ, env, clear=True):
        llm = ChatUberLLM(model=MODEL)
    assert llm.openai_api_key.get_secret_value() == "ull_env"
    assert llm.openai_api_base == DEFAULT_BASE_URL
    assert llm.root_client.api_key == "ull_env"


def test_missing_key_raises_even_with_openai_key() -> None:
    with mock.patch.dict(os.environ, _clean_env(OPENAI_API_KEY="sk-openai"), clear=True):
        with pytest.raises(ValueError, match="UBERLLM_API_KEY"):
            ChatUberLLM(model=MODEL)


def test_uberllm_api_key_alias() -> None:
    with mock.patch.dict(os.environ, _clean_env(), clear=True):
        llm = ChatUberLLM(model=MODEL, uberllm_api_key="ull_alias")
    assert llm.openai_api_key.get_secret_value() == "ull_alias"
    assert "uberllm_api_key" not in llm.model_kwargs


def test_explicit_base_url_wins() -> None:
    with mock.patch.dict(os.environ, _clean_env(UBERLLM_BASE_URL="https://env.test/v1"), clear=True):
        llm = ChatUberLLM(model=MODEL, api_key="ull_x", base_url="https://kwarg.test/v1")
    assert llm.openai_api_base == "https://kwarg.test/v1"


def test_stream_usage_on_by_default_and_overridable() -> None:
    assert ChatUberLLM(model=MODEL, api_key="ull_x").stream_usage is True
    assert ChatUberLLM(model=MODEL, api_key="ull_x", stream_usage=False).stream_usage is False


def test_identity() -> None:
    llm = ChatUberLLM(model=MODEL, api_key="ull_x")
    assert llm._llm_type == "uberllm-chat"
    assert llm._get_ls_params()["ls_provider"] == "uberllm"
    assert llm.lc_secrets == {"openai_api_key": "UBERLLM_API_KEY"}


def test_serialization_hides_key_and_round_trips() -> None:
    llm = ChatUberLLM(model=MODEL, api_key="ull_secret_value", temperature=0.3)
    data = dumpd(llm)
    assert "ull_secret_value" not in str(data)
    assert data["id"] == ["langchain_uberllm", "chat_models", "ChatUberLLM"]
    restored = load(
        data,
        secrets_map={"UBERLLM_API_KEY": "ull_secret_value"},
        valid_namespaces=["langchain_uberllm"],
        allowed_objects=[ChatUberLLM],
    )
    assert isinstance(restored, ChatUberLLM)
    assert restored.model_name == MODEL
    assert restored.temperature == 0.3


def test_request_payload_uses_chat_completions() -> None:
    llm = ChatUberLLM(
        model=MODEL,
        api_key="ull_x",
        max_tokens=50,
        extra_body={"provider": {"max_price": {"prompt": 0.2, "completion": 0.4}}},
    )
    payload = llm._get_request_payload("hi")
    assert payload["model"] == MODEL
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["extra_body"]["provider"]["max_price"]["prompt"] == 0.2
