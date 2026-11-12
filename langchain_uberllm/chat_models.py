"""UberLLM chat model for LangChain."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import LangSmithParams
from langchain_core.load.serializable import SerializedConstructor, SerializedNotImplemented
from langchain_openai import ChatOpenAI
from pydantic import model_validator

DEFAULT_BASE_URL = "https://api.uberllm.dev/v1"
"""UberLLM's OpenAI-compatible endpoint."""

API_KEY_ENV = "UBERLLM_API_KEY"
BASE_URL_ENV = "UBERLLM_BASE_URL"


class ChatUberLLM(ChatOpenAI):
    """Chat models served through UberLLM (https://uberllm.dev).

    UberLLM is one OpenAI-compatible API key for hundreds of chat models
    (GPT, Claude, Gemini, DeepSeek, Qwen, GLM, Llama, Mistral, Kimi and
    others). This class is `ChatOpenAI` pointed at UberLLM, so everything
    `ChatOpenAI` does (streaming, tool calling, structured output, async,
    batching, the Responses API) works the same way.

    Setup:
        ```bash
        pip install -U langchain-uberllm
        export UBERLLM_API_KEY="ull_..."
        ```

        Create a key at https://uberllm.dev (Dashboard -> Keys). Model ids and
        live prices: https://uberllm.dev/models

    Key init args:
        model: Model id, for example `"deepseek/deepseek-v4-flash"`,
            `"anthropic/claude-sonnet-5.5"` or `"google/gemini-3.5-flash"`.
        api_key: UberLLM key. Read from `UBERLLM_API_KEY` when not passed.
            `uberllm_api_key=` is accepted too.
        base_url: Defaults to `https://api.uberllm.dev/v1`, or
            `UBERLLM_BASE_URL` when that is set.
        temperature, max_tokens, timeout, max_retries, model_kwargs, ...:
            Same as `ChatOpenAI`.

    Instantiate:
        ```python
        from langchain_uberllm import ChatUberLLM

        llm = ChatUberLLM(model="deepseek/deepseek-v4-flash", temperature=0)
        ```

    Invoke:
        ```python
        llm.invoke("Say hello in five words.").content
        ```

    Stream:
        ```python
        for chunk in llm.stream("Count to five."):
            print(chunk.content, end="", flush=True)
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a city.'''

            city: str = Field(description="City name")


        llm.bind_tools([GetWeather]).invoke("Weather in Paris?").tool_calls
        ```

    Structured output:
        ```python
        class Joke(BaseModel):
            setup: str
            punchline: str


        llm.with_structured_output(Joke).invoke("A joke about cats")
        ```

    Spend cap per request (UberLLM routing option, USD per 1M tokens):
        ```python
        llm = ChatUberLLM(
            model="deepseek/deepseek-v4-flash",
            extra_body={"provider": {"max_price": {"prompt": 0.2, "completion": 0.4}}},
        )
        ```
    """

    @model_validator(mode="before")
    @classmethod
    def _uberllm_defaults(cls, values: Any) -> Any:
        """Point the client at UberLLM and take the key from UBERLLM_API_KEY.

        Runs before `ChatOpenAI` resolves its settings, so an `OPENAI_API_KEY`
        or `OPENAI_BASE_URL` in the environment is never used by this class:
        an OpenAI key is never sent to UberLLM, and an UberLLM key is never
        sent to OpenAI.
        """
        if not isinstance(values, dict):
            return values
        values = dict(values)

        alias_key = values.pop("uberllm_api_key", None)
        if values.get("api_key") is None and values.get("openai_api_key") is None:
            key = alias_key if alias_key is not None else os.environ.get(API_KEY_ENV)
            if not key:
                msg = (
                    "No UberLLM API key. Set the UBERLLM_API_KEY environment "
                    "variable or pass api_key='ull_...'. Create a key at "
                    "https://uberllm.dev (Dashboard -> Keys)."
                )
                raise ValueError(msg)
            values["api_key"] = key

        if values.get("base_url") is None and values.get("openai_api_base") is None:
            values["base_url"] = os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL

        # UberLLM reports token usage (and cost) on streamed responses.
        values.setdefault("stream_usage", True)
        return values

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "uberllm-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map the secret constructor argument to its environment variable."""
        return {"openai_api_key": API_KEY_ENV}

    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """Serialize, naming the key's secret `UBERLLM_API_KEY`.

        LangChain merges `lc_secrets` up the class hierarchy with the parent
        winning, so `ChatOpenAI` would otherwise label the key
        `OPENAI_API_KEY` and `load()` would look up the wrong secret.
        """
        data = super().to_json()
        secret = data.get("kwargs", {}).get("openai_api_key")
        if isinstance(secret, dict) and secret.get("type") == "secret":
            secret["id"] = [API_KEY_ENV]
        return data

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Return the LangChain serialization namespace."""
        return ["langchain_uberllm", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Report UberLLM as the provider in LangSmith traces."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "uberllm"
        return params
