"""Live tests against https://api.uberllm.dev. Skipped without UBERLLM_API_KEY.

Each run makes a handful of small requests to a low-cost model (well under
one US cent in total).
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel, Field

from langchain_uberllm import ChatUberLLM, list_models

pytestmark = pytest.mark.skipif(
    not os.environ.get("UBERLLM_API_KEY"), reason="UBERLLM_API_KEY not set"
)

MODEL = os.environ.get("UBERLLM_TEST_MODEL", "deepseek/deepseek-v4-flash")


@pytest.fixture
def llm() -> ChatUberLLM:
    return ChatUberLLM(model=MODEL, temperature=0, max_tokens=200)


def test_invoke(llm: ChatUberLLM) -> None:
    msg = llm.invoke("Reply with exactly the word: pong")
    assert isinstance(msg, AIMessage)
    assert "pong" in str(msg.content).lower()
    assert msg.usage_metadata and msg.usage_metadata["total_tokens"] > 0


async def test_ainvoke(llm: ChatUberLLM) -> None:
    msg = await llm.ainvoke("Reply with exactly the word: pong")
    assert "pong" in str(msg.content).lower()


def test_stream(llm: ChatUberLLM) -> None:
    full: AIMessageChunk | None = None
    n = 0
    for chunk in llm.stream("Count from 1 to 5, separated by spaces."):
        full = chunk if full is None else full + chunk
        n += 1
    assert full is not None and n > 1
    assert "5" in str(full.content)
    assert full.usage_metadata and full.usage_metadata["output_tokens"] > 0


class GetWeather(BaseModel):
    """Get the current weather in a city."""

    city: str = Field(description="City name, e.g. Paris")


def test_bind_tools(llm: ChatUberLLM) -> None:
    msg = llm.bind_tools([GetWeather], tool_choice="any").invoke(
        "What is the weather in Paris right now?"
    )
    assert msg.tool_calls, msg
    call = msg.tool_calls[0]
    assert call["name"] == "GetWeather"
    assert "paris" in call["args"]["city"].lower()


class Person(BaseModel):
    """A person mentioned in the text."""

    name: str
    age: int


@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
def test_with_structured_output(llm: ChatUberLLM, method: str) -> None:
    structured = llm.with_structured_output(Person, method=method)
    out = structured.invoke("Ada is 36 years old. Extract the person.")
    assert isinstance(out, Person)
    assert out.name.lower().startswith("ada")
    assert out.age == 36


def test_list_models_live() -> None:
    ids = {m["id"] for m in list_models()}
    assert MODEL in ids
