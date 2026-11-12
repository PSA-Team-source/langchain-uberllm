# langchain-uberllm

LangChain integration for [UberLLM](https://uberllm.dev): one OpenAI-compatible
API key for hundreds of chat models (GPT, Claude, Gemini, DeepSeek, Qwen, GLM,
Llama, Mistral, Kimi and others), usually below list price.

`ChatUberLLM` is LangChain's `ChatOpenAI` pointed at `https://api.uberllm.dev/v1`
with your UberLLM key, so streaming, tool calling, structured output, async,
batching and agents work exactly as they do with `ChatOpenAI`.

- Models and live prices: <https://uberllm.dev/models>
- API docs: <https://uberllm.dev/docs>
- Get a key: <https://uberllm.dev/signup> (prepaid credit; failed requests are not billed)

## Install

```sh
pip install -U langchain-uberllm
```

Until the package is on PyPI, install it from GitHub:

```sh
pip install "git+https://github.com/PSA-Team-source/langchain-uberllm"
```

## Set your key

```sh
export UBERLLM_API_KEY="ull_..."
```

Create the key in **Dashboard → Keys** on <https://uberllm.dev>. You can also pass
`api_key="ull_..."` (or `uberllm_api_key=`) to the constructor.

`ChatUberLLM` never reads `OPENAI_API_KEY` or `OPENAI_BASE_URL`, so an OpenAI key in
your environment is never sent to UberLLM, and your UberLLM key is never sent to
OpenAI.

## Chat

```python
from langchain_uberllm import ChatUberLLM

llm = ChatUberLLM(model="deepseek/deepseek-v4-flash", temperature=0)

print(llm.invoke("Say hello in five words.").content)

for chunk in llm.stream("Count from 1 to 5."):
    print(chunk.content, end="", flush=True)
```

Any chat model id from <https://uberllm.dev/models> works, for example
`anthropic/claude-sonnet-5.5`, `openai/gpt-5.5` or `google/gemini-3.5-flash`.

## Tool calling

```python
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a city."""

    city: str = Field(description="City name, e.g. Paris")


msg = llm.bind_tools([GetWeather]).invoke("What's the weather in Paris?")
print(msg.tool_calls)
```

## Structured output

```python
class Person(BaseModel):
    name: str
    age: int


extract = llm.with_structured_output(Person)
print(extract.invoke("Ada is 36 years old."))  # Person(name='Ada', age=36)
```

`method="json_schema"` (the default) and `method="function_calling"` both work.

## Agents

```python
from langchain.agents import create_agent  # pip install langchain


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny and 21 C in {city}."


agent = create_agent(llm, tools=[get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
print(result["messages"][-1].content)
```

## List models and prices

```python
from langchain_uberllm import list_models

for m in list_models():  # chat models; type="image" | "video" | "music" | "audio" | "all"
    p = m["pricing"]
    print(m["id"], p.get("input_usd"), p.get("output_usd"), m.get("context_length"))
```

Prices are USD per 1M tokens. `list_input_usd` / `list_output_usd`, when present,
are the official list price for the same model.

## Cap what a request can cost

UberLLM's routing options go in `extra_body`. `provider.max_price` (USD per 1M
tokens) skips any seller priced above it:

```python
llm = ChatUberLLM(
    model="deepseek/deepseek-v4-flash",
    extra_body={"provider": {"max_price": {"prompt": 0.2, "completion": 0.4}}},
)
```

Every request, with its cost, is listed in **Dashboard → Activity**.

## Settings

| Argument | Environment variable | Default |
|---|---|---|
| `api_key` / `uberllm_api_key` | `UBERLLM_API_KEY` | required |
| `base_url` | `UBERLLM_BASE_URL` | `https://api.uberllm.dev/v1` |
| `stream_usage` | | `True` (token usage and cost on streamed replies) |

Every other `ChatOpenAI` argument (`temperature`, `max_tokens`, `timeout`,
`max_retries`, `model_kwargs`, `use_responses_api`, ...) works the same way.
UberLLM has no embeddings endpoint, so this package has no embeddings class.

## Development

```sh
pip install -e . pytest langchain-tests
pytest tests/unit_tests                                  # offline
UBERLLM_API_KEY=ull_... pytest tests/integration_tests   # live, a fraction of a cent
```

The unit tests include LangChain's standard `ChatModelUnitTests`; the live tests
include the standard `ChatModelIntegrationTests`.

## Support

support@uberllm.dev · [Docs](https://uberllm.dev/docs/langchain) · [Terms](https://uberllm.dev/terms)

UberLLM is operated by AL Global LLC and is not affiliated with Uber Technologies, Inc.

## License

MIT
