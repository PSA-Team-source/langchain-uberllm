"""LangChain integration for UberLLM (https://uberllm.dev)."""

from importlib import metadata

from langchain_uberllm.chat_models import ChatUberLLM
from langchain_uberllm.models import list_models

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = ["ChatUberLLM", "__version__", "list_models"]
