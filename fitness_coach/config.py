import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI model tiers — only used when OPENAI_API_KEY is set.
# When running locally via Ollama, both get_llm() and get_fast_llm() use LOCAL_MODEL.
OPENAI_SMART_MODEL = os.getenv("OPENAI_SMART_MODEL", "gpt-4o")
OPENAI_FAST_MODEL  = os.getenv("OPENAI_FAST_MODEL",  "gpt-4o-mini")
LOCAL_MODEL        = os.getenv("LOCAL_MODEL",         "qwen2.5:1.5b")


def using_openai() -> bool:
    return bool(_OPENAI_API_KEY)


def provider_label() -> str:
    if using_openai():
        return (
            f"OpenAI  smart → {OPENAI_SMART_MODEL}  ·  fast → {OPENAI_FAST_MODEL}"
        )
    return f"Ollama ({LOCAL_MODEL})"


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Smart model — reasoning, tool use, complex generation (e.g. Coach, Generator)."""
    if using_openai():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_SMART_MODEL, temperature=temperature, api_key=_OPENAI_API_KEY)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LOCAL_MODEL, temperature=temperature)


@lru_cache(maxsize=None)
def get_fast_llm(temperature: float = 0.0) -> BaseChatModel:
    """Fast model — classification, binary checks, structured extraction (e.g. Router, Logger)."""
    if using_openai():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_FAST_MODEL, temperature=temperature, api_key=_OPENAI_API_KEY)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LOCAL_MODEL, temperature=temperature)
