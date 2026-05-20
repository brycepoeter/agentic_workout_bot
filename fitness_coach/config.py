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

_LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
_LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
_LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def using_openai() -> bool:
    return bool(_OPENAI_API_KEY)


def using_langfuse() -> bool:
    return bool(_LANGFUSE_PUBLIC_KEY and _LANGFUSE_SECRET_KEY)


def get_trace_config(session_id: str | None = None) -> dict:
    """Returns a RunnableConfig dict with Langfuse callbacks and optional session grouping.

    Pass the result directly as the config= argument to any graph .invoke() call.
    Returns an empty dict when Langfuse is not configured.
    """
    if not using_langfuse():
        return {}
    from langfuse.langchain import CallbackHandler
    cfg: dict = {"callbacks": [CallbackHandler()]}
    if session_id:
        cfg["metadata"] = {"langfuse_session_id": session_id}
    return cfg


def provider_label() -> str:
    llm_label = (
        f"OpenAI  smart → {OPENAI_SMART_MODEL}  ·  fast → {OPENAI_FAST_MODEL}"
        if using_openai()
        else f"Ollama ({LOCAL_MODEL})"
    )
    trace_label = f"Langfuse ({_LANGFUSE_HOST})" if using_langfuse() else "off"
    return f"{llm_label}  ·  Tracing: {trace_label}"


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
