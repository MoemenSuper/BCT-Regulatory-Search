from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from dotenv import load_dotenv
from functools import lru_cache
import os
import threading
import requests
from urllib.parse import urlparse
from typing import Any, Optional


load_dotenv()
def _ollama_messages(value):
    messages = value.to_messages() if hasattr(value, "to_messages") else value
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    return [
        {
            "role": roles.get(getattr(message, "type", "human"), "user"),
            "content": str(message.content),
        }
        for message in messages
    ]


def _create_ollama_llm():
    base_url = os.environ.get("BCT_LOCAL_LLM_URL", "http://127.0.0.1:11434").rstrip("/")
    host = urlparse(base_url).hostname
    allow_remote = os.environ.get("BCT_ALLOW_REMOTE_LOCAL_LLM") == "1"
    if host not in {"127.0.0.1", "::1", "localhost"} and not allow_remote:
        raise ValueError(
            "BCT_LOCAL_LLM_URL must be loopback unless "
            "BCT_ALLOW_REMOTE_LOCAL_LLM=1 is explicitly set"
        )
    model = os.environ.get("BCT_LOCAL_LLM_MODEL", "qwen3.5:9b-q4_K_M")
    if model.casefold().endswith(":cloud"):
        raise ValueError("The local profile cannot use an Ollama cloud model")

    def invoke(value):
        # Qwen3.5 and similar Ollama thinking models leave message.content empty
        # unless think is disabled; keep that off by default for structured routing.
        payload = {
            "model": model,
            "messages": _ollama_messages(value),
            "stream": False,
            "think": os.environ.get("BCT_LOCAL_LLM_THINK", "0") == "1",
            "options": {
                "temperature": 0,
                "num_predict": int(
                    os.environ.get("BCT_LOCAL_LLM_MAX_TOKENS", "2048")
                ),
                # Evidence is whole pages, not single chunks; Ollama's default
                # 4k window would silently truncate the prompt.
                "num_ctx": int(os.environ.get("BCT_LOCAL_LLM_NUM_CTX", "16384")),
            },
        }
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=float(os.environ.get("BCT_LOCAL_LLM_TIMEOUT_SECONDS", "180")),
        )
        response.raise_for_status()
        message = response.json().get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # Last-resort recovery if a thinking model still returned empty content.
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                content = thinking
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned an empty or malformed chat response")
        return AIMessage(content=content)

    return RunnableLambda(invoke)


def _groq_api_keys():
    keys = []
    for name in ("GROQ_API_KEY", *(f"GROQ_API_KEY_{index}" for index in range(2, 8))):
        value = (os.environ.get(name) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _groq_api_key():
    keys = _groq_api_keys()
    return keys[0] if keys else None


def _groq_rate_limited(error: BaseException) -> bool:
    text = str(error).casefold()
    return any(
        token in text
        for token in (
            "429",
            "rate limit",
            "ratelimit",
            "too_many_requests",
            "quota",
        )
    )


def _chat_groq(api_key: str) -> ChatGroq:
    return ChatGroq(
        model=os.environ.get("BCT_GROQ_MODEL", "openai/gpt-oss-120b"),
        groq_api_key=api_key,
        temperature=0,
        # Medium reasoning + a tight max_tokens often yields empty message.content
        # (tokens spent on hidden reasoning). Prefer low effort and a larger budget.
        reasoning_effort=os.environ.get("BCT_GROQ_REASONING_EFFORT", "low"),
        reasoning_format="hidden",
        max_tokens=int(os.environ.get("BCT_GROQ_MAX_TOKENS", "8192")),
    )


class _RotatingGroqLLM(Runnable):
    """ChatGroq wrapper that advances through GROQ_API_KEY[_N] on rate limits."""

    def __init__(self):
        keys = _groq_api_keys()
        if not keys:
            raise RuntimeError("GROQ_API_KEY is required for the cloud answer provider")
        self._keys = keys
        self._lock = threading.Lock()
        self._cursor = 0
        self._clients = {key: _chat_groq(key) for key in keys}

    def _next_start(self) -> int:
        with self._lock:
            start = self._cursor % len(self._keys)
            self._cursor += 1
            return start

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        start = self._next_start()
        last_error: Exception | None = None
        for offset in range(len(self._keys)):
            key = self._keys[(start + offset) % len(self._keys)]
            try:
                return self._clients[key].invoke(input, config=config, **kwargs)
            except Exception as error:  # noqa: BLE001 - rotate only on quota
                last_error = error
                if _groq_rate_limited(error) and offset + 1 < len(self._keys):
                    continue
                raise
        raise RuntimeError(f"Groq unavailable after trying configured keys: {last_error}")


@lru_cache(maxsize=3)
def create_llm(provider="groq"):
    if provider == "ollama":
        return _create_ollama_llm()
    if provider != "groq":
        raise ValueError(f"Unsupported answer provider: {provider}")
    return _RotatingGroqLLM()
