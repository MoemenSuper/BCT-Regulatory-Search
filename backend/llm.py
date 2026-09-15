from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from functools import lru_cache
import os
import requests
from urllib.parse import urlparse

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


def _groq_api_key():
    for name in ("GROQ_API_KEY", *(f"GROQ_API_KEY_{index}" for index in range(2, 8))):
        if os.environ.get(name):
            return os.environ[name]
    return None


@lru_cache(maxsize=3)
def create_llm(provider="groq"):
    if provider == "ollama":
        return _create_ollama_llm()
    if provider != "groq":
        raise ValueError(f"Unsupported answer provider: {provider}")
    return ChatGroq(
        model=os.environ.get("BCT_GROQ_MODEL", "openai/gpt-oss-120b"),
        groq_api_key=_groq_api_key(),
        temperature=0,
        reasoning_effort=os.environ.get("BCT_GROQ_REASONING_EFFORT", "medium"),
        reasoning_format="hidden",
        max_tokens=int(os.environ.get("BCT_GROQ_MAX_TOKENS", "2048")),
    )
