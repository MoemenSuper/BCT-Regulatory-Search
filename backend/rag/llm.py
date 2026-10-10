import anthropic
from google.genai import errors as gemini_errors
import groq
from langchain_groq import ChatGroq
import openai
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from dotenv import load_dotenv
from functools import lru_cache
import logging
import os
import threading
import requests
from urllib.parse import urlparse
from typing import Any, Optional


load_dotenv()
logger = logging.getLogger(__name__)

# Answer-model transport failures (any provider's API errors, Ollama HTTP errors). Callers must
# not turn these into refusals or weaker answers: the API reports them as a temporary outage (503).
PROVIDER_ERRORS = (
    groq.APIError, openai.APIError, anthropic.APIError, gemini_errors.APIError, requests.RequestException,
)
_TIMEOUT = float(os.environ.get("BCT_LLM_TIMEOUT_SECONDS", "180"))
_MAX_TOKENS = 8192


def _chat_messages(value):
    """A LangChain prompt (or a plain string) as [{"role", "content"}] for any chat API."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages = value.to_messages() if hasattr(value, "to_messages") else value
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    return [
        {
            "role": roles.get(getattr(message, "type", "human"), "user"),
            "content": str(message.content),
        }
        for message in messages
    ]


def _local_llm_target() -> tuple[str, str]:
    """The Ollama address and model, refusing anything that would run outside the server."""
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
    return base_url, model


def local_llm_status() -> dict:
    """Can the "All local" profile answer? Asks Ollama which models it has (2 s at most).

    problem is None when ready, else "remote_not_allowed", "cloud_model", "unreachable" or
    "missing" (the model is not installed; installed lists what is).
    """
    status = {"ready": False, "problem": None, "model": os.environ.get("BCT_LOCAL_LLM_MODEL", "qwen3.5:9b-q4_K_M"),
              "installed": []}
    try:
        base_url, model = _local_llm_target()
    except ValueError:
        status["problem"] = "cloud_model" if status["model"].casefold().endswith(":cloud") else "remote_not_allowed"
        return status
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        response.raise_for_status()
        status["installed"] = sorted(item.get("name", "") for item in response.json().get("models") or [])
    except (requests.RequestException, ValueError):
        status["problem"] = "unreachable"
        return status
    if model not in status["installed"] and f"{model}:latest" not in status["installed"]:
        status["problem"] = "missing"
        return status
    status["ready"] = True
    return status


def _create_ollama_llm():
    base_url, model = _local_llm_target()

    def invoke(value):
        # Qwen3.5 and similar Ollama thinking models leave message.content empty
        # unless think is disabled; keep that off by default for structured routing.
        payload = {
            "model": model,
            "messages": _chat_messages(value),
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
        # An empty reply is an unusable draft for the answer ladder, not a server outage.
        return AIMessage(content=content if isinstance(content, str) else "")

    return RunnableLambda(invoke)


# The answer model, chosen by the administrator. Every model call (routing, evidence selection,
# drafting, titles) goes through create_llm(answer_provider()), so one key is enough.
# provider: (key variable, model variable, default model)
ANSWER_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "BCT_OPENAI_MODEL", "gpt-6.1-sol"),
    "anthropic": ("ANTHROPIC_API_KEY", "BCT_ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "groq": ("GROQ_API_KEY", "BCT_GROQ_MODEL", "openai/gpt-oss-120b"),
    # Last: a Gemini key may be there only to read scanned pages (cloud profile).
    "gemini": ("GEMINI_API_KEY", "BCT_GEMINI_ANSWER_MODEL", "gemini-3.8-flash"),
}


def answer_provider() -> str:
    """BCT_ANSWER_PROVIDER, else the first provider (in the order above) that has a key."""
    chosen = (os.environ.get("BCT_ANSWER_PROVIDER") or "").strip().casefold()
    if chosen:
        return chosen
    for name, (key, _model, _default) in ANSWER_PROVIDERS.items():
        if (os.environ.get(key) or "").strip():
            return name
    return "groq"


def _model(provider: str) -> str:
    _key, variable, default = ANSWER_PROVIDERS[provider]
    return (os.environ.get(variable) or "").strip() or default


def _api_key(provider: str) -> str:
    key = (os.environ.get(ANSWER_PROVIDERS[provider][0]) or "").strip()
    if not key:
        raise RuntimeError(f"{ANSWER_PROVIDERS[provider][0]} is required for the {provider} answer model")
    return key


def _split_system(value) -> tuple[str, list[dict]]:
    """System instructions apart (each API takes them in their own field), then the turns."""
    messages = _chat_messages(value)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def _create_openai_llm():
    from openai import OpenAI

    client = OpenAI(api_key=_api_key("openai"), timeout=_TIMEOUT)
    model = _model("openai")
    effort = os.environ.get("BCT_OPENAI_REASONING_EFFORT", "low").strip()

    def invoke(value):
        system, turns = _split_system(value)
        response = client.responses.create(
            model=model,
            instructions=system or None,
            input=turns,
            max_output_tokens=_MAX_TOKENS,
            **({"reasoning": {"effort": effort}} if effort else {}),
        )
        return AIMessage(content=response.output_text or "")

    return RunnableLambda(invoke)


def _create_anthropic_llm():
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key("anthropic"), timeout=_TIMEOUT)
    model = _model("anthropic")
    # Routing, picking evidence and writing a short grounded answer: low effort keeps each of
    # the 3-4 calls per question fast. Claude's newest models take no temperature setting.
    effort = os.environ.get("BCT_ANTHROPIC_EFFORT", "low").strip()

    def invoke(value):
        system, turns = _split_system(value)
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            messages=turns,
            **({"system": system} if system else {}),
            output_config={"effort": effort},
            # A declined request is re-run on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request ({getattr(response.stop_details, 'category', None)})")
        return AIMessage(content="".join(block.text for block in response.content if block.type == "text"))

    return RunnableLambda(invoke)


def _create_gemini_llm():
    from google import genai
    from google.genai import types

    # Like the OpenAI and Anthropic clients (2 retries by default), retry "high demand" (503) and
    # rate limits (429) instead of failing the question at once. Timeout is in milliseconds here.
    client = genai.Client(api_key=_api_key("gemini"), http_options=types.HttpOptions(
        timeout=int(_TIMEOUT * 1000),
        retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2, http_status_codes=[429, 500, 503]),
    ))
    model = _model("gemini")
    thinking = os.environ.get("BCT_GEMINI_THINKING_LEVEL", "low").strip()

    def invoke(value):
        system, turns = _split_system(value)
        response = client.models.generate_content(
            model=model,
            contents=[{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                      for m in turns],
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # no tools here
                max_output_tokens=_MAX_TOKENS,
                **({"thinking_config": types.ThinkingConfig(thinking_level=thinking)} if thinking else {}),
            ),
        )
        return AIMessage(content=response.text or "")

    return RunnableLambda(invoke)


def _groq_api_keys():
    """Collect GROQ_API_KEY plus every GROQ_API_KEY_<n>, de-duplicated."""
    keys: list[str] = []
    primary = (os.environ.get("GROQ_API_KEY") or "").strip()
    if primary:
        keys.append(primary)
    numbered: list[tuple[int, str]] = []
    for name, raw in os.environ.items():
        if not name.startswith("GROQ_API_KEY_"):
            continue
        suffix = name.removeprefix("GROQ_API_KEY_")
        if not suffix.isdigit():
            continue
        value = (raw or "").strip()
        if value:
            numbered.append((int(suffix), value))
    for _, value in sorted(numbered):
        if value not in keys:
            keys.append(value)
    return keys


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
            # HTTP 413 "Request too large ... tokens per minute": the per-key minute budget, so
            # another key can still take the request.
            "413",
            "rate_limit_exceeded",
        )
    )


def _chat_groq(api_key: str) -> ChatGroq:
    return ChatGroq(
        model=_model("groq"),
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
                    logger.warning("Groq key %d of %d rate-limited, trying the next one: %s",
                                   (start + offset) % len(self._keys) + 1, len(self._keys), str(error)[:200])
                    continue
                logger.warning("Groq call failed (%d key(s) configured): %s", len(self._keys), str(error)[:300])
                raise
        raise RuntimeError(f"Groq unavailable after trying configured keys: {last_error}")


_FACTORIES = {
    "ollama": _create_ollama_llm,
    "openai": _create_openai_llm,
    "anthropic": _create_anthropic_llm,
    "gemini": _create_gemini_llm,
    "groq": _RotatingGroqLLM,
}


@lru_cache(maxsize=5)
def create_llm(provider: str):
    """The chat model for one provider (cached; admin.set_secrets clears the cache on key changes)."""
    if provider not in _FACTORIES:
        raise ValueError(f"Unsupported answer provider: {provider}")
    return _FACTORIES[provider]()
