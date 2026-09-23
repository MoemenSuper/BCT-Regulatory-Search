
import llm
from langchain_core.messages import HumanMessage
import pytest


def test_local_answer_provider_uses_configurable_loopback_ollama(monkeypatch):
    received = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "local answer"}}

    def fake_post(url, **kwargs):
        received["url"] = url
        received.update(kwargs)
        return Response()

    monkeypatch.setenv("BCT_LOCAL_LLM_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("BCT_LOCAL_LLM_MODEL", "future-bct-model:27b")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    llm.create_llm.cache_clear()

    model = llm.create_llm("ollama")
    response = model.invoke([HumanMessage(content="Question")])

    assert response.content == "local answer"
    assert received["url"] == "http://127.0.0.1:11434/api/chat"
    assert received["json"]["model"] == "future-bct-model:27b"
    assert received["json"]["stream"] is False
    assert received["json"]["think"] is False


def test_local_answer_provider_rejects_a_remote_endpoint_by_default(monkeypatch):
    monkeypatch.setenv("BCT_LOCAL_LLM_URL", "http://192.0.2.1:11434")
    monkeypatch.delenv("BCT_ALLOW_REMOTE_LOCAL_LLM", raising=False)
    llm.create_llm.cache_clear()

    try:
        with pytest.raises(ValueError, match="loopback"):
            llm.create_llm("ollama")
    finally:
        llm.create_llm.cache_clear()


def test_groq_provider_accepts_the_numbered_key_configuration(monkeypatch):
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.kwargs = kwargs

        def invoke(self, *_args, **_kwargs):
            return "ok"

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for index in range(3, 20):
        monkeypatch.delenv(f"GROQ_API_KEY_{index}", raising=False)
    monkeypatch.setenv("GROQ_API_KEY_2", "configured-key")
    monkeypatch.setattr(llm, "ChatGroq", FakeClient)
    llm.create_llm.cache_clear()

    try:
        model = llm.create_llm("groq")
    finally:
        llm.create_llm.cache_clear()

    assert len(captured) == 1
    assert captured[0]["groq_api_key"] == "configured-key"
    assert model.invoke([HumanMessage(content="hi")]) == "ok"


def test_groq_provider_loads_keys_beyond_slot_seven(monkeypatch):
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append(kwargs["groq_api_key"])

        def invoke(self, *_args, **_kwargs):
            return "ok"

    monkeypatch.setenv("GROQ_API_KEY", "key-1")
    for index in range(2, 14):
        monkeypatch.setenv(f"GROQ_API_KEY_{index}", f"key-{index}")
    monkeypatch.setattr(llm, "ChatGroq", FakeClient)
    llm.create_llm.cache_clear()

    try:
        model = llm.create_llm("groq")
    finally:
        llm.create_llm.cache_clear()

    assert captured == [f"key-{index}" for index in range(1, 14)]
    assert len(model._keys) == 13


def test_groq_provider_rotates_across_keys_on_rate_limit(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.key = kwargs["groq_api_key"]

        def invoke(self, *_args, **_kwargs):
            calls.append(self.key)
            if self.key == "key-a":
                raise RuntimeError("Error code: 429 - rate limit exceeded")
            return f"from-{self.key}"

    monkeypatch.setenv("GROQ_API_KEY", "key-a")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-b")
    for index in range(3, 20):
        monkeypatch.delenv(f"GROQ_API_KEY_{index}", raising=False)
    monkeypatch.setattr(llm, "ChatGroq", FakeClient)
    llm.create_llm.cache_clear()

    try:
        model = llm.create_llm("groq")
        result = model.invoke([HumanMessage(content="hi")])
    finally:
        llm.create_llm.cache_clear()

    assert result == "from-key-b"
    assert calls == ["key-a", "key-b"]
