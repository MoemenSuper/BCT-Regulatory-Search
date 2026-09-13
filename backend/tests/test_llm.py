
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
    captured = {}
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY_2", "configured-key")
    monkeypatch.setattr(
        llm,
        "ChatGroq",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    llm.create_llm.cache_clear()

    try:
        llm.create_llm("groq")
    finally:
        llm.create_llm.cache_clear()

    assert captured["groq_api_key"] == "configured-key"
