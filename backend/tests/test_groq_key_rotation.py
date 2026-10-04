"""A per-key Groq limit (429, or 413 'request too large' on the minute budget) moves to the next key."""

import pytest

import llm


class _Client:
    def __init__(self, error=None):
        self.error = error

    def invoke(self, value, config=None, **kwargs):
        if self.error:
            raise self.error
        return "ok"


def _rotating(monkeypatch, clients):
    for name in [name for name in llm.os.environ if name.startswith("GROQ_API_KEY")]:
        monkeypatch.delenv(name)
    for index, _client in enumerate(clients, 1):
        monkeypatch.setenv(f"GROQ_API_KEY_{index}", f"key-{index}")
    monkeypatch.setattr(llm, "_chat_groq", lambda key: clients[int(key.split("-")[1]) - 1])
    return llm._RotatingGroqLLM()


def test_request_too_large_moves_to_next_key(monkeypatch):
    too_large = RuntimeError("Error code: 413 - {'error': {'code': 'rate_limit_exceeded'}}")
    model = _rotating(monkeypatch, [_Client(too_large), _Client()])

    assert model.invoke("question") == "ok"


def test_invalid_key_is_not_retried(monkeypatch):
    model = _rotating(monkeypatch, [_Client(RuntimeError("Error code: 401 - Invalid API Key")), _Client()])

    with pytest.raises(RuntimeError, match="401"):
        model.invoke("question")
