"""One key is enough: the answer model is OpenAI, Claude, Gemini or Groq, chosen by the administrator.

The SDK clients are replaced by fakes: these tests check what each provider is sent (system
instructions apart, the model, no settings a provider rejects) and how its reply is read.
"""
from types import SimpleNamespace

import pytest
from langchain_core.prompts import ChatPromptTemplate

from rag import llm

PROMPT = ChatPromptTemplate.from_messages([("system", "Answer from the evidence only."), ("human", "{q}")])


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for name in [n for n in llm.os.environ if n.endswith("_API_KEY") or n.startswith("GROQ_API_KEY")]:
        monkeypatch.delenv(name)
    for name in ("BCT_ANSWER_PROVIDER", "BCT_OPENAI_MODEL", "BCT_ANTHROPIC_MODEL", "BCT_GEMINI_ANSWER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    llm.create_llm.cache_clear()
    yield
    llm.create_llm.cache_clear()


def test_the_provider_with_a_key_answers_and_the_admin_choice_wins(monkeypatch):
    assert llm.answer_provider() == "groq"  # nothing configured: the historical default
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert llm.answer_provider() == "gemini"
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert llm.answer_provider() == "groq"  # a Gemini key may only be there to read scanned pages
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert llm.answer_provider() == "openai"
    monkeypatch.setenv("BCT_ANSWER_PROVIDER", "Anthropic")
    assert llm.answer_provider() == "anthropic"


def test_openai_gets_instructions_apart_and_its_default_model(monkeypatch):
    sent = {}

    class Responses:
        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(output_text="Réponse OpenAI")

    monkeypatch.setattr("openai.OpenAI", lambda **_: SimpleNamespace(responses=Responses()))
    monkeypatch.setenv("OPENAI_API_KEY", "o")

    reply = (PROMPT | llm.create_llm("openai")).invoke({"q": "Quel ratio ?"})

    assert reply.content == "Réponse OpenAI"
    assert sent["model"] == "gpt-6.1-sol"
    assert sent["instructions"] == "Answer from the evidence only."
    assert sent["input"] == [{"role": "user", "content": "Quel ratio ?"}]


def test_claude_gets_no_temperature_and_a_refusal_is_an_error(monkeypatch):
    sent = {}
    reply_blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="Réponse Claude")]
    stop = {"reason": "end_turn"}

    class Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(content=reply_blocks, stop_reason=stop["reason"],
                                   stop_details=SimpleNamespace(category="general_harms"))

    monkeypatch.setattr("anthropic.Anthropic", lambda **_: SimpleNamespace(beta=SimpleNamespace(messages=Messages())))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    model = llm.create_llm("anthropic")

    assert (PROMPT | model).invoke({"q": "Quel ratio ?"}).content == "Réponse Claude"
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["system"] == "Answer from the evidence only."
    assert sent["messages"] == [{"role": "user", "content": "Quel ratio ?"}]
    assert "temperature" not in sent and sent["output_config"] == {"effort": "low"}

    stop["reason"] = "refusal"
    with pytest.raises(RuntimeError, match="declined"):
        (PROMPT | model).invoke({"q": "Quel ratio ?"})


def test_gemini_gets_a_system_instruction_and_model_turns(monkeypatch):
    sent = {}

    class Models:
        def generate_content(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(text="Réponse Gemini")

    monkeypatch.setattr("google.genai.Client", lambda **_: SimpleNamespace(models=Models()))
    monkeypatch.setenv("GEMINI_API_KEY", "g")

    reply = llm.create_llm("gemini").invoke("Titre ?")  # a plain string, like the title call

    assert reply.content == "Réponse Gemini"
    assert sent["model"] == "gemini-3.8-flash"
    assert sent["contents"] == [{"role": "user", "parts": [{"text": "Titre ?"}]}]
    assert sent["config"].system_instruction is None


def test_a_missing_key_says_which_one(monkeypatch):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.create_llm("openai")


def test_the_admin_can_only_pick_a_known_provider(tmp_path):
    from extras.app_settings import AppSettingsStore

    store = AppSettingsStore(tmp_path / "settings.sqlite3")
    try:
        with pytest.raises(ValueError, match="BCT_ANSWER_PROVIDER"):
            store.update_secrets({"BCT_ANSWER_PROVIDER": "chatgpt"})
        assert store.update_secrets({"BCT_ANSWER_PROVIDER": "OpenAI"})["answer_provider"] == "openai"
    finally:
        store.update_secrets({"BCT_ANSWER_PROVIDER": None})
        store.close()
