from urllib.error import URLError

import gradio_app


def test_first_ui_message_creates_a_backend_conversation(monkeypatch):
    received = {}

    def fake_post(payload):
        received.update(payload)
        return {
            "conversation_id": "session-1",
            "answer": "The deadline is 30 days.",
            "sources": [],
            "graph_trace": {"status": "NOT_REQUESTED"},
        }

    monkeypatch.setattr(gradio_app, "_post_chat", fake_post)

    answer, conversation_id = gradio_app.chat_with_api("What is the deadline?", [], None)

    assert received == {"question": "What is the deadline?"}
    assert answer == "The deadline is 30 days."
    assert conversation_id == "session-1"


def test_follow_up_resumes_the_same_backend_conversation(monkeypatch):
    received = {}

    def fake_post(payload):
        received.update(payload)
        return {
            "conversation_id": "session-1",
            "answer": "It was amended in 2020.",
            "sources": [
                {"file": "Circular_2020.pdf", "page": 3, "score": 0.91}
            ],
            "graph_trace": {"status": "APPLIED"},
        }

    monkeypatch.setattr(gradio_app, "_post_chat", fake_post)

    answer, conversation_id = gradio_app.chat_with_api(
        "Was it amended later?", [], "session-1"
    )

    assert received == {
        "question": "Was it amended later?",
        "conversation_id": "session-1",
    }
    assert "Circular_2020.pdf, page 3" in answer
    assert conversation_id == "session-1"


def test_ui_failure_is_safe_and_preserves_the_session_id(monkeypatch):
    monkeypatch.setattr(
        gradio_app,
        "_post_chat",
        lambda _payload: (_ for _ in ()).throw(URLError("offline")),
    )

    answer, conversation_id = gradio_app.chat_with_api(
        "And the next rule?", [], "session-1"
    )

    assert "temporarily unavailable" in answer
    assert conversation_id == "session-1"


def test_ui_rejects_a_malformed_backend_response(monkeypatch):
    monkeypatch.setattr(
        gradio_app,
        "_post_chat",
        lambda _payload: {"answer": "unsupported answer"},
    )

    answer, conversation_id = gradio_app.chat_with_api("Question", [], None)

    assert "temporarily unavailable" in answer
    assert conversation_id is None
