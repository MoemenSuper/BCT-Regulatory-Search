import json
import os
from urllib.request import Request, urlopen

import gradio as gr


API_URL = os.environ.get("BCT_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.environ.get("BCT_API_TIMEOUT_SECONDS", "120"))
UNAVAILABLE_MESSAGE = (
    "The regulatory search service is temporarily unavailable. "
    "Your conversation was not changed; please try again."
)


def _post_chat(payload):
    request = Request(
        f"{API_URL}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _render_sources(sources):
    if not sources:
        return ""

    lines = []
    for source in sources:
        filename = source.get("file")
        page = source.get("page")
        if filename and page is not None:
            lines.append(f"- {filename}, page {page}")
    return "\n\nSources:\n" + "\n".join(lines) if lines else ""


def chat_with_api(message, _history, conversation_id):
    payload = {"question": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        result = _post_chat(payload)
        answer = result["answer"]
        resumed_id = result["conversation_id"]
        if not isinstance(answer, str) or not isinstance(resumed_id, str):
            raise ValueError("Malformed chat response")
        rendered_answer = answer + _render_sources(result.get("sources", []))
        return rendered_answer, resumed_id
    except Exception:
        return UNAVAILABLE_MESSAGE, conversation_id


def build_demo():
    with gr.Blocks(title="BCT Regulatory Search") as demo:
        conversation_id = gr.BrowserState(
            None,
            storage_key="bct-regulatory-search-conversation-id",
        )
        chatbot = gr.Chatbot()
        new_conversation = gr.Button("New conversation")
        gr.ChatInterface(
            fn=chat_with_api,
            chatbot=chatbot,
            additional_inputs=[conversation_id],
            additional_outputs=[conversation_id],
            save_history=True,
        )
        new_conversation.click(
            lambda: ([], None),
            outputs=[chatbot, conversation_id],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue().launch()
