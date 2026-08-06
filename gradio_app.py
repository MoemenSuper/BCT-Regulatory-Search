import gradio as gr

from conversation import chat
from embedding import create_embedding_model
from vector_store import load_vector_store

embedding_model = create_embedding_model()
vector_store = load_vector_store(embedding_model)

with gr.Blocks() as demo:
    memory_state = gr.State({
        "topics": [],
        "first_topic": None,
        "current_topic": None
    })

    gr.ChatInterface(
        fn=lambda message, history, memory_summary: chat(
            message, history, memory_summary, vector_store
        ),
        additional_inputs=[memory_state],
        additional_outputs=[memory_state],
        save_history=True,
    )

if __name__ == "__main__":
    demo.queue().launch()