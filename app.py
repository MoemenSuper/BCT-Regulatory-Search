from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from embedding import create_embedding_model
from vector_store import load_vector_store
from reranker import create_reranker
from pydantic import BaseModel, Field, field_validator
from conversation import chat
from conversation_memory import open_conversation_store
import logging
from bm25 import load_documents_from_chroma, create_bm25
from regulatory_graph.runtime import open_relationship_graph_runtime


logger = logging.getLogger(__name__)
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned_question = value.strip()

        if not cleaned_question:
            raise ValueError("Question must not be blank.")

        return cleaned_question

class Source(BaseModel):
    file: str
    page: int | str | None
    score: float

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[Source]
    graph_trace: dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, before the first request
    embedding_model = create_embedding_model()
    app.state.vector_store = load_vector_store(embedding_model)
    app.state.reranker = create_reranker()

    documents = load_documents_from_chroma(app.state.vector_store)

    app.state.bm25_documents = documents
    app.state.bm25 = create_bm25(documents)
    graph_runtime = open_relationship_graph_runtime()
    conversation_store = open_conversation_store()
    app.state.conversation_store = conversation_store
    app.state.graph_retriever = (
        graph_runtime.retriever if graph_runtime is not None else None
    )
    try:
        yield
    finally:
        if graph_runtime is not None:
            graph_runtime.close()
        conversation_store.close()
app = FastAPI(title="BCT Regulatory Search API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def post_chat(payload: ChatRequest, request: Request):
    conversation_store = request.app.state.conversation_store
    if payload.conversation_id is None:
        conversation_id = conversation_store.create()
    else:
        conversation_id = payload.conversation_id

    memory_state = conversation_store.load(conversation_id)
    if memory_state is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        result = chat(
            payload.question,
            memory_state,
            request.app.state.vector_store,
            request.app.state.reranker,
            request.app.state.bm25,
            request.app.state.bm25_documents,
            graph_retriever=request.app.state.graph_retriever,
        )
    except Exception:
        logger.exception("Chat request failed.")
        raise HTTPException(status_code=500, detail="Chat service failed.")

    conversation_store.save(conversation_id, result["memory_state"])
    return {
        "conversation_id": conversation_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "graph_trace": result["graph_trace"],
    }
