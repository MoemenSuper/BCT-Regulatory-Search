from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from embedding import create_embedding_model
from vector_store import load_vector_store
from reranker import create_reranker
from pydantic import BaseModel, Field
from conversation import chat
import logging


logger = logging.getLogger(__name__)
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class Source(BaseModel):
    file: str
    page: int | str | None
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, before the first request
    embedding_model = create_embedding_model()
    app.state.vector_store = load_vector_store(embedding_model)
    app.state.reranker = create_reranker()
    yield
    # shutdown — nothing to clean up yet

app = FastAPI(title="BCT Regulatory Search API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def post_chat(payload: ChatRequest, request: Request):
    empty_memory = {"topics": [], "first_topic": None, "current_topic": None}

    try:
        result = chat(
            payload.question,
            empty_memory,
            request.app.state.vector_store,
            request.app.state.reranker,
        )
    except Exception:
        logger.exception("Chat request failed.")
        raise HTTPException(status_code=500, detail="Chat service failed.")

    return {"answer": result["answer"], "sources": result["sources"]}