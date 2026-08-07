from contextlib import asynccontextmanager
from fastapi import FastAPI
from embedding import create_embedding_model
from vector_store import load_vector_store
from reranker import create_reranker

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