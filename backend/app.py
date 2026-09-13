from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import re
import secrets
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from conversation import chat
from conversation_memory import open_conversation_store, summarize_conversation_title
from llm import create_llm
from runtime_profiles import RuntimeProfile, RuntimeProfileManager, parse_profile, profile_options
from runtime_retrieval import LocalRetrievalBackend, create_voyage_backend_from_environment
from source_documents import SourceDocumentResolver, render_page_png, source_info


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile: str = Field(default_factory=lambda: os.environ.get("BCT_DEFAULT_PROFILE", RuntimeProfile.CLOUD.value))

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be blank.")
        return cleaned

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        return parse_profile(value).value


class Source(BaseModel):
    file: str
    page: int | str | None
    score: float
    excerpt: str = ""


class ChatResponse(BaseModel):
    conversation_id: str
    profile: str
    status: str
    answer: str
    sources: list[Source]
    graph_trace: dict
    memory_state: dict[str, Any]


def create_runtime_profile_manager(local_backend=None):
    return RuntimeProfileManager(
        local_backend,
        create_voyage_backend_from_environment,
        local_retrieval_factory=create_local_backend,
    )


def create_local_backend():
    from embedding import create_embedding_model
    from vector_store import load_vector_store
    from reranker import create_reranker
    from bm25 import load_documents_from_chroma, create_bm25

    embedding_model = create_embedding_model()
    vector_store = load_vector_store(embedding_model)
    reranker = create_reranker()
    documents = load_documents_from_chroma(vector_store)
    bm25 = create_bm25(documents)

    ocr_vector_store = None
    ocr_documents = []
    ocr_bm25 = None
    if os.environ.get("BCT_OCR_CHROMA_DB"):
        ocr_vector_store = load_vector_store(
            embedding_model,
            persist_directory=os.environ["BCT_OCR_CHROMA_DB"],
            collection_name=os.environ.get("BCT_OCR_CHROMA_COLLECTION", "bct_arabic_ocr_secondary_v1"),
        )
        ocr_documents = load_documents_from_chroma(ocr_vector_store)
        ocr_bm25 = create_bm25(ocr_documents)
    return LocalRetrievalBackend(
        vector_store,
        reranker,
        bm25,
        documents,
        ocr_vector_store=ocr_vector_store,
        ocr_bm25=ocr_bm25,
        ocr_documents=ocr_documents,
    )


def open_relationship_graph_runtime():
    from regulatory_graph_lite.runtime import open_relationship_graph_runtime as open_runtime

    return open_runtime()


def _graph_enabled() -> bool:
    return os.environ.get("BCT_ENABLE_GRAPH") == "1"


def _neo4j_connected(graph_runtime) -> bool:
    if graph_runtime is None:
        return False
    verify = getattr(getattr(graph_runtime, "driver", None), "verify_connectivity", None)
    if verify is None:
        return True
    try:
        verify()
    except Exception:
        return False
    return True


def graph_lite_status(graph_runtime=None, graph_retriever=None) -> dict[str, bool]:
    enabled = _graph_enabled()
    connected = _neo4j_connected(graph_runtime)
    return {
        "graph_enabled": enabled,
        "neo4j_connected": connected,
        "graph_ready": bool(enabled and connected and graph_retriever is not None),
    }


def announce_graph_lite_status(status: dict[str, bool]) -> None:
    banner = (
        "Graph Lite\n"
        f"  graph_enabled: {status['graph_enabled']}\n"
        f"  neo4j_connected: {status['neo4j_connected']}\n"
        f"  graph_ready: {status['graph_ready']}"
    )
    print(banner, flush=True)
    if status["graph_enabled"] and not status["graph_ready"]:
        logger.warning("Graph Lite is enabled but not ready; ordinary RAG remains active.")
    else:
        logger.info(
            "Graph Lite ready=%s enabled=%s connected=%s",
            status["graph_ready"],
            status["graph_enabled"],
            status["neo4j_connected"],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.profile_manager = create_runtime_profile_manager()
    graph_runtime = open_relationship_graph_runtime() if _graph_enabled() else None
    conversation_store = open_conversation_store()
    app.state.conversation_store = conversation_store
    app.state.graph_runtime = graph_runtime
    app.state.graph_retriever = (
        graph_runtime.retriever if graph_runtime is not None else None
    )
    app.state.source_resolver = SourceDocumentResolver()
    announce_graph_lite_status(
        graph_lite_status(graph_runtime, app.state.graph_retriever)
    )
    try:
        yield
    finally:
        if graph_runtime is not None:
            graph_runtime.close()


app = FastAPI(title="BCT Regulatory Search API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request):
    runtime = getattr(request.app.state, "graph_runtime", None)
    retriever = getattr(request.app.state, "graph_retriever", None)
    return {"status": "ok", **graph_lite_status(runtime, retriever)}


@app.get("/profiles")
def profiles():
    return profile_options()


_QUESTION_SPLIT = re.compile(r"(?<=[?؟])\s+(?=\S)")


def _sub_questions(text: str) -> list[str]:
    """Run a multi-question message as consecutive turns so every per-question
    gate (retrieval budget, instrument identity, temporal flag) applies once per question.
    ponytail: punctuation-only split; "A et B ?" stays one query. Capped at 3 turns."""
    parts = [part.strip() for part in _QUESTION_SPLIT.split(text) if len(part.strip()) >= 12]
    return parts[:3] or [text]


_TITLE_PROMPT = (
    "Résume la question réglementaire ci-dessous en un titre de 3 à 7 mots, dans la même "
    "langue que la question, sans guillemets ni ponctuation finale. Réponds uniquement par le titre.\n\n"
)


def _conversation_title(question: str, provider: str) -> str:
    """One short LLM call on a conversation's first turn; falls back to the heuristic."""
    try:
        reply = create_llm(provider).invoke(_TITLE_PROMPT + question[:600])
        title = " ".join(str(reply.content).strip().strip("\"'«»").split())
        if 3 <= len(title) <= 80:
            return title
    except Exception:
        logger.info("title_llm_unavailable provider=%s", provider)
    return summarize_conversation_title(question, max_length=56)


@app.post("/chat", response_model=ChatResponse)
def post_chat(payload: ChatRequest, request: Request):
    try:
        runtime = request.app.state.profile_manager.get(payload.profile)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Runtime profile is unavailable.")
        raise HTTPException(status_code=503, detail="Selected runtime is unavailable.")

    store = request.app.state.conversation_store
    conversation_id = store.create() if payload.conversation_id is None else payload.conversation_id
    memory_state = store.load(conversation_id)
    if memory_state is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    for question in _sub_questions(payload.question):
        try:
            result = chat(
                question,
                memory_state,
                graph_retriever=request.app.state.graph_retriever,
                retrieval_backend=runtime.retrieval_backend,
                llm_provider=runtime.answer_provider,
            )
        except (OSError, RuntimeError, ValueError):
            logger.exception("Runtime profile is unavailable.")
            raise HTTPException(status_code=503, detail="Selected runtime is unavailable.")
        except Exception:
            logger.exception("Chat request failed.")
            raise HTTPException(status_code=500, detail="Chat service failed.")

        memory_state = result["memory_state"]
        store.save_with_turn(
            conversation_id,
            memory_state,
            question=question,
            standalone_query=(memory_state.get("turns") or [{}])[-1].get("standalone_query", question),
            answer=result["answer"],
            sources=result["sources"],
            graph_trace=result["graph_trace"],
            profile=runtime.spec.value.value,
            answer_status=result.get("status"),
        )
    if payload.conversation_id is None:
        store.rename(conversation_id, _conversation_title(payload.question, runtime.answer_provider))
    return {
        "conversation_id": conversation_id,
        "profile": runtime.spec.value.value,
        "status": result.get("status", "answered"),
        "answer": result["answer"],
        "sources": result["sources"],
        "graph_trace": result["graph_trace"],
        "memory_state": result["memory_state"],
    }



@app.get("/conversations")
def list_conversations(request: Request, limit: int = Query(default=100, ge=1, le=500)):
    return request.app.state.conversation_store.list_conversations(limit=limit)


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, request: Request):
    if not conversation_id or len(conversation_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid conversation identifier.")
    transcript = request.app.state.conversation_store.transcript(conversation_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return transcript


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: RenameRequest, request: Request):
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be blank.")
    try:
        request.app.state.conversation_store.rename(conversation_id, title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "title": title}


@app.delete("/conversations/{conversation_id}", status_code=204, response_class=Response)
def delete_conversation(conversation_id: str, request: Request):
    try:
        request.app.state.conversation_store.delete(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.")


def _resolve_source(filename: str, request: Request):
    try:
        return request.app.state.source_resolver.resolve(filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source filename.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source PDF is not available on this machine.")


@app.get("/sources/{filename}/info")
def get_source_info(filename: str, request: Request):
    try:
        return source_info(_resolve_source(filename, request))
    except RuntimeError:
        logger.exception("Source viewer is unavailable.")
        raise HTTPException(status_code=503, detail="Source viewer is unavailable.")


@app.get("/sources/{filename}/page/{page_number}.png")
def get_source_page(
    filename: str,
    page_number: int,
    request: Request,
    quote: str = Query(default="", max_length=2000),
    scale: float = Query(default=1.8, ge=0.75, le=3.0),
):
    try:
        payload, headers = render_page_png(
            _resolve_source(filename, request),
            page_number=page_number,
            quote=quote,
            scale=scale,
        )
        return Response(content=payload, media_type="image/png", headers=headers)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source request.")
    except IndexError:
        raise HTTPException(status_code=404, detail="Source page does not exist.")
    except RuntimeError:
        logger.exception("Source viewer is unavailable.")
        raise HTTPException(status_code=503, detail="Source viewer is unavailable.")


@app.get("/sources/{filename}")
def get_source_pdf(filename: str, request: Request):
    document = _resolve_source(filename, request)
    return FileResponse(
        document.path,
        media_type="application/pdf",
        filename=document.filename,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=300"},
    )

def _refresh_runtime_asset_environment() -> None:
    from ingestion.index import configure_runtime_assets

    root_value = os.environ.get("BCT_RUNTIME_ASSET_ROOT")
    if not root_value:
        raise RuntimeError("BCT_RUNTIME_ASSET_ROOT is not configured")
    configure_runtime_assets(root_value)


def _install_ingestion_routes(target: FastAPI) -> None:
    # Importing File/Form performs python-multipart validation, so keep the whole
    # administrator upload feature optional at process startup.
    from fastapi import File, Form, Header, UploadFile
    from starlette.concurrency import run_in_threadpool

    configured_token = os.environ.get("BCT_INGESTION_TOKEN")
    if not configured_token:
        raise RuntimeError(
            "BCT_ENABLE_INGESTION=1 requires BCT_INGESTION_TOKEN; "
            "use the ingest.py CLI if you do not need the HTTP administrator endpoint"
        )

    def require_admin(authorization: str | None) -> None:
        expected = f"Bearer {configured_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Administrator authorization required.")

    @target.post("/documents")
    async def ingest_document(
        request: Request,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        publication_date: str | None = Form(default=None),
        document_type: str | None = Form(default=None),
        category: str | None = Form(default=None),
        document_number: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
    ):
        require_admin(authorization)
        max_bytes = int(os.environ.get("BCT_MAX_PDF_BYTES", str(50 * 1024 * 1024)))
        suffix = ".pdf"
        temporary_path = None
        size = 0
        try:
            with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temporary_path = Path(handle.name)
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(status_code=413, detail="PDF exceeds the configured size limit.")
                    handle.write(chunk)

            from ingestion.pipeline import IngestionConfig, IngestionPipeline

            config = IngestionConfig.from_environment()
            pipeline = IngestionPipeline(config)
            try:
                report = await run_in_threadpool(
                    pipeline.ingest,
                    temporary_path,
                    original_filename=file.filename or "document.pdf",
                    metadata={
                        key: value
                        for key, value in {
                            "title": title,
                            "publication_date": publication_date,
                            "type": document_type,
                            "category": category,
                            "document_number": document_number,
                        }.items()
                        if value is not None
                    },
                )
            finally:
                pipeline.close()
            _refresh_runtime_asset_environment()
            request.app.state.profile_manager.reset()
            request.app.state.source_resolver.refresh()
            return report
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("Document ingestion failed.")
            raise HTTPException(status_code=422, detail="Document ingestion failed.")
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @target.get("/documents")
    def list_documents(limit: int = 100, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        from ingestion.pipeline import IngestionConfig
        from ingestion.registry import IngestionRegistry

        config = IngestionConfig.from_environment()
        registry = IngestionRegistry(config.registry_path)
        try:
            return registry.list_ready(limit=limit)
        finally:
            registry.close()


if os.environ.get("BCT_ENABLE_INGESTION") == "1":
    _install_ingestion_routes(app)
