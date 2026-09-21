from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
import csv
import io
from datetime import datetime, timezone

from app_settings import open_app_settings
from conversation import chat
from conversation_memory import open_conversation_store, summarize_conversation_title
from identity import (
    SESSION_COOKIE,
    clear_session_cookie,
    open_auth_store,
    require_admin,
    require_approved_user,
    require_user,
    set_session_cookie,
)
from langchain_core.callbacks import get_usage_metadata_callback

from llm import create_llm
from runtime_profiles import RuntimeProfile, RuntimeProfileManager, parse_profile, profile_options
from runtime_retrieval import (
    LocalRetrievalBackend,
    create_voyage_backend_from_environment,
    track_cloud_retrieval_usage,
)
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


def supersession_status() -> dict[str, object]:
    """JSONL SUPERSEDES index readiness for health / admin overview."""
    try:
        from jsonl_supersession import load_edges, resolve_edges_path
        from ingestion.index import resolve_active_assets

        root_value = os.environ.get("BCT_RUNTIME_ASSET_ROOT") or os.environ.get(
            "BCT_VOYAGE_PROVIDER_ROOT"
        )
        if not root_value:
            return {"ready": False, "edge_count": 0}
        root = Path(root_value)
        active = resolve_active_assets(root) if (root / "ACTIVE.json").exists() else root
        path = resolve_edges_path(active)
        if path is None:
            return {"ready": False, "edge_count": 0}
        count = len(load_edges(path))
        return {"ready": count > 0, "edge_count": count}
    except Exception:
        logger.info("Supersession edges unavailable for status.", exc_info=True)
        return {"ready": False, "edge_count": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth_store = open_auth_store()
    settings_store = open_app_settings()
    auth_store.bootstrap_admin()
    app.state.auth_store = auth_store
    app.state.settings_store = settings_store
    app.state.profile_manager = create_runtime_profile_manager()
    conversation_store = open_conversation_store()
    app.state.conversation_store = conversation_store
    app.state.source_resolver = SourceDocumentResolver()
    status = supersession_status()
    print(
        f"Supersession edges\n  ready: {status['ready']}\n  edge_count: {status['edge_count']}",
        flush=True,
    )
    try:
        yield
    finally:
        closer = getattr(conversation_store, "close", None)
        if callable(closer):
            closer()
        settings_store.close()
        auth_store.close()


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
    return {"status": "ok", "supersession": supersession_status()}


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ProfileSelfUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    avatar_icon: str | None = Field(default=None, max_length=200_000)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ProfileUpdateRequest(BaseModel):
    profile: str

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        return parse_profile(value).value


class CloudRetrievalProviderUpdateRequest(BaseModel):
    provider: str

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        from runtime_retrieval import cloud_embed_spec

        return cloud_embed_spec(value).key


class SecretsUpdateRequest(BaseModel):
    secrets: dict[str, str | None]


class TokenLimitRequest(BaseModel):
    token_limit: int = Field(ge=0, le=100_000_000)


@app.post("/auth/register")
def register(payload: RegisterRequest, request: Request):
    store = request.app.state.auth_store
    try:
        user = store.create_user(email=payload.email, password=payload.password, role="user", status="pending")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "user": user.public_dict(),
        "message": "Registration received. An administrator must approve your account before you can sign in.",
    }


@app.post("/auth/login")
def login(payload: LoginRequest, request: Request, response: Response):
    store = request.app.state.auth_store
    user = store.authenticate(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = store.create_session(user.id)
    set_session_cookie(response, token)
    return {"user": user.public_dict()}


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    request.app.state.auth_store.delete_session(token)
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/auth/me")
def me(user=Depends(require_user)):
    return {"user": user.public_dict()}


@app.post("/auth/profile")
@app.patch("/auth/me")
def update_me(payload: ProfileSelfUpdateRequest, request: Request, user=Depends(require_user)):
    if payload.display_name is None and payload.avatar_icon is None:
        raise HTTPException(status_code=400, detail="No profile fields to update.")
    try:
        updated = request.app.state.auth_store.update_profile(
            user.id,
            display_name=payload.display_name,
            avatar_icon=payload.avatar_icon,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    return {"user": updated.public_dict()}


@app.post("/auth/password")
def change_password(payload: PasswordChangeRequest, request: Request, user=Depends(require_user)):
    try:
        request.app.state.auth_store.change_password(
            user.id,
            current_password=payload.current_password,
            new_password=payload.new_password,
        )
    except PermissionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    return {"ok": True}

@app.get("/profiles")
def profiles(_user=Depends(require_approved_user)):
    return profile_options()


@app.get("/admin/overview")
def admin_overview(request: Request, _admin=Depends(require_admin)):
    users = request.app.state.auth_store.list_users()
    settings = request.app.state.settings_store.public_configuration()
    docs = []
    try:
        from ingestion.pipeline import IngestionConfig
        from ingestion.registry import IngestionRegistry

        config = IngestionConfig.from_environment()
        registry = IngestionRegistry(config.registry_path)
        try:
            docs = registry.list_ready(limit=20)
        finally:
            registry.close()
    except Exception:
        logger.info("Ingestion registry unavailable for overview.")
    refusals_total = 0
    try:
        refusals_total = request.app.state.conversation_store.count_answer_refusals()
    except Exception:
        logger.exception("Answer refusal log unavailable for overview.")
    return {
        "users_total": len(users),
        "users_pending": sum(1 for user in users if user.status == "pending"),
        "users_approved": sum(1 for user in users if user.status == "approved"),
        "users_rejected": sum(1 for user in users if user.status == "rejected"),
        "documents_ready": len(docs),
        "active_profile": settings["active_profile"],
        "answer_refusals_total": refusals_total,
        "supersession": supersession_status(),
    }


@app.get("/admin/answer-refusals")
def admin_answer_refusals(
    request: Request,
    limit: int = Query(default=500, ge=1, le=5000),
    bucket: list[str] | None = Query(default=None),
    reason: list[str] | None = Query(default=None),
    _admin=Depends(require_admin),
):
    store = request.app.state.conversation_store
    buckets = [item for item in (bucket or []) if str(item).strip()]
    reasons = [item for item in (reason or []) if str(item).strip()]
    return {
        "total": store.count_answer_refusals(reasons=reasons, buckets=buckets),
        "total_all": store.count_answer_refusals(),
        "buckets": buckets,
        "reasons": reasons,
        "reason_options": store.distinct_answer_refusal_reasons(),
        "items": store.list_answer_refusals(limit=limit, reasons=reasons, buckets=buckets),
    }


@app.get("/admin/answer-refusals/export")
def admin_export_answer_refusals(
    request: Request,
    bucket: list[str] | None = Query(default=None),
    reason: list[str] | None = Query(default=None),
    _admin=Depends(require_admin),
):
    """Download refusal log as CSV (honours optional reason filters)."""
    store = request.app.state.conversation_store
    buckets = [item for item in (bucket or []) if str(item).strip()]
    reasons = [item for item in (reason or []) if str(item).strip()]
    items = store.list_answer_refusals(limit=100_000, reasons=reasons, buckets=buckets)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "created_at",
        "user_email",
        "user_id",
        "answer_status",
        "profile",
        "question",
        "reason_title",
        "reason",
        "diagnostics",
        "conversation_id",
        "refusal_id",
    ])
    for item in items:
        writer.writerow([
            item.get("created_at") or "",
            item.get("user_email") or "",
            item.get("user_id") or "",
            item.get("answer_status") or "",
            item.get("profile") or "",
            item.get("question") or "",
            item.get("reason_title") or "",
            item.get("reason") or "",
            " | ".join(str(part) for part in (item.get("diagnostics") or [])),
            item.get("conversation_id") or "",
            item.get("refusal_id") or "",
        ])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"answer-refusals-{stamp}.csv"
    payload = buffer.getvalue().encode("utf-8-sig")
    return Response(
        content=payload,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/admin/users")
def admin_list_users(request: Request, _admin=Depends(require_admin)):
    return [user.public_dict() for user in request.app.state.auth_store.list_users()]


@app.post("/admin/users/{user_id}/approve")
def admin_approve_user(user_id: str, request: Request, admin=Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own account status.")
    try:
        user = request.app.state.auth_store.set_status(user_id, "approved")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    return {"user": user.public_dict()}


@app.post("/admin/users/{user_id}/reject")
def admin_reject_user(user_id: str, request: Request, admin=Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own account status.")
    try:
        user = request.app.state.auth_store.set_status(user_id, "rejected")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    return {"user": user.public_dict()}


@app.post("/admin/users/{user_id}/promote")
def admin_promote_user(user_id: str, request: Request, admin=Depends(require_admin)):
    """Grant administrator role. There is intentionally no demote endpoint."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You are already an administrator.")
    try:
        user = request.app.state.auth_store.promote_to_admin(user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"user": user.public_dict()}


@app.put("/admin/users/{user_id}/token-limit")
def admin_set_token_limit(
    user_id: str,
    payload: TokenLimitRequest,
    request: Request,
    _admin=Depends(require_admin),
):
    try:
        user = request.app.state.auth_store.set_token_limit(user_id, payload.token_limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"user": user.public_dict()}


@app.post("/admin/users/{user_id}/reset-tokens")
def admin_reset_token_usage(user_id: str, request: Request, _admin=Depends(require_admin)):
    try:
        user = request.app.state.auth_store.reset_token_usage(user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    return {"user": user.public_dict()}


@app.delete("/admin/users/{user_id}", status_code=204, response_class=Response)
def admin_delete_user(user_id: str, request: Request, admin=Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")
    try:
        request.app.state.auth_store.delete_user(user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    except PermissionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/admin/config")
def admin_get_config(request: Request, _admin=Depends(require_admin)):
    return request.app.state.settings_store.public_configuration()


@app.put("/admin/config/profile")
def admin_set_profile(payload: ProfileUpdateRequest, request: Request, _admin=Depends(require_admin)):
    profile = request.app.state.settings_store.set_active_profile(payload.profile)
    request.app.state.profile_manager.reset()
    return {"active_profile": profile.value}


@app.put("/admin/config/cloud-retrieval-provider")
def admin_set_cloud_retrieval_provider(
    payload: CloudRetrievalProviderUpdateRequest,
    request: Request,
    _admin=Depends(require_admin),
):
    provider = request.app.state.settings_store.set_cloud_retrieval_provider(payload.provider)
    request.app.state.profile_manager.reset()
    return {
        "cloud_retrieval_provider": provider,
        "config": request.app.state.settings_store.public_configuration(),
    }


@app.put("/admin/config/secrets")
def admin_set_secrets(payload: SecretsUpdateRequest, request: Request, _admin=Depends(require_admin)):
    try:
        config = request.app.state.settings_store.update_secrets(payload.secrets)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    request.app.state.profile_manager.reset()
    # Cached Groq/Ollama clients keep the old API key until cleared.
    create_llm.cache_clear()
    return config


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


def _estimate_tokens(*parts: str) -> int:
    # ponytail: Groq-only fallback when usage_metadata is missing
    total_chars = sum(len(part or "") for part in parts)
    return max(1, (total_chars + 3) // 4)


def _tokens_from_usage(usage_by_model: dict) -> int:
    """Sum Groq/LangChain total_tokens across every model call in a turn."""
    total = 0
    for meta in usage_by_model.values():
        if not isinstance(meta, dict):
            continue
        reported = meta.get("total_tokens")
        if reported is None:
            reported = int(meta.get("input_tokens") or 0) + int(meta.get("output_tokens") or 0)
        total += max(0, int(reported or 0))
    return total


def _billable_llm_tokens(usage_by_model: dict, provider: str, *fallback_parts: str) -> int:
    """Cloud LLM only. Local Ollama is not metered."""
    used = _tokens_from_usage(usage_by_model)
    if used > 0:
        return used
    if provider == "groq":
        return _estimate_tokens(*fallback_parts)
    return 0


@app.post("/chat", response_model=ChatResponse)
def post_chat(
    payload: ChatRequest,
    request: Request,
    user=Depends(require_approved_user),
):
    auth_store = request.app.state.auth_store
    try:
        auth_store.ensure_token_budget(user)
    except PermissionError as error:
        raise HTTPException(status_code=402, detail=str(error)) from error

    profile = request.app.state.settings_store.active_profile().value
    try:
        runtime = request.app.state.profile_manager.get(profile)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Runtime profile is unavailable.")
        raise HTTPException(status_code=503, detail="Selected runtime is unavailable.")

    store = request.app.state.conversation_store
    conversation_id = (
        store.create(user.id) if payload.conversation_id is None else payload.conversation_id
    )
    memory_state = store.load(conversation_id, user_id=user.id)
    if memory_state is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    for question in _sub_questions(payload.question):
        try:
            auth_store.ensure_token_budget(user)
        except PermissionError as error:
            raise HTTPException(status_code=402, detail=str(error)) from error
        try:
            # Groq usage via LangChain callback; Voyage embed/rerank via live API usage.
            with track_cloud_retrieval_usage() as voyage_usage:
                with get_usage_metadata_callback() as usage_cb:
                    result = chat(
                        question,
                        memory_state,
                        retrieval_backend=runtime.retrieval_backend,
                        llm_provider=runtime.answer_provider,
                    )
                    turn_usage = dict(usage_cb.usage_metadata)
                embed_tokens = int(voyage_usage.embed_tokens)
                rerank_tokens = int(voyage_usage.rerank_tokens)
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
            user_id=user.id,
            question=question,
            standalone_query=(memory_state.get("turns") or [{}])[-1].get("standalone_query", question),
            answer=result["answer"],
            sources=result["sources"],
            graph_trace=result["graph_trace"],
            profile=runtime.spec.value.value,
            answer_status=result.get("status"),
        )
        llm_tokens = _billable_llm_tokens(
            turn_usage,
            runtime.answer_provider,
            question,
            str(result.get("answer") or ""),
        )
        user = auth_store.consume_cloud_usage(
            user.id,
            llm=llm_tokens,
            embed=embed_tokens,
            rerank=rerank_tokens,
        )
        if result.get("refusal_reason"):
            store.record_answer_refusal(
                conversation_id=conversation_id,
                user_id=user.id,
                user_email=user.email,
                question=question,
                answer_status=result.get("status"),
                reason=result["refusal_reason"],
                diagnostics=result.get("refusal_diagnostics") or [],
                profile=runtime.spec.value.value,
            )
    if payload.conversation_id is None:
        with get_usage_metadata_callback() as usage_cb:
            store.rename(
                conversation_id,
                _conversation_title(payload.question, runtime.answer_provider),
                user_id=user.id,
            )
            title_tokens = _tokens_from_usage(usage_cb.usage_metadata)
        if title_tokens <= 0 and runtime.answer_provider == "groq":
            title_tokens = _estimate_tokens(payload.question)
        if title_tokens and runtime.answer_provider == "groq":
            user = auth_store.consume_cloud_usage(user.id, llm=title_tokens)
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
def list_conversations(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    user=Depends(require_approved_user),
):
    return request.app.state.conversation_store.list_conversations(user_id=user.id, limit=limit)


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    request: Request,
    user=Depends(require_approved_user),
):
    if not conversation_id or len(conversation_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid conversation identifier.")
    transcript = request.app.state.conversation_store.transcript(
        conversation_id, user_id=user.id
    )
    if transcript is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return transcript


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@app.patch("/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: str,
    payload: RenameRequest,
    request: Request,
    user=Depends(require_approved_user),
):
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be blank.")
    try:
        request.app.state.conversation_store.rename(
            conversation_id, title, user_id=user.id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "title": title}


@app.delete("/conversations/{conversation_id}", status_code=204, response_class=Response)
def delete_conversation(
    conversation_id: str,
    request: Request,
    user=Depends(require_approved_user),
):
    try:
        request.app.state.conversation_store.delete(conversation_id, user_id=user.id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.")


class TurnFeedbackRequest(BaseModel):
    rating: str = Field(pattern="^(up|down)$")


@app.post("/conversations/{conversation_id}/turns/{turn_id}/feedback")
def post_turn_feedback(
    conversation_id: str,
    turn_id: str,
    payload: TurnFeedbackRequest,
    request: Request,
    user=Depends(require_approved_user),
):
    store = request.app.state.conversation_store
    turn = store.get_turn(conversation_id, turn_id, user_id=user.id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found.")
    if payload.rating == "up":
        return {"ok": True, "rating": "up"}
    store.record_answer_refusal(
        conversation_id=conversation_id,
        user_id=user.id,
        user_email=user.email,
        question=turn["question"],
        answer_status=turn.get("answer_status") or "answered",
        reason=f"user_thumbs_down:turn:{turn_id}",
        diagnostics=[f"user_thumbs_down:{turn_id}"],
        profile=turn.get("profile"),
    )
    return {"ok": True, "rating": "down"}


def _resolve_source(filename: str, request: Request):
    try:
        return request.app.state.source_resolver.resolve(filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source filename.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source PDF is not available on this machine.")


@app.get("/sources/{filename}/info")
def get_source_info(filename: str, request: Request, _user=Depends(require_approved_user)):
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
    _user=Depends(require_approved_user),
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
def get_source_pdf(filename: str, request: Request, _user=Depends(require_approved_user)):
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
    from jsonl_supersession import clear_supersession_cache

    clear_supersession_cache()
    root_value = os.environ.get("BCT_RUNTIME_ASSET_ROOT")
    if not root_value:
        raise RuntimeError("BCT_RUNTIME_ASSET_ROOT is not configured")
    configure_runtime_assets(root_value)


def _install_ingestion_routes(target: FastAPI) -> None:
    @target.post("/documents")
    async def ingest_document(
        request: Request,
        file: UploadFile = File(...),
        _admin=Depends(require_admin),
    ):
        # ponytail: admin catalog fields were form theater; filename is enough for listing.
        filename = (file.filename or "").strip()
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are accepted.")
        content_type = (file.content_type or "").lower()
        if content_type and content_type not in {"application/pdf", "application/x-pdf", "binary/octet-stream"}:
            raise HTTPException(status_code=400, detail="Only PDF uploads are accepted.")
        metadata = {"title": Path(filename).stem or filename or "document"}

        max_bytes = int(os.environ.get("BCT_MAX_PDF_BYTES", str(50 * 1024 * 1024)))
        temporary_path = None
        size = 0
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
                temporary_path = Path(handle.name)
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(status_code=413, detail="PDF exceeds the configured size limit.")
                    handle.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

            from ingestion.pipeline import IngestionConfig, IngestionPipeline

            if not os.environ.get("BCT_RUNTIME_ASSET_ROOT"):
                raise HTTPException(status_code=503, detail="Runtime asset root is not configured.")

            config = IngestionConfig.from_environment()
            pipeline = IngestionPipeline(config)
            try:
                report = await run_in_threadpool(
                    pipeline.ingest,
                    temporary_path,
                    original_filename=filename or "document.pdf",
                    metadata=metadata,
                )
            finally:
                pipeline.close()
            _refresh_runtime_asset_environment()
            request.app.state.profile_manager.reset()
            request.app.state.source_resolver.refresh()
            return report
        except HTTPException:
            raise
        except Exception as error:
            # Surface unexpected ingest failures to the admin UI with the PDF name.
            label = Path(filename).name if filename else "document.pdf"
            logger.exception("Document ingestion failed for %s.", label)
            detail = str(error).strip() or "Document ingestion failed."
            if label not in detail:
                detail = f"{label}: {detail}"
            raise HTTPException(status_code=422, detail=detail) from error
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @target.get("/documents")
    def list_documents(
        request: Request,
        limit: int = 100,
        _admin=Depends(require_admin),
    ):
        from ingestion.pipeline import IngestionConfig
        from ingestion.registry import IngestionRegistry

        config = IngestionConfig.from_environment()
        registry = IngestionRegistry(config.registry_path)
        try:
            return registry.list_ready(limit=limit)
        finally:
            registry.close()


_install_ingestion_routes(app)
