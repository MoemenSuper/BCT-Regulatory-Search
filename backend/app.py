from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    AttemptLimiter,
    clear_session_cookie,
    open_auth_store,
    require_admin,
    require_approved_user,
    require_user,
    set_session_cookie,
)
from langchain_core.callbacks import get_usage_metadata_callback

from hardware import memory_gb
from llm import ANSWER_PROVIDERS, PROVIDER_ERRORS, answer_provider, create_llm
from runtime_profiles import RuntimeProfile, RuntimeProfileManager, parse_profile
from runtime_retrieval import create_local_backend
# Optional cloud profile: only called when the cloud profile is selected.
from cloud.voyage_client import track_cloud_retrieval_usage
from cloud.voyage_retrieval import create_voyage_backend_from_environment
from source_documents import SourceDocumentResolver, render_page_png, source_info


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile: str = Field(default_factory=lambda: os.environ.get("BCT_DEFAULT_PROFILE", RuntimeProfile.LOCAL_HYBRID.value))

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
    memory_state: dict[str, Any]


def create_runtime_profile_manager(local_backend=None):
    return RuntimeProfileManager(
        local_backend,
        create_voyage_backend_from_environment,
        local_retrieval_factory=create_local_backend,
    )


def supersession_status() -> dict[str, object]:
    """JSONL SUPERSEDES index readiness for health / admin overview."""
    try:
        from jsonl_supersession import load_edges, resolve_edges_path
        from ingestion.index import resolve_active_assets

        root_value = os.environ.get("BCT_RUNTIME_ASSET_ROOT")
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


def _warm_start(profile_manager, settings_store) -> None:
    """Load the active profile's search models and index before serving requests."""
    profile = settings_store.active_profile()
    logger.info(f"Loading search models for profile {profile.value!r}...")
    started = time.monotonic()
    try:
        profile_manager.get(profile)
    except Exception:
        # A broken backend must not block login/admin; the first chat reports the error.
        logger.exception("Warm start failed for profile %s.", profile.value)
        logger.warning("Search models failed to load; they will retry on the first question.")
        return
    logger.info(f"Search models ready in {time.monotonic() - started:.1f}s.")


def _startup_checks(auth_store) -> None:
    """Settings an administrator must fix, as "Startup check" lines in the log (first deployments)."""
    problems = []
    provider = answer_provider()
    key = ANSWER_PROVIDERS.get(provider, ("",))[0]
    if key and not (os.environ.get(key) or "").strip():
        problems.append(f"no key for the answer model ({provider}): set {key} in .env or Admin > Configuration; "
                        "every question will fail")
    pdfs = indexed_pdf_count()
    if not pdfs:
        problems.append("the search index has no PDF: every question will find nothing (bct-assets volume, "
                        "BCT_RUNTIME_ASSET_ROOT)")
    if not any(user.role == "admin" for user in auth_store.list_users()):
        problems.append("no administrator account: set BCT_BOOTSTRAP_ADMIN_EMAIL and BCT_BOOTSTRAP_ADMIN_PASSWORD")
    memory = memory_gb()
    if memory and memory[0] < 8:
        problems.append(f"only {memory[0]:.1f} GB of memory: search needs about 3 GB, indexing an upload 4 GB more")
    root = os.environ.get("BCT_RUNTIME_ASSET_ROOT")
    if root and Path(root).exists() and shutil.disk_usage(root).free < 5 * 2**30:
        problems.append(f"less than 5 GB of free disk for {root}: uploads may fail")
    for problem in problems:
        logger.warning("Startup check: %s.", problem)
    if not problems:
        logger.info("Startup check: OK (answer model %s, %d PDFs indexed).", provider, pdfs)


def _start_enrichment(app: FastAPI):
    """Background visual reading of pages the quick ingest pass left pending."""
    if os.environ.get("BCT_ENABLE_INGESTION") != "1" or not os.environ.get("BCT_RUNTIME_ASSET_ROOT"):
        return None
    if os.environ.get("BCT_ENRICHMENT", "1") == "0":
        return None
    from ingestion.enrichment import EnrichmentWorker
    from ingestion.pipeline import IngestionConfig
    from ingestion.registry import IngestionRegistry

    config = IngestionConfig.from_environment()
    registry = IngestionRegistry(config.registry_path)
    try:
        interrupted = registry.requeue_interrupted()
    finally:
        registry.close()
    if interrupted:
        logger.warning("%d PDF(s) were being indexed when the server stopped; queued again.", interrupted)

    worker = EnrichmentWorker(config, on_activated=lambda: _reload_corpus(app))
    worker.start()
    return worker


def _foreground():
    """Chat requests take priority: background enrichment pauses between pages while one runs."""
    from ingestion.enrichment import FOREGROUND

    with FOREGROUND.busy():
        yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth_store = open_auth_store()
    settings_store = open_app_settings()
    auth_store.bootstrap_admin()
    app.state.auth_store = auth_store
    app.state.settings_store = settings_store
    # Failed logins per email+client IP pair and per client IP; registrations per client IP.
    app.state.login_pair_limiter = AttemptLimiter(limit=5, window_seconds=15 * 60)
    app.state.login_ip_limiter = AttemptLimiter(limit=50, window_seconds=15 * 60)
    app.state.register_limiter = AttemptLimiter(limit=20, window_seconds=60 * 60)
    app.state.profile_manager = create_runtime_profile_manager()
    conversation_store = open_conversation_store()
    app.state.conversation_store = conversation_store
    app.state.source_resolver = SourceDocumentResolver()
    _startup_checks(auth_store)
    status = supersession_status()
    print(
        f"Supersession edges\n  ready: {status['ready']}\n  edge_count: {status['edge_count']}",
        flush=True,
    )
    if os.environ.get("BCT_WARM_START") == "1":
        await run_in_threadpool(_warm_start, app.state.profile_manager, settings_store)
    app.state.enrichment = _start_enrichment(app)
    try:
        yield
    finally:
        if app.state.enrichment is not None:
            await run_in_threadpool(app.state.enrichment.stop)
        closer = getattr(conversation_store, "close", None)
        if callable(closer):
            closer()
        settings_store.close()
        auth_store.close()


app = FastAPI(title="BCT Regulatory Search API", lifespan=lifespan)


@app.middleware("http")
async def log_unexpected_errors(request: Request, call_next):
    """Any error no route handled: its traceback in the log, and a reference the user can quote."""
    try:
        return await call_next(request)
    except Exception:
        reference = uuid.uuid4().hex[:8]
        logger.exception("Unexpected error %s on %s %s", reference, request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"Unexpected server error (reference {reference})."})

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


class SecretsUpdateRequest(BaseModel):
    secrets: dict[str, str | None]


class TokenLimitRequest(BaseModel):
    token_limit: int = Field(ge=0, le=100_000_000)


class DocumentsDeleteRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=500)


def _too_many_attempts() -> HTTPException:
    return HTTPException(status_code=429, detail="Too many attempts. Try again later.")


def _client_ip(request: Request) -> str:
    # Uvicorn rewrites the peer from X-Forwarded-For when the proxy is trusted
    # (FORWARDED_ALLOW_IPS, default 127.0.0.1 — the Vite dev proxy). A proxy it does not
    # trust, or Docker Desktop's port NAT, makes every client share one key.
    return request.client.host if request.client else "unknown"


@app.post("/auth/register")
def register(payload: RegisterRequest, request: Request):
    store = request.app.state.auth_store
    if not request.app.state.register_limiter.hit(_client_ip(request)):
        raise _too_many_attempts()
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
    # Attempts are counted before the password check so parallel requests cannot all slip
    # under the limit; a success uncounts itself. Keyed per email+IP so a stranger's
    # failures cannot lock the owner out from another address.
    pair_limiter = request.app.state.login_pair_limiter
    ip_limiter = request.app.state.login_ip_limiter
    ip = _client_ip(request)
    pair = f"{payload.email.strip().casefold()}|{ip}"
    if not ip_limiter.hit(ip):
        raise _too_many_attempts()
    if not pair_limiter.hit(pair):
        ip_limiter.release(ip)
        raise _too_many_attempts()
    user = store.authenticate(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    pair_limiter.reset(pair)
    ip_limiter.release(ip)
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

@lru_cache(maxsize=1)
def _indexed_sources(native_chunks: str, _mtime: float) -> frozenset[str]:
    """Distinct PDFs in the live index: the ones shipped with the app plus the uploaded ones."""
    sources = set()
    with open(native_chunks, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sources.add(Path(json.loads(line)["metadata"]["source"]).name)
    return frozenset(sources)


def indexed_sources() -> frozenset[str]:
    path = os.environ.get("BCT_NATIVE_CHUNKS_PATH")
    if not path or not os.path.isfile(path):
        return frozenset()
    return _indexed_sources(path, os.path.getmtime(path))


def indexed_pdf_count() -> int:
    return len(indexed_sources())


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
            docs = registry.list_ready(limit=1000)
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
        "documents_ready": indexed_pdf_count(),
        "documents_enriching": sum(1 for doc in docs if doc.get("status") == "enriching"),
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


def _csv_cell(value) -> str:
    """Users type the questions: a leading = + - @ would run as a spreadsheet formula."""
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


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
        writer.writerow(_csv_cell(value) for value in [
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


def _audit(request: Request, admin, action: str, target: str = "", detail: str = "") -> None:
    """Record a successful admin action. Bare test apps (no auth store, no admin) skip it."""
    store = getattr(request.app.state, "auth_store", None)
    if store is not None and admin is not None:
        store.record_audit(admin, action, target, detail)


# ---- Relations between texts (supersession edges): review, approve, reject, add by hand ----


class RelationDecisionRequest(BaseModel):
    # "review" takes a decision back (an added relation is then removed).
    status: str = Field(pattern="^(approved|rejected|review)$")


class RelationAddRequest(BaseModel):
    source_file: str = Field(min_length=5, max_length=300)
    source_page: int = Field(ge=1, le=5000)
    action: str = Field(pattern="^(REPLACE|ABROGATE|MODIFY)$")
    target: str = Field(min_length=4, max_length=120)  # the old text: "Cir 2021-01", "note 2016-08"
    target_article: str | None = Field(default=None, max_length=40)


def _found_edges():
    """Edges the PDFs of the live index give (supersession_edges.jsonl of the active version)."""
    from jsonl_supersession import load_edges, resolve_edges_path

    native = os.environ.get("BCT_NATIVE_CHUNKS_PATH")
    path = resolve_edges_path(Path(native).parent) if native else None
    return load_edges(path) if path else []


def _relation_row(edge_id: str) -> dict:
    from supersession_review import listing

    row = next((row for row in listing(_found_edges()) if row["id"] == edge_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Relation not found.")
    return row


@app.get("/admin/relations")
def admin_relations(_admin=Depends(require_admin)):
    from supersession_edges import instrument_from_filename
    from supersession_review import listing

    # The old text's PDF when the index has it, so the admin can open both sides.
    by_instrument = {instrument_from_filename(name): name for name in sorted(indexed_sources())}
    items = listing(_found_edges())
    for item in items:
        item["target_file"] = by_instrument.get(item["target_instrument"])
    # Every indexed PDF, for the "add a relation" form's file picker.
    return {"items": items, "sources": sorted(indexed_sources())}


@app.post("/admin/relations/{edge_id}/decision")
def admin_relation_decision(edge_id: str, payload: RelationDecisionRequest, request: Request,
                            admin=Depends(require_admin)):
    from supersession_edges import SupersessionEdge
    from supersession_review import decide

    row = _relation_row(edge_id)
    edge = SupersessionEdge(**{key: row[key] for key in SupersessionEdge.__dataclass_fields__})
    decide(edge, None if payload.status == "review" else payload.status, admin.email)
    _audit(request, admin, f"relation.{payload.status}",
           f"{row['source_file']} p.{row['source_page']} -> {row['target_instrument']}")
    return _relation_row(edge_id) if not (payload.status == "review" and row["status"] == "added") else {"removed": True}


@app.post("/admin/relations")
def admin_relation_add(payload: RelationAddRequest, request: Request, admin=Depends(require_admin)):
    """A relation the PDFs did not give. The page must name the old text: its sentence is the proof."""
    from supersession_edges import SupersessionEdge, instrument_from_filename, instruments_from_text
    from supersession_review import decide, declaring_sentence, edge_id, page_text

    source_file = Path(payload.source_file).name
    if source_file not in indexed_sources():
        raise HTTPException(status_code=400, detail=f"{source_file} is not in the search index.")
    source_instrument = instrument_from_filename(source_file)
    if source_instrument is None:
        raise HTTPException(status_code=400, detail=f"The file name {source_file} does not give a circular or note number.")
    targets = sorted(instruments_from_text(payload.target) or instruments_from_text(f"circulaire {payload.target}"))
    if len(targets) != 1:
        raise HTTPException(status_code=400, detail="Write the old text as e.g. Cir 2021-01 or Note 2016-08.")
    text = page_text(Path(os.environ["BCT_NATIVE_CHUNKS_PATH"]), source_file, payload.source_page)
    if not text:
        raise HTTPException(status_code=400, detail=f"Page {payload.source_page} of {source_file} is not in the index.")
    quote = declaring_sentence(text, targets[0])
    if quote is None:
        raise HTTPException(status_code=400, detail=f"Page {payload.source_page} of {source_file} does not mention {payload.target}.")
    edge = SupersessionEdge(source_instrument=source_instrument, source_file=source_file,
                            source_page=payload.source_page, action=payload.action, target_instrument=targets[0],
                            target_article=(payload.target_article or "").strip() or None, quote=quote)
    decide(edge, "added", admin.email)
    _audit(request, admin, "relation.added", f"{source_file} p.{payload.source_page} -> {targets[0]}")
    return _relation_row(edge_id(edge))


@app.get("/admin/audit")
def admin_audit_log(request: Request, limit: int = Query(default=500, ge=1, le=5000), _admin=Depends(require_admin)):
    return {"items": request.app.state.auth_store.list_audit(limit=limit)}


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
    _audit(request, admin, "user.approve", user.email)
    return {"user": user.public_dict()}


@app.post("/admin/users/{user_id}/reject")
def admin_reject_user(user_id: str, request: Request, admin=Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own account status.")
    try:
        user = request.app.state.auth_store.set_status(user_id, "rejected")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    _audit(request, admin, "user.reject", user.email)
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
    _audit(request, admin, "user.promote", user.email)
    return {"user": user.public_dict()}


@app.put("/admin/users/{user_id}/token-limit")
def admin_set_token_limit(
    user_id: str,
    payload: TokenLimitRequest,
    request: Request,
    admin=Depends(require_admin),
):
    try:
        user = request.app.state.auth_store.set_token_limit(user_id, payload.token_limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _audit(request, admin, "user.token_limit", user.email, str(payload.token_limit))
    return {"user": user.public_dict()}


@app.post("/admin/users/{user_id}/reset-tokens")
def admin_reset_token_usage(user_id: str, request: Request, admin=Depends(require_admin)):
    try:
        user = request.app.state.auth_store.reset_token_usage(user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    _audit(request, admin, "user.reset_tokens", user.email)
    return {"user": user.public_dict()}


@app.delete("/admin/users/{user_id}", status_code=204, response_class=Response)
def admin_delete_user(user_id: str, request: Request, admin=Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")
    # Read the email first: the account is gone afterwards.
    target = request.app.state.auth_store.get_by_id(user_id)
    try:
        request.app.state.auth_store.delete_user(user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="User not found.") from error
    except PermissionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _audit(request, admin, "user.delete", target.email if target else user_id)


@app.get("/admin/config")
def admin_get_config(request: Request, _admin=Depends(require_admin)):
    return request.app.state.settings_store.public_configuration()


@app.put("/admin/config/profile")
def admin_set_profile(payload: ProfileUpdateRequest, request: Request, admin=Depends(require_admin)):
    profile = request.app.state.settings_store.set_active_profile(payload.profile)
    request.app.state.profile_manager.reset()
    _audit(request, admin, "config.profile", profile.value)
    return {"active_profile": profile.value}


@app.put("/admin/config/secrets")
def admin_set_secrets(payload: SecretsUpdateRequest, request: Request, admin=Depends(require_admin)):
    try:
        config = request.app.state.settings_store.update_secrets(payload.secrets)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if "BCT_SPEED_MODE" in payload.secrets:
        from reranker import create_reranker

        create_reranker.cache_clear()  # the search below is rebuilt with the new reranker
    request.app.state.profile_manager.reset()
    # Cached Groq/Ollama clients keep the old API key until cleared.
    create_llm.cache_clear()
    # Only the key name is recorded, never the value.
    for key, value in payload.secrets.items():
        cleared = value is None or not value.strip()
        _audit(request, admin, "config.secret_clear" if cleared else "config.secret_set", key)
    return config


_QUESTION_SPLIT = re.compile(r"(?<=[?؟])\s+(?=\S)")


def _sub_questions(text: str) -> list[str]:
    """Run a multi-question message as consecutive turns so every per-question
    gate (retrieval budget, instrument identity, temporal flag) applies once per question.
    ponytail: punctuation-only split; "A et B ?" stays one query. Capped at 3 turns: from the
    third question on, the rest of the message is one turn, so no question is silently dropped."""
    parts = [part.strip() for part in _QUESTION_SPLIT.split(text) if len(part.strip()) >= 12]
    if len(parts) > 3:
        parts = parts[:2] + [" ".join(parts[2:])]
    return parts or [text]


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
    """Cloud answer models only (estimated when the provider reports no usage). Local Ollama is not metered."""
    used = _tokens_from_usage(usage_by_model)
    if used > 0:
        return used
    if provider != "ollama":
        return _estimate_tokens(*fallback_parts)
    return 0


@app.post("/chat", response_model=ChatResponse)
def post_chat(
    payload: ChatRequest,
    request: Request,
    user=Depends(require_approved_user),
    _busy=Depends(_foreground),
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
        started = time.monotonic()
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
        except PROVIDER_ERRORS:
            logger.exception("Answer model unavailable.")
            raise HTTPException(
                status_code=503,
                detail="Le service de réponse est temporairement indisponible. Réessayez dans un instant.",
            )
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("Runtime profile is unavailable.")
            if "CUDA error" in str(error):
                logger.error("The GPU driver was reset; this process cannot use the GPU again. Restart the server.")
            raise HTTPException(status_code=503, detail="Selected runtime is unavailable.")

        logger.info(
            "chat status=%s seconds=%.1f sources=%d refusal=%s diagnostics=%s question=%r",
            result.get("status"), time.monotonic() - started, len(result.get("sources") or []),
            result.get("refusal_reason"), (result.get("refusal_diagnostics") or [])[:5], question[:120],
        )
        memory_state = result["memory_state"]
        store.save_with_turn(
            conversation_id,
            memory_state,
            user_id=user.id,
            question=question,
            standalone_query=(memory_state.get("turns") or [{}])[-1].get("standalone_query", question),
            answer=result["answer"],
            sources=result["sources"],
            profile=runtime.spec.value.value,
            answer_status=result.get("status"),
        )
        llm_tokens = _billable_llm_tokens(
            turn_usage,
            runtime.answer_provider,
            question,
            str(result.get("answer") or ""),
        )
        user = auth_store.consume_usage(
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
        if title_tokens <= 0 and runtime.answer_provider != "ollama":
            title_tokens = _estimate_tokens(payload.question)
        if title_tokens and runtime.answer_provider != "ollama":
            user = auth_store.consume_usage(user.id, llm=title_tokens)
    return {
        "conversation_id": conversation_id,
        "profile": runtime.spec.value.value,
        "status": result.get("status", "answered"),
        "answer": result["answer"],
        "sources": result["sources"],
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
    # "none" takes a rating back; a user can change their mind as often as they like.
    rating: str = Field(pattern="^(up|down|none)$")


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
    # A thumbs-down is one row in the admin's refusal log; any new rating replaces it.
    reason = f"user_thumbs_down:turn:{turn_id}"
    store.delete_answer_refusals(reason=reason, user_id=user.id)
    if payload.rating != "down":
        return {"ok": True, "rating": payload.rating}
    store.record_answer_refusal(
        conversation_id=conversation_id,
        user_id=user.id,
        user_email=user.email,
        question=turn["question"],
        answer_status=turn.get("answer_status") or "answered",
        reason=reason,
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


def _reload_corpus(app: FastAPI) -> None:
    """After an ingest or removal: point search and the PDF viewer at the new active asset version."""
    _refresh_runtime_asset_environment()
    app.state.profile_manager.reset()
    app.state.source_resolver.refresh()


# Admin PDF upload, listing and removal. A router (not @app) so tests can mount these
# routes on a bare FastAPI app without the full startup.
documents_router = APIRouter()


@documents_router.post("/documents")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    doc_kind: str = Form("regulatory"),
    related_to: str = Form(""),
    admin=Depends(require_admin),
):
    # Filename is the listing title; doc_kind chooses primary vs secondary grounding.
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted.")
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in {"application/pdf", "application/x-pdf", "binary/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted.")
    from document_authority import normalize_doc_kind

    metadata = {
        "title": Path(filename).stem or filename or "document",
        "doc_kind": normalize_doc_kind(doc_kind),
    }
    related = (related_to or "").strip()
    if related:
        metadata["related_to"] = related[:300]

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
        worker = getattr(request.app.state, "enrichment", None)
        try:
            # With the background worker the PDF is only stored and queued: it is indexed with
            # the next batch, so a large upload never rebuilds the index once per PDF.
            report = await run_in_threadpool(
                pipeline.queue if worker is not None else pipeline.ingest,
                temporary_path,
                original_filename=filename or "document.pdf",
                metadata=metadata,
            )
        finally:
            pipeline.close()
        if worker is not None:
            worker.wake()
        else:
            await run_in_threadpool(_reload_corpus, request.app)
        logger.info("upload %s: %s%s", filename, report.get("status"), " (duplicate)" if report.get("duplicate") else "")
        if not report.get("duplicate"):
            _audit(request, admin, "document.upload", filename, metadata["doc_kind"])
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

@documents_router.get("/documents")
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
        return registry.list_ready(limit=limit, include_pending=True)
    finally:
        registry.close()

@documents_router.get("/documents/enrichment")
def enrichment_status(request: Request, _admin=Depends(require_admin)):
    worker = getattr(request.app.state, "enrichment", None)
    return worker.snapshot() if worker is not None else {"state": "disabled"}

@documents_router.post("/documents/{document_id}/retry-enrichment")
def retry_enrichment(document_id: str, request: Request, admin=Depends(require_admin)):
    from ingestion.pipeline import IngestionConfig
    from ingestion.registry import IngestionRegistry

    registry = IngestionRegistry(IngestionConfig.from_environment().registry_path)
    try:
        known = registry.get(document_id)
        if known is None or known.get("status") not in {"enriching", "ready_degraded"}:
            raise HTTPException(status_code=404, detail="No document awaiting page reading with that id.")
        requeued = registry.retry_failed_pages(document_id)
        progress = registry.progress(document_id)
    finally:
        registry.close()
    worker = getattr(request.app.state, "enrichment", None)
    if worker is not None:
        worker.wake(reset_cooldown=True)
    _audit(request, admin, "document.retry_reading", str(known.get("original_filename") or document_id))
    return {"document_id": document_id, "requeued_pages": requeued, "enrichment": progress}

@documents_router.delete("/documents/{document_id}", status_code=200)
async def delete_document(
    document_id: str,
    request: Request,
    admin=Depends(require_admin),
):
    from ingestion.pipeline import IngestionConfig, IngestionPipeline

    if not os.environ.get("BCT_RUNTIME_ASSET_ROOT"):
        raise HTTPException(status_code=503, detail="Runtime asset root is not configured.")
    config = IngestionConfig.from_environment()
    pipeline = IngestionPipeline(config)
    try:
        try:
            report = await run_in_threadpool(pipeline.remove, document_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logger.exception("Document removal failed for %s.", document_id)
            detail = str(error).strip() or "Document removal failed."
            raise HTTPException(status_code=422, detail=detail) from error
        await run_in_threadpool(_reload_corpus, request.app)
        _audit(request, admin, "document.delete", report.get("filename") or document_id)
        return report
    finally:
        pipeline.close()

@documents_router.post("/documents/delete", status_code=200)
async def delete_documents(
    body: DocumentsDeleteRequest,
    request: Request,
    admin=Depends(require_admin),
):
    """Remove several PDFs, then reload the search index once."""
    from ingestion.pipeline import IngestionConfig, IngestionPipeline

    if not os.environ.get("BCT_RUNTIME_ASSET_ROOT"):
        raise HTTPException(status_code=503, detail="Runtime asset root is not configured.")
    pipeline = IngestionPipeline(IngestionConfig.from_environment())
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    try:
        for document_id in dict.fromkeys(body.document_ids):
            try:
                report = await run_in_threadpool(pipeline.remove, document_id)
                removed.append(document_id)
                _audit(request, admin, "document.delete", (report or {}).get("filename") or document_id)
            except Exception as error:
                logger.exception("Document removal failed for %s.", document_id)
                failed.append({"document_id": document_id, "error": str(error).strip() or "Document removal failed."})
        if removed:
            await run_in_threadpool(_reload_corpus, request.app)
        return {"removed": removed, "failed": failed}
    finally:
        pipeline.close()


app.include_router(documents_router)
