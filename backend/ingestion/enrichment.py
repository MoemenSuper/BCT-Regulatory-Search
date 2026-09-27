"""Background enrichment: visually read pending pages, then re-activate in batches.

Upload does a quick native pass (document searchable in minutes). This worker then
reads chart / scanned / garbled pages with the profile's visual backend (EasyOCR +
PaddleOCR-VL locally, Gemini in cloud), one page at a time, yielding to chat.
Each read page is checkpointed in the ingestion registry, so a restart resumes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from langfuse import get_client, propagate_attributes

from .extract import render_page_png
from .local_visual import build_visual_transcriber
from .pipeline import IngestionConfig, IngestionPipeline, ingest_session_id
from .registry import IngestionRegistry

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Foreground:
    """Counts in-flight user requests; background work waits until they are done."""

    def __init__(self) -> None:
        self._active = 0
        self._last = 0.0
        self._condition = threading.Condition()
        self.on_busy = None  # set by the worker: frees the GPU a page read is holding

    @contextmanager
    def busy(self):
        with self._condition:
            self._active += 1
        if self.on_busy is not None:
            self.on_busy()
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._last = time.monotonic()
                self._condition.notify_all()

    def wait_idle(self, stop: threading.Event, grace: float) -> float:
        """Block while requests are in flight or ended less than ``grace`` seconds ago."""
        started = time.monotonic()
        with self._condition:
            while not stop.is_set():
                quiet_for = time.monotonic() - self._last
                if self._active == 0 and quiet_for >= grace:
                    break
                self._condition.wait(timeout=1.0 if self._active else max(0.1, grace - quiet_for))
        return time.monotonic() - started


FOREGROUND = Foreground()


class EnrichmentWorker:
    def __init__(self, config: IngestionConfig, *, on_activated=None, foreground: Foreground = FOREGROUND) -> None:
        self.config = config
        self.on_activated = on_activated
        self.foreground = foreground
        self.batch_pages = int(_env_float("BCT_ENRICH_BATCH_PAGES", 20))
        self.batch_seconds = _env_float("BCT_ENRICH_BATCH_SECONDS", 600)
        self.idle_grace = _env_float("BCT_ENRICH_IDLE_SECONDS", 3)
        self.max_attempts = int(_env_float("BCT_ENRICH_MAX_ATTEMPTS", 3))
        self.breaker_failures = int(_env_float("BCT_ENRICH_BREAKER_FAILURES", 3))
        self.cooldown_seconds = _env_float("BCT_ENRICH_COOLDOWN_SECONDS", 600)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._transcriber = None
        self._cooldown_until = 0.0
        self._state: dict = {"state": "idle"}
        self._document: tuple[str, str] = ("", "")
        self._preempted = False

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self.foreground.on_busy = self._preempt
        self._thread = threading.Thread(target=self._run, name="bct-enrichment", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def wake(self, *, reset_cooldown: bool = False) -> None:
        if reset_cooldown:
            self._cooldown_until = 0.0
        self._wake.set()

    def _preempt(self) -> None:
        """Chat arrived: kill the OCR worker (it holds most of the VRAM even between pages) so search
        models get the GPU; an interrupted page is re-read later."""
        kill = getattr(self._transcriber, "kill", None)
        if callable(kill):
            self._preempted = True
            kill()

    def snapshot(self) -> dict:
        state = dict(self._state)
        if self._cooldown_until > time.monotonic():
            state["state"] = "cooldown"
            state["cooldown_seconds"] = int(self._cooldown_until - time.monotonic())
        return state

    def _run(self) -> None:
        registry = IngestionRegistry(self.config.registry_path)
        try:
            while not self._stop.is_set():
                if self._cooldown_until > time.monotonic():
                    self._release_models()
                    self._wake.wait(self._cooldown_until - time.monotonic())
                    self._wake.clear()
                    continue
                if not self._round(registry):
                    self._state = {"state": "idle"}
                    self._release_models()
                    self._wake.wait(30)
                    self._wake.clear()
        except Exception:
            logger.exception("Enrichment worker crashed.")
        finally:
            self._release_models()
            registry.close()

    def run_until_idle(self, max_passes: int | None = None) -> None:
        """CLI: enrich every queued document now (no chat to yield to)."""
        registry = IngestionRegistry(self.config.registry_path)
        try:
            for _ in range(max_passes or self.max_attempts + 1):
                if registry.next_enriching() is None:
                    return
                self._cooldown_until = 0.0
                self._round(registry)
        finally:
            self._release_models()
            registry.close()

    def _round(self, registry: IngestionRegistry) -> bool:
        """One pass over each enriching document, so a page that keeps failing never blocks the next PDF."""
        seen: set[str] = set()
        while not self._stop.is_set() and self._cooldown_until <= time.monotonic():
            job = registry.next_enriching(exclude=seen)
            if job is None:
                break
            seen.add(str(job["content_sha256"]))
            self.enrich_document(registry, job)
        return bool(seen)

    def _release_models(self) -> None:
        closer = getattr(self._transcriber, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.exception("Failed to release visual models.")
        self._transcriber = None

    # ---- one document ----

    @contextmanager
    def _trace(self, name: str, **observation):
        """One short Langfuse trace per page / activation, grouped per document by session.

        A multi-hour document-level root span would only export when it ends (or never, if
        the process is killed); short traces show up live and survive restarts.
        """
        content_hash, filename = self._document
        profile = (os.environ.get("BCT_DEFAULT_PROFILE") or "local_hybrid").strip().casefold()
        with get_client().start_as_current_observation(name=name, **observation) as span, propagate_attributes(
            session_id=ingest_session_id(content_hash),
            trace_name=name,
            tags=["ingestion", "enrichment", profile],
            metadata={"filename": filename[:200]},
        ):
            yield span

    def enrich_document(self, registry: IngestionRegistry, job: dict) -> None:
        self._document = (str(job["content_sha256"]), str(job.get("original_filename") or ""))
        try:
            self._read_pages(registry, *self._document)
        except Exception as error:
            logger.exception("Enrichment failed for %s.", self._document[1])
            self._open_breaker(f"{type(error).__name__}: {error}")

    def _read_pages(self, registry: IngestionRegistry, content_hash: str, filename: str) -> dict:
        import pymupdf

        known = registry.get(content_hash) or {}
        pdf_path = Path(str(known.get("stored_path") or ""))
        pages = registry.pending_pages(content_hash)
        read = failed = since_activation = consecutive = 0
        waited_total = 0.0
        breaker = None
        last_activation = time.monotonic()
        if pages and self._transcriber is None:
            self._transcriber = build_visual_transcriber(self.config.gemini_cache_dir)
        if pages and self._transcriber is None:
            # Visual backend switched off: these pages stay native-only; say so once.
            for plan in pages:
                registry.page_failed(content_hash, int(plan["page_number"]), error="visual_not_configured", max_attempts=1)
            pages, failed = [], len(pages)
        with pymupdf.open(pdf_path) as pdf:
            for plan in pages:
                if self._stop.is_set():
                    break
                page_number = int(plan["page_number"])
                self._state = {"state": "waiting_for_chat", "document": filename, "page": page_number}
                waited = self.foreground.wait_idle(self._stop, self.idle_grace)
                waited_total += waited
                if self._stop.is_set():
                    break
                self._preempted = False
                self._state = {"state": "reading", "document": filename, "page": page_number}
                outcome = self._read_page(registry, pdf, content_hash, plan, waited)
                if outcome is None:
                    continue  # interrupted by chat: stays pending, not an attempt
                if outcome:
                    read += 1
                    since_activation += 1
                    consecutive = 0
                else:
                    failed += 1
                    consecutive += 1
                    if consecutive >= self.breaker_failures:
                        breaker = f"{consecutive} consecutive page failures; pausing visual reading"
                        self._open_breaker(breaker)
                        break
                if since_activation and (
                    since_activation >= self.batch_pages
                    or time.monotonic() - last_activation >= self.batch_seconds
                ):
                    self._activate(content_hash)
                    since_activation = 0
                    last_activation = time.monotonic()
        # Settle the document status when its queue drained (ready / ready_degraded).
        if since_activation or not registry.pending_pages(content_hash):
            self._activate(content_hash)
        return {"pages_read": read, "pages_failed": failed, "waited_for_chat_s": round(waited_total, 1), "breaker": breaker}

    def _read_page(self, registry: IngestionRegistry, pdf, content_hash: str, plan: dict, waited: float) -> bool | None:
        page_number = int(plan["page_number"])
        with self._trace(
            "enrich-page",
            as_type="chain",
            input={
                "page": page_number,
                "language": plan["language"],
                "chart_suspect": bool(plan["chart_suspect"]),
                "priority": plan["priority"],
                "attempt": int(plan["attempts"]) + 1,
            },
            metadata={"waited_for_chat_s": round(waited, 2)},
        ) as span:
            started = time.monotonic()
            try:
                png = render_page_png(pdf.load_page(page_number - 1))
                visual = self._transcriber.transcribe(
                    image_png=png,
                    source_pdf_sha256=content_hash,
                    page_number=page_number,
                    language=plan["language"],
                    chart_suspect=bool(plan["chart_suspect"]),
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if self._preempted:
                    span.update(status_message=message[:1000], output={"read": False, "preempted_by_chat": True})
                    return None
                registry.page_failed(content_hash, page_number, error=message, max_attempts=self.max_attempts)
                span.update(level="WARNING", status_message=message[:1000], output={"read": False})
                return False
            seconds = time.monotonic() - started
            model = getattr(self._transcriber, "last_model", None) or getattr(self._transcriber, "model", None)
            registry.page_done(
                content_hash, page_number, result_json=visual.model_dump_json(), model=model, seconds=seconds
            )
            span.update(
                output={
                    "read": True,
                    "model": model,
                    "seconds": round(seconds, 1),
                    "transcription_chars": len(visual.transcription),
                    "chart_notes_chars": len(visual.chart_notes or ""),
                    "blank": not (visual.transcription.strip() or (visual.chart_notes or "").strip()),
                }
            )
            return True

    def _activate(self, content_hash: str) -> None:
        self._state = {**self._state, "state": "activating"}
        with self._trace("activate-enrichment", input={"before": self._state}) as span:
            pipeline = IngestionPipeline(self.config)
            try:
                report = pipeline.enrich_activate(content_hash)
            finally:
                pipeline.close()
            span.update(output={key: report.get(key) for key in ("status", "enrichment", "asset_version", "native_chunks_added")})
        if self.on_activated is not None:
            try:
                self.on_activated()
            except Exception:
                logger.exception("Runtime refresh after enrichment failed.")

    def _open_breaker(self, reason: str) -> None:
        """Stop hammering a failing backend (e.g. OCR worker OOM crash); free its memory and retry later."""
        self._cooldown_until = time.monotonic() + self.cooldown_seconds
        self._release_models()
        with self._trace(
            "open-circuit-breaker",
            level="WARNING",
            status_message=reason[:1000],
            metadata={
                "cooldown_s": self.cooldown_seconds,
                "until": datetime.fromtimestamp(time.time() + self.cooldown_seconds, timezone.utc).isoformat(),
            },
        ):
            pass
