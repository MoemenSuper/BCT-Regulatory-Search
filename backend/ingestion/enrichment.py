"""Background ingestion: index uploaded PDFs in batches, then visually read pending pages.

An upload only stores the PDF and queues it. This worker first indexes the queue, up to
BCT_INGEST_BATCH PDFs per new asset version (one index rebuild per batch, not per PDF), so the
PDFs become searchable from their text. Then it reads chart / scanned / garbled pages with the
profile's visual backend (EasyOCR + PaddleOCR-VL locally, Gemini in cloud), one page at a time.
It always yields to chat, and checkpoints in the ingestion registry, so a restart resumes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .extract import load_layout, page_regions, read_page_visual
from .local_visual import build_visual_transcriber
from . import docling_layout
from .docling_layout import NotEnoughMemory
from .pipeline import IngestionConfig, IngestionPipeline
from .registry import SEARCHABLE, IngestionRegistry

logger = logging.getLogger(__name__)


def _memory_shortage() -> str | None:
    """Why the layout reader cannot start now (free memory, needed memory), or None when it can."""
    try:
        docling_layout.check_memory()
    except NotEnoughMemory as error:
        return str(error)
    return None


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
        self.batch_documents = int(_env_float("BCT_INGEST_BATCH", 25))
        self.memory_wait_seconds = _env_float("BCT_INGEST_MEMORY_WAIT_SECONDS", 120)
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
        if state.get("state") == "waiting_for_memory":
            # Live numbers: the admin page shows the memory freed by closing programs at once.
            detail = _memory_shortage()
            if detail is None:
                return {"state": "indexing", "queued": state.get("queued", 0)}  # starts within seconds
            return {**state, "detail": detail}
        if self._cooldown_until > time.monotonic():
            state["state"] = "cooldown"
            state["cooldown_seconds"] = int(self._cooldown_until - time.monotonic())
        return state

    def _run(self) -> None:
        registry = IngestionRegistry(self.config.registry_path)
        try:
            while not self._stop.is_set():
                if self._state.get("state") == "waiting_for_memory" and self._cooldown_until > time.monotonic():
                    # Free memory is checked every few seconds: indexing starts as soon as there is enough.
                    if self._wake.wait(5):
                        self._wake.clear()
                    if _memory_shortage() is None:
                        self._cooldown_until = 0.0
                    continue
                if self._cooldown_until > time.monotonic():
                    self._release_models()
                    self._wake.wait(self._cooldown_until - time.monotonic())
                    self._wake.clear()
                    continue
                if self._index_queued(registry):
                    continue
                if not self._round(registry):
                    self._state = {"state": "idle"}
                    self._release_models()
                    self._wake.wait(30)
                    self._wake.clear()
        except BaseException:
            # Ctrl+C or a server stop interrupts the layout reader mid-PDF: not a crash, the PDF is
            # queued again at the next start (requeue_interrupted).
            if not self._stop.is_set():
                logger.exception("Enrichment worker crashed.")
            else:
                logger.info("Server stopping: indexing interrupted, resumed at the next start.")
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

    def _index_queued(self, registry: IngestionRegistry) -> bool:
        """Index one batch of queued uploads; False when the queue is empty."""
        if not registry.next_queued(1):
            return False
        self._release_models()  # the page readers' memory goes to the layout reader
        waiting = registry.connection.execute("SELECT count(*) FROM ingestion_documents WHERE status='queued'").fetchone()[0]
        self._state = {"state": "indexing", "queued": waiting}
        pipeline = IngestionPipeline(self.config)
        try:
            reports = pipeline.ingest_queued(self.batch_documents, before_each=self._before_indexing)
        except Exception:
            logger.exception("Indexing a batch of uploads failed.")
            self._cooldown_until = time.monotonic() + 60
            return True
        finally:
            pipeline.close()
        if any(report.get("status") in SEARCHABLE for report in reports.values()) and self.on_activated is not None:
            try:
                self.on_activated()
            except Exception:
                logger.exception("Runtime refresh after indexing failed.")
        if not reports:
            # Nothing could start (not enough memory): wait for free memory, never crash the server.
            self._state = {"state": "waiting_for_memory", "queued": waiting, "detail": _memory_shortage() or ""}
            self._cooldown_until = time.monotonic() + self.memory_wait_seconds
        return True

    def _before_indexing(self, job: dict) -> None:
        """Before each PDF of a batch: chat first."""
        self._state = {**self._state, "document": str(job.get("original_filename") or "")}
        self.foreground.wait_idle(self._stop, self.idle_grace)
        if self._stop.is_set():
            raise NotEnoughMemory("server stopping")

    def _release_models(self) -> None:
        closer = getattr(self._transcriber, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.exception("Failed to release visual models.")
        self._transcriber = None

    # ---- one document ----

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
        # Image-region pages read the boxes Docling found; its layout is cached beside the PDF.
        layout = load_layout(pdf_path, pdf_path.parent / "docling-layout.json") if any(
            plan["image_regions"] for plan in pages) else {}
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
                try:
                    outcome = self._read_page(registry, pdf, content_hash, plan, layout)
                except NotEnoughMemory as error:
                    # Not a page error: this machine lacks the memory to read pictures now. The waiting
                    # pages are marked unread at once (the PDF stays searchable from its text), instead
                    # of failing 3 times into a pause; "Relire" on the admin page tries again.
                    rest = pages[pages.index(plan):]
                    for page in rest:
                        registry.page_failed(content_hash, int(page["page_number"]),
                                             error=f"not enough memory: {error}", max_attempts=1)
                    failed += len(rest)
                    logger.warning("Pictures and scans of %s not read (%d page(s)): %s", filename, len(rest), error)
                    break
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

    def _read_page(self, registry: IngestionRegistry, pdf, content_hash: str, plan: dict,
                   layout: dict) -> bool | None:
        page_number = int(plan["page_number"])
        started = time.monotonic()
        try:
            visual = read_page_visual(
                self._transcriber,
                pdf.load_page(page_number - 1),
                content_hash=content_hash,
                page_number=page_number,
                language=plan["language"],
                regions=page_regions(layout, page_number) if plan["image_regions"] else [],
            )
        except NotEnoughMemory:
            raise
        except Exception as error:
            if self._preempted:
                return None  # a chat question stopped the reader; the page is read again later
            message = f"{type(error).__name__}: {error}"
            registry.page_failed(content_hash, page_number, error=message, max_attempts=self.max_attempts)
            logger.warning("Reading page %d of %s failed (attempt %d): %s",
                           page_number, self._document[1], int(plan["attempts"]) + 1, message)
            return False
        seconds = time.monotonic() - started
        model = getattr(self._transcriber, "last_model", None) or getattr(self._transcriber, "model", None)
        registry.page_done(
            content_hash, page_number, result_json=visual.model_dump_json(), model=model, seconds=seconds
        )
        return True

    def _activate(self, content_hash: str) -> None:
        self._state = {**self._state, "state": "activating"}
        pipeline = IngestionPipeline(self.config)
        try:
            report = pipeline.enrich_activate(content_hash)
        finally:
            pipeline.close()
        logger.info("Pages read so far of %s are now searchable: status=%s", self._document[1], report.get("status"))
        if self.on_activated is not None:
            try:
                self.on_activated()
            except Exception:
                logger.exception("Runtime refresh after enrichment failed.")

    def _open_breaker(self, reason: str) -> None:
        """Stop hammering a failing backend (e.g. OCR worker OOM crash); free its memory and retry later."""
        self._cooldown_until = time.monotonic() + self.cooldown_seconds
        self._release_models()
        logger.warning("Page reading paused for %d s after repeated failures: %s", self.cooldown_seconds, reason[:1000])
