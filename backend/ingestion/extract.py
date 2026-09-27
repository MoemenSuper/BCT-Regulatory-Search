from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langfuse import get_client

from answer_evidence import evidence_warning

from .models import Block, Page, StructuredDocument
from .quality import arabic_character_ratio, assess_page_quality, contains_sensitive_literals


# Cap rendered page images so large statistical PDFs do not OOM the API process.
_MAX_RENDER_EDGE_PX = float(os.environ.get("BCT_PAGE_RENDER_MAX_EDGE", "1600"))
_MAX_RENDER_PIXELS = float(os.environ.get("BCT_PAGE_RENDER_MAX_PIXELS", str(3_500_000)))


def render_page_png(page, *, preferred_scale: float = 2.0) -> bytes:
    """Rasterize a PDF page for visual OCR / chart persistence with a hard pixel budget."""
    import pymupdf

    rect = page.rect
    width = max(float(rect.width), 1.0)
    height = max(float(rect.height), 1.0)
    scale = min(preferred_scale, _MAX_RENDER_EDGE_PX / max(width, height))
    pixels = width * height * scale * scale
    if pixels > _MAX_RENDER_PIXELS:
        scale *= (_MAX_RENDER_PIXELS / pixels) ** 0.5
    scale = max(0.35, float(scale))
    for attempt in (scale, scale * 0.6, 0.35):
        attempt = max(0.25, float(attempt))
        try:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(attempt, attempt), alpha=False)
            try:
                return pixmap.tobytes("png")
            finally:
                pixmap = None
                try:
                    pymupdf.TOOLS.store_shrink(100)
                except Exception:
                    pass
        except Exception:
            if attempt <= 0.35:
                raise
            continue
    raise RuntimeError("page render failed")


_ARTICLE_FR = re.compile(r"^\s*(Article\s+(?:premier|1er|\d+(?:\s*(?:bis|ter|quater))?)(?:\s*\([^)]*\))?)\s*[:\-–—]?\s*(.*)$", re.I)
_HEADING_FR = re.compile(r"^\s*((?:TITRE|CHAPITRE|SECTION|SOUS[-\s]?SECTION|ANNEXE)\b.*)$", re.I)
_ARTICLE_AR = re.compile(r"^\s*((?:الفصل|فصل|المادة|مادة)\s+(?:[\d٠-٩]+|الأول(?:ى)?|الثاني(?:ة)?|الثالث(?:ة)?))\s*[:\-–—]?\s*(.*)$", re.I)
_HEADING_AR = re.compile(r"^\s*((?:العنوان|الباب|القسم|الجزء|الفرع|الملحق|ملحق)\b.*)$", re.I)
_LIST = re.compile(r"^\s*(?:[-•▪◦]|\d+[.)]|[أ-ي][.)])\s+")


@dataclass(frozen=True)
class ChartSignals:
    image_count: int
    drawing_cluster_count: int
    max_image_area_ratio: float
    max_drawing_area_ratio: float
    suspect: bool

    def as_metadata(self) -> dict[str, object]:
        return {
            "chart_image_count": self.image_count,
            "chart_drawing_clusters": self.drawing_cluster_count,
            "chart_max_image_area_ratio": round(self.max_image_area_ratio, 4),
            "chart_max_drawing_area_ratio": round(self.max_drawing_area_ratio, 4),
            "has_chart": self.suspect,
        }


@dataclass
class Hierarchy:
    headings: list[str]

    def update(self, text: str, language: str) -> tuple[str, str]:
        normalized = " ".join(text.split())
        article = (_ARTICLE_AR if language == "ar" else _ARTICLE_FR).match(normalized)
        if article:
            heading = article.group(1).strip()
            body = article.group(2).strip()
            self.headings = [value for value in self.headings if not _is_article(value, language)]
            self.headings.append(heading)
            return "article", body or heading
        heading = (_HEADING_AR if language == "ar" else _HEADING_FR).match(normalized)
        if heading:
            value = heading.group(1).strip()
            # Keep the hierarchy intentionally shallow. Retrieval experiments did not
            # justify a large legal-structure parser.
            self.headings = [value]
            return "heading", value
        if _LIST.match(normalized):
            return "list_item", normalized
        return "paragraph", text.strip()


def _is_article(text: str, language: str) -> bool:
    return bool((_ARTICLE_AR if language == "ar" else _ARTICLE_FR).match(text))


def classify_blocks(
    blocks: list[Block],
    language: str,
    hierarchy: Hierarchy | None = None,
) -> Hierarchy:
    """Classify one ordered block sequence while optionally carrying state across pages."""
    hierarchy = hierarchy or Hierarchy([])
    for block in blocks:
        block_type, normalized = hierarchy.update(block.text, language)
        block.type = block_type  # type: ignore[assignment]
        block.text = normalized
        block.heading_path = list(hierarchy.headings)
    return hierarchy


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_document_language(filename: str, texts: list[str]) -> str:
    stem = Path(filename).stem.casefold()
    if stem.endswith("_ar"):
        return "ar"
    if stem.endswith("_fr"):
        return "fr"
    joined = "\n".join(texts[:8])
    return "ar" if arabic_character_ratio(joined) >= 0.20 else "fr" if joined.strip() else "unknown"


def _native_blocks(page, page_number: int) -> list[Block]:
    blocks: list[Block] = []
    for raw in page.get_text("blocks", sort=True):
        if len(raw) < 5:
            continue
        text = str(raw[4]).strip()
        if not text:
            continue
        block_type = int(raw[6]) if len(raw) > 6 and isinstance(raw[6], (int, float)) else 0
        if block_type != 0:
            continue
        blocks.append(
            Block(
                type="paragraph",
                text=text,
                page_number=page_number,
                metadata={"extraction_method": "native"},
            )
        )
    return blocks


def _text_blocks(text: str, page_number: int, *, extraction_method: str) -> list[Block]:
    pieces = [piece.strip() for piece in text.replace("\r\n", "\n").split("\n\n") if piece.strip()]
    if len(pieces) <= 1:
        pieces = [piece.strip() for piece in text.splitlines() if piece.strip()]
    return [
        Block(
            type="paragraph",
            text=piece,
            page_number=page_number,
            metadata={"extraction_method": extraction_method},
        )
        for piece in pieces
    ]


def _rect_area(bbox) -> float:
    try:
        return abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def page_chart_signals(page) -> ChartSignals:
    """Geometric chart/figure suspects: large images and/or clustered drawings.

    PyMuPDF documents that cluster_drawings wraps pie/bar charts and similar
    vector figures; get_image_info reports displayed raster images. Native text
    (including legends) is left to _native_blocks.
    """
    page_area = abs(float(page.rect.width) * float(page.rect.height)) or 1.0
    images: list = []
    try:
        images = list(page.get_image_info(xrefs=True) or [])
    except Exception:
        images = []
    max_image = 0.0
    for info in images:
        bbox = info.get("bbox") if isinstance(info, dict) else None
        if bbox is None and isinstance(info, dict):
            bbox = info.get("transform")
        max_image = max(max_image, _rect_area(bbox) / page_area)

    image_blocks = 0
    try:
        for raw in page.get_text("blocks") or []:
            if len(raw) > 6 and int(raw[6]) == 1:
                image_blocks += 1
                max_image = max(max_image, _rect_area(raw[:4]) / page_area)
    except Exception:
        pass

    clusters: list = []
    try:
        clusters = list(page.cluster_drawings() or [])
    except Exception:
        clusters = []
    max_drawing = 0.0
    for rect in clusters:
        try:
            max_drawing = max(max_drawing, abs(float(rect.width) * float(rect.height)) / page_area)
        except Exception:
            continue

    image_count = max(len(images), image_blocks)
    # Substantial figure area → suspect chart/figure even when legend text is fine.
    suspect = (
        max_image >= 0.12
        or max_drawing >= 0.08
        or (image_count >= 1 and max_image >= 0.05)
        or (len(clusters) >= 1 and max_drawing >= 0.04)
    )
    return ChartSignals(
        image_count=image_count,
        drawing_cluster_count=len(clusters),
        max_image_area_ratio=max_image,
        max_drawing_area_ratio=max_drawing,
        suspect=suspect,
    )


def _visual_enabled() -> bool:
    backend = (os.environ.get("BCT_VISUAL_BACKEND") or "").strip().casefold()
    if backend in {"off", "0", "none", "false"}:
        return False
    if backend in {"local", "gemini"}:
        return True
    return os.environ.get("BCT_GEMINI_VISUAL", "1") == "1"


def _visual_plan(
    *,
    language: str,
    native_text: str,
    requires_fallback: bool,
    chart_suspect: bool = False,
) -> tuple[bool, bool]:
    """Return (should_visualize, require_complete_visual).

    Native text is primary. OCR/VLM runs when the page cannot stand on its
    extractable text (scanned / garbled / empty), or when the page is
    chart-suspect (stats layout: prose + chart — merge chart notes into native).
    Chart visual never fails closed the whole document.
    """
    if not _visual_enabled():
        return False, False
    arabic_mode = os.environ.get("BCT_GEMINI_ARABIC_MODE", "risk").strip().casefold()
    if requires_fallback:
        # Empty/garbled: try visual when configured. Only Arabic mode=all requires it.
        if language == "ar" and arabic_mode == "all":
            return True, True
        return True, False
    if chart_suspect and os.environ.get("BCT_GEMINI_CHART_VISION", "1") == "1":
        # Stats pages are usually prose + a chart. Always run visual and merge
        # chart_notes into native text; do not skip when the body is already rich.
        return True, False
    if language != "ar":
        return False, False
    if arabic_mode == "off":
        return False, False
    if arabic_mode == "risk":
        return contains_sensitive_literals(native_text), False
    if arabic_mode == "all":
        return True, True
    raise ValueError("BCT_GEMINI_ARABIC_MODE must be one of: all, risk, off")


def _merge_chart_text(native_text: str, visual) -> str:
    """Keep extractable legend/body; append visual chart notes / missing visual text."""
    pieces = [native_text.strip()] if native_text.strip() else []
    chart_notes = str(getattr(visual, "chart_notes", "") or "").strip()
    transcription = str(getattr(visual, "transcription", "") or "").strip()
    if chart_notes and chart_notes not in native_text:
        pieces.append(chart_notes)
    elif transcription and transcription not in native_text:
        # Avoid duplicating a full page replace when native already holds the body.
        extras = [
            line.strip()
            for line in transcription.splitlines()
            if line.strip() and line.strip() not in native_text
        ]
        if extras:
            pieces.append("\n".join(extras))
    return "\n\n".join(piece for piece in pieces if piece).strip()


class PdfExtractor:
    """Page-preserving extraction: PyMuPDF native text + selective visual repair.

    Visual backend is injected (Gemini for cloud; EasyOCR/PaddleOCR-VL for local).
    """

    def __init__(self, *, visual_transcriber=None, visual_model: str | None = None) -> None:
        self.visual_transcriber = visual_transcriber
        self.visual_model = visual_model or getattr(visual_transcriber, "model", None)

    def extract(self, pdf_path: str | Path, *, visual_results: dict | None = None) -> StructuredDocument:
        """Extract every page.

        ``visual_results`` switches to deferred mode: no model is called. A page that
        needs visual reading takes its result from the mapping (``VisualPage`` or an
        error string); a missing entry leaves the page ``visual_pending`` for the
        background enrichment worker.
        """
        deferred = visual_results is not None
        try:
            import pymupdf
        except ImportError as error:
            raise RuntimeError("PDF ingestion requires PyMuPDF") from error

        path = Path(pdf_path)
        content_hash = sha256_file(path)
        with pymupdf.open(path) as pdf:
            if pdf.page_count < 1:
                raise ValueError("PDF contains no pages")
            max_pages = int(os.environ.get("BCT_MAX_PDF_PAGES", "1000"))
            if pdf.page_count > max_pages:
                raise ValueError(f"PDF exceeds the configured page limit ({max_pages})")

            native_by_page: list[tuple[list[Block], str]] = []
            for index in range(pdf.page_count):
                page = pdf.load_page(index)
                blocks = _native_blocks(page, index + 1)
                text = "\n".join(block.text for block in blocks).strip()
                if not text:
                    text = page.get_text("text", sort=True).strip()
                    if text:
                        blocks = _text_blocks(text, index + 1, extraction_method="native")
                native_by_page.append((blocks, text))

            language = detect_document_language(path.name, [text for _blocks, text in native_by_page])
            pages: list[Page] = []
            visual_count = 0
            hierarchy = Hierarchy([])
            for index, (native_blocks, native_text) in enumerate(native_by_page):
                page_number = index + 1
                page = pdf.load_page(index)
                chart = page_chart_signals(page)
                quality = assess_page_quality(native_text, len(native_blocks))
                page_language = "ar" if arabic_character_ratio(native_text) >= 0.20 else language
                # Native text can look healthy while its digits are garbled by a broken
                # font map (header reads "لسنة 6112" for 2016). Such a page is only
                # usable through visual reading of the page image.
                digits_unreliable = evidence_warning({"source": path.name, "text": native_text}) if native_text else None
                visual = None
                visual_error = None
                should_visualize, require_complete = _visual_plan(
                    language=page_language,
                    native_text=native_text,
                    requires_fallback=quality.requires_fallback or bool(digits_unreliable),
                    chart_suspect=chart.suspect,
                )
                pixmap_png = None
                page_image_tmp: str | None = None
                visual_pending = False
                if deferred:
                    if should_visualize:
                        known = visual_results.get(page_number)
                        if known is None:
                            visual_pending = True
                        elif isinstance(known, str):
                            visual_error = known
                        else:
                            visual = known
                            visual_count += 1
                    if chart.suspect:
                        try:
                            png = render_page_png(page)
                            handle = tempfile.NamedTemporaryFile(prefix="bct-page-", suffix=".png", delete=False)
                            with handle:
                                handle.write(png)
                            page_image_tmp = handle.name
                        except Exception as error:
                            visual_error = visual_error or f"render_failed:{type(error).__name__}: {error}"
                elif should_visualize or chart.suspect:
                    with get_client().start_as_current_observation(
                        name="read-page-visual",
                        as_type="chain",
                        input={
                            "page": page_number,
                            "language": page_language,
                            "native_chars": len(native_text),
                            "native_quality": quality.score,
                            "native_flags": list(quality.flags),
                            "digits_unreliable": digits_unreliable,
                            "chart": chart.as_metadata(),
                            "visualize": should_visualize,
                            "require_complete": require_complete,
                        },
                    ) as page_trace:
                        # Adaptive scale + immediate disk spill: stats PDFs used to keep every
                        # 2× PNG in metadata until pipeline persistence and OOM the API worker.
                        try:
                            pixmap_png = render_page_png(page)
                        except Exception as error:
                            visual_error = f"render_failed:{type(error).__name__}: {error}"
                            pixmap_png = None
                        if pixmap_png is not None and chart.suspect:
                            handle = tempfile.NamedTemporaryFile(prefix="bct-page-", suffix=".png", delete=False)
                            try:
                                handle.write(pixmap_png)
                                handle.close()
                                page_image_tmp = handle.name
                            except Exception:
                                Path(handle.name).unlink(missing_ok=True)
                                raise
                            if not should_visualize:
                                pixmap_png = None
                        if should_visualize:
                            if self.visual_transcriber is None:
                                visual_error = visual_error or "visual_not_configured"
                            elif pixmap_png is None:
                                visual_error = visual_error or "page_render_unavailable"
                            else:
                                try:
                                    visual = self.visual_transcriber.transcribe(
                                        image_png=pixmap_png,
                                        source_pdf_sha256=content_hash,
                                        page_number=page_number,
                                        language=page_language,
                                        chart_suspect=chart.suspect,
                                    )
                                    visual_count += 1
                                except Exception as error:  # provider/runtime failure is recorded per page
                                    visual_error = f"{type(error).__name__}: {error}"
                                finally:
                                    pixmap_png = None
                        page_trace.update(
                            output={
                                "visual_chars": len(visual.transcription) if visual else 0,
                                "chart_notes_chars": len(getattr(visual, "chart_notes", "") or "") if visual else 0,
                                "visual_complete": bool(visual and visual.complete),
                                "visual_error": visual_error,
                                "page_image_kept": bool(page_image_tmp),
                            },
                            **({"level": "WARNING", "status_message": visual_error} if visual_error else {}),
                        )

                # Required visual (Arabic mode=all) still fails closed unless operators
                # opt into BCT_ALLOW_DEGRADED_INGESTION=1. Blank cover / image pages in
                # born-digital stats PDFs must not abort the whole document.
                allow_degraded = os.environ.get("BCT_ALLOW_DEGRADED_INGESTION", "0") == "1"
                visual_complete = bool(
                    visual is not None
                    and visual.complete
                    and (visual.transcription.strip() or str(getattr(visual, "chart_notes", "") or "").strip())
                )
                required_missing = should_visualize and require_complete and not allow_degraded and not visual_complete
                if required_missing and not deferred:
                    detail = visual_error or "visual_returned_incomplete_transcription"
                    raise ValueError(
                        f"Page {page_number} requires complete visual extraction: {detail}"
                    )

                use_visual_as_primary = bool(
                    (quality.requires_fallback or digits_unreliable)
                    and visual is not None
                    and visual.transcription.strip()
                    and visual.complete
                )
                flags = list(quality.flags)
                if digits_unreliable:
                    flags.append(f"native_digits_unreliable:{digits_unreliable}")
                if chart.suspect:
                    flags.append("chart_suspect")
                if quality.requires_fallback and not use_visual_as_primary and not native_text.strip():
                    # Keep an empty page (common: covers, photo spreads) and continue.
                    flags.append("native_unusable_retained")
                if use_visual_as_primary:
                    raw_text = visual.transcription.strip()
                    chart_notes = str(getattr(visual, "chart_notes", "") or "").strip()
                    if chart_notes and chart_notes not in raw_text:
                        raw_text = f"{raw_text}\n\n{chart_notes}".strip()
                    chosen_blocks = _text_blocks(raw_text, page_number, extraction_method="vlm")
                    method = "vlm"
                    flags.append("native_replaced_by_visual")
                    still_unreliable = evidence_warning({"source": path.name, "text": raw_text})
                    if still_unreliable:
                        flags.append(f"visual_digits_unreliable:{still_unreliable}")
                else:
                    raw_text = native_text
                    chosen_blocks = list(native_blocks)
                    method = "native"
                    if quality.requires_fallback or digits_unreliable:
                        flags.append("fallback_unavailable_native_retained")
                    # Chart pages: keep legend text; append visual chart notes when available.
                    if chart.suspect and visual is not None and visual.complete:
                        merged = _merge_chart_text(raw_text, visual)
                        if merged != raw_text:
                            raw_text = merged
                            chosen_blocks = _text_blocks(raw_text, page_number, extraction_method="native")
                            flags.append("chart_notes_merged")
                            method = "native"
                    if required_missing:
                        # Native text of a page that must be read visually is not quotable.
                        raw_text, chosen_blocks = "", []
                        flags.append("visual_pending_required" if visual_pending else "visual_required_failed")
                if visual_pending:
                    flags.append("visual_pending")

                hierarchy = classify_blocks(chosen_blocks, page_language, hierarchy)
                metadata = {
                    "native_raw_text": native_text,
                    "native_quality_score": quality.score,
                    "native_quality_flags": list(quality.flags),
                    "latin_character_ratio": quality.latin_character_ratio,
                    "single_arabic_token_ratio": quality.single_arabic_token_ratio,
                    "visual_attempted": should_visualize,
                    "visual_error": visual_error,
                    **chart.as_metadata(),
                }
                if should_visualize:
                    metadata["visual_plan"] = {
                        "language": page_language,
                        "chart_suspect": chart.suspect,
                        "require_complete": require_complete,
                        # Pages invisible to search without visual reading go first.
                        "priority": 0 if (quality.requires_fallback or digits_unreliable or require_complete)
                        else 1 if chart.suspect else 2,
                    }
                if page_image_tmp:
                    # Transient path; pipeline moves under immutable page-images/.
                    metadata["page_image_tmp"] = page_image_tmp
                if visual is not None:
                    metadata.update(
                        {
                            "visual_text": visual.transcription.strip(),
                            "visual_complete": visual.complete,
                            "visual_uncertain_regions": list(visual.uncertain_regions),
                            "visual_sensitive_items": [item.model_dump() for item in visual.items],
                            "visual_model": self.visual_model,
                            "contains_chart": bool(getattr(visual, "contains_chart", False) or chart.suspect),
                            "chart_notes": str(getattr(visual, "chart_notes", "") or "").strip(),
                        }
                    )
                    if visual.uncertain_regions:
                        flags.append("gemini_reported_uncertainty")
                    if getattr(visual, "contains_chart", False):
                        flags.append("gemini_contains_chart")
                        metadata["has_chart"] = True

                pages.append(
                    Page(
                        page_number=page_number,
                        raw_text=raw_text,
                        quality_score=1.0 if use_visual_as_primary and visual and visual.complete else quality.score,
                        extraction_method=method,
                        quality_flags=flags,
                        metadata=metadata,
                        blocks=chosen_blocks,
                    )
                )

        return StructuredDocument(
            filename=path.name,
            language=language,
            content_sha256=content_hash,
            pages=pages,
            metadata={
                "native_extractor": "pymupdf",
                "visual_provider": self.visual_model if visual_count else None,
                "visual_page_count": visual_count,
                "visual_pending_pages": [page.page_number for page in pages if "visual_pending" in page.quality_flags],
                "page_count": len(pages),
            },
        )
