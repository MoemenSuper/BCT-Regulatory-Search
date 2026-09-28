from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from langfuse import get_client
from pydantic import BaseModel

from answer_evidence import evidence_warning

from .gemini_visual import VisualPage
from .models import Block, Page, StructuredDocument
from .quality import arabic_character_ratio, assess_page_quality, contains_sensitive_literals


# Cap rendered page images so large statistical PDFs do not OOM the API process.
_MAX_RENDER_EDGE_PX = float(os.environ.get("BCT_PAGE_RENDER_MAX_EDGE", "1600"))
_MAX_RENDER_PIXELS = float(os.environ.get("BCT_PAGE_RENDER_MAX_PIXELS", str(3_500_000)))


def render_page_png(page, *, preferred_scale: float = 2.0, clip=None) -> bytes:
    """Rasterize a PDF page, or one region of it (``clip``), for visual reading with a hard pixel budget."""
    import pymupdf

    rect = pymupdf.Rect(clip) if clip is not None else page.rect
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
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(attempt, attempt), clip=rect if clip is not None else None, alpha=False)
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


def repair_lam_alef(chars: list[dict]) -> str:
    """Join one span's characters, undoing PyMuPDF's RTL lam-alef ligature bug.

    For the ligatures لا/لأ/لإ/لآ PyMuPDF emits the alef first, as a zero-width glyph at
    the start of the lam ("خالل" for "خلال"); the correct order is lam then alef.
    https://github.com/pymupdf/PyMuPDF/issues/2199
    """
    out, i = [], 0
    while i < len(chars):
        char = chars[i]
        if (i + 1 < len(chars) and char["c"] in "اأإآ" and chars[i + 1]["c"] == "ل"
                and abs(char["bbox"][2] - char["bbox"][0]) < 0.05):
            out.append("ل" + char["c"])
            i += 2
            continue
        out.append(char["c"])
        i += 1
    return "".join(out)


def _native_blocks(page, page_number: int) -> list[Block]:
    blocks: list[Block] = []
    for raw in page.get_text("rawdict", sort=True)["blocks"]:
        if raw.get("type", 0) != 0:
            continue
        text = "\n".join(
            "".join(repair_lam_alef(span["chars"]) for span in line["spans"])
            for line in raw.get("lines", [])
        ).strip()
        if not text:
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


def _visual_enabled() -> bool:
    backend = (os.environ.get("BCT_VISUAL_BACKEND") or "").strip().casefold()
    if backend in {"off", "0", "none", "false"}:
        return False
    if backend in {"local", "gemini"}:
        return True
    return os.environ.get("BCT_GEMINI_VISUAL", "1") == "1"


def _whole_page_plan(*, language: str, native_text: str, requires_fallback: bool) -> tuple[bool, bool]:
    """Return (read the whole page visually, the reading is required).

    The text layer is primary. The whole page is read only when it cannot stand on its text
    layer (scanned / empty / garbled digits), or for Arabic when it holds sensitive literals
    (risk mode) or always (all mode). Images inside a readable page are image regions instead.
    """
    if not _visual_enabled():
        return False, False
    arabic_mode = os.environ.get("BCT_GEMINI_ARABIC_MODE", "risk").strip().casefold()
    if requires_fallback:
        # Empty/garbled: try visual when configured. Only Arabic mode=all requires it.
        return True, language == "ar" and arabic_mode == "all"
    if language != "ar" or arabic_mode == "off":
        return False, False
    if arabic_mode == "risk":
        return contains_sensitive_literals(native_text), False
    if arabic_mode == "all":
        return True, True
    raise ValueError("BCT_GEMINI_ARABIC_MODE must be one of: all, risk, off")


class ImageRegions(BaseModel):
    """Visual readings of a page's image regions, in the page's reading order."""

    texts: list[str]
    complete: bool


def read_page_visual(transcriber, page, *, content_hash: str, page_number: int, language: str, regions: list):
    """Read what the PDF text layer lacks: each image region, or else the whole page.

    Shared by extraction and the background enrichment worker, so both read pages the same way.
    """
    if not regions:
        return transcriber.transcribe(
            image_png=render_page_png(page), source_pdf_sha256=content_hash,
            page_number=page_number, language=language, image_region=False,
        )
    texts, complete = [], True
    for box in regions:
        seen = transcriber.transcribe(
            image_png=render_page_png(page, preferred_scale=3.0, clip=box), source_pdf_sha256=content_hash,
            page_number=page_number, language=language, image_region=True,
        )
        text = seen.transcription.strip()
        notes = (seen.chart_notes or "").strip()
        texts.append(f"{text}\n{notes}".strip() if notes and notes not in text else text)
        complete = complete and seen.complete
    return ImageRegions(texts=texts, complete=complete)


_LAYOUT_VERSION = "docling-hybrid-v6"


def load_layout(path: Path, cache: Path | None) -> dict[int, list]:
    """Docling layout per page: block texts, and {"image": box} for boxes to read visually."""
    import json

    if cache is not None and cache.is_file():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("version") == _LAYOUT_VERSION:
            return {int(page): items for page, items in stored["pages"].items()}
    from .docling_layout import page_blocks

    layout = page_blocks(path, cache.with_suffix(".docling.json") if cache is not None else None)
    if cache is not None:
        cache.write_text(json.dumps({"version": _LAYOUT_VERSION, "pages": layout}, ensure_ascii=False), encoding="utf-8")
    return layout


def page_regions(layout: dict[int, list], page_number: int) -> list:
    return [item["image"] for item in layout.get(page_number, []) if isinstance(item, dict)]


class PdfExtractor:
    """Page-preserving extraction: Docling layout + PDF text layer, visual reading where it has no text.

    Visual backend is injected (Gemini for cloud; EasyOCR/PaddleOCR-VL for local).
    """

    def __init__(self, *, visual_transcriber=None, visual_model: str | None = None) -> None:
        self.visual_transcriber = visual_transcriber
        self.visual_model = visual_model or getattr(visual_transcriber, "model", None)

    def extract(
        self, pdf_path: str | Path, *, visual_results: dict | None = None, layout_cache: Path | None = None,
    ) -> StructuredDocument:
        """Extract every page.

        Docling gives each page's blocks in reading order. Boxes the PDF text layer cannot fill
        (pictures, a table or text that is an image) are image regions: read visually and put
        back in place. A page that cannot stand on its text layer is read visually as a whole.

        ``visual_results`` switches to deferred mode: no model is called. A page that needs
        visual reading takes its result from the mapping (``VisualPage`` for a whole page,
        ``ImageRegions`` for its regions, or an error string); a missing entry leaves the page
        ``visual_pending`` for the background enrichment worker. ``layout_cache`` stores the
        Docling layout so the re-extraction after enrichment does not run Docling again.
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

            layout = load_layout(path, layout_cache)
            native_by_page: list[tuple[list[Block], str, list]] = []
            for index in range(pdf.page_count):
                page = pdf.load_page(index)
                items = layout.get(index + 1, [])
                texts = [item for item in items if isinstance(item, str)]
                # Pages where Docling finds no text (scans, covers) keep the PDF text layer, which
                # the quality check then sends to visual reading like any other empty page.
                blocks = [Block(type="paragraph", text=text, page_number=index + 1,
                                metadata={"extraction_method": "native", "layout": "docling"})
                          for text in texts] or _native_blocks(page, index + 1)
                text = "\n".join(block.text for block in blocks).strip()
                if not text:
                    text = page.get_text("text", sort=True).strip()
                    if text:
                        blocks = _text_blocks(text, index + 1, extraction_method="native")
                # Reading order with image regions in place (after the text when Docling had none).
                ordered = items if texts else [block.text for block in blocks] + [i for i in items if isinstance(i, dict)]
                native_by_page.append((blocks, text, ordered))

            language = detect_document_language(path.name, [text for _blocks, text, _ordered in native_by_page])
            pages: list[Page] = []
            visual_count = 0
            hierarchy = Hierarchy([])
            for index, (native_blocks, native_text, ordered) in enumerate(native_by_page):
                page_number = index + 1
                page = pdf.load_page(index)
                regions = [item["image"] for item in ordered if isinstance(item, dict)]
                quality = assess_page_quality(native_text, len(native_blocks))
                page_language = "ar" if arabic_character_ratio(native_text) >= 0.20 else language
                # Native text can look healthy while its digits are garbled by a broken
                # font map (header reads "لسنة 6112" for 2016). Such a page is only
                # usable through visual reading of the page image.
                digits_unreliable = evidence_warning({"source": path.name, "text": native_text}) if native_text else None
                whole, require_complete = _whole_page_plan(
                    language=page_language,
                    native_text=native_text,
                    requires_fallback=quality.requires_fallback or bool(digits_unreliable),
                )
                read_regions = bool(regions) and not whole and _visual_enabled()
                should_visualize = whole or read_regions
                visual = None
                visual_error = None
                visual_pending = False
                if should_visualize and deferred:
                    known = visual_results.get(page_number)
                    if known is None:
                        visual_pending = True
                    elif isinstance(known, str):
                        visual_error = known
                    else:
                        visual = known
                        visual_count += 1
                elif should_visualize:
                    with get_client().start_as_current_observation(
                        name="read-page-visual",
                        as_type="chain",
                        input={"page": page_number, "language": page_language, "native_chars": len(native_text),
                               "native_flags": list(quality.flags), "digits_unreliable": digits_unreliable,
                               "image_regions": len(regions) if read_regions else 0, "require_complete": require_complete},
                    ) as page_trace:
                        if self.visual_transcriber is None:
                            visual_error = "visual_not_configured"
                        else:
                            try:
                                visual = read_page_visual(
                                    self.visual_transcriber, page, content_hash=content_hash, page_number=page_number,
                                    language=page_language, regions=regions if read_regions else [],
                                )
                                visual_count += 1
                            except Exception as error:  # provider/runtime failure is recorded per page
                                visual_error = f"{type(error).__name__}: {error}"
                        page_trace.update(
                            output={"visual_error": visual_error, "visual_complete": bool(visual and visual.complete)},
                            **({"level": "WARNING", "status_message": visual_error} if visual_error else {}),
                        )

                whole_visual = visual if isinstance(visual, VisualPage) else None
                region_visual = visual if isinstance(visual, ImageRegions) else None
                # Required visual (Arabic mode=all) still fails closed unless operators
                # opt into BCT_ALLOW_DEGRADED_INGESTION=1. Blank cover / image pages in
                # born-digital stats PDFs must not abort the whole document.
                allow_degraded = os.environ.get("BCT_ALLOW_DEGRADED_INGESTION", "0") == "1"
                visual_complete = bool(
                    whole_visual is not None
                    and whole_visual.complete
                    and (whole_visual.transcription.strip() or whole_visual.chart_notes.strip())
                )
                required_missing = whole and require_complete and not allow_degraded and not visual_complete
                if required_missing and not deferred:
                    detail = visual_error or "visual_returned_incomplete_transcription"
                    raise ValueError(f"Page {page_number} requires complete visual extraction: {detail}")

                use_visual_as_primary = bool(
                    (quality.requires_fallback or digits_unreliable)
                    and whole_visual is not None
                    and whole_visual.transcription.strip()
                    and whole_visual.complete
                )
                flags = list(quality.flags)
                if digits_unreliable:
                    flags.append(f"native_digits_unreliable:{digits_unreliable}")
                if regions:
                    flags.append("image_regions")
                if quality.requires_fallback and not use_visual_as_primary and not native_text.strip():
                    # Keep an empty page (common: covers, photo spreads) and continue.
                    flags.append("native_unusable_retained")
                if use_visual_as_primary:
                    raw_text = whole_visual.transcription.strip()
                    notes = whole_visual.chart_notes.strip()
                    if notes and notes not in raw_text:
                        raw_text = f"{raw_text}\n\n{notes}".strip()
                    chosen_blocks = _text_blocks(raw_text, page_number, extraction_method="vlm")
                    method = "vlm"
                    flags.append("native_replaced_by_visual")
                    still_unreliable = evidence_warning({"source": path.name, "text": raw_text})
                    if still_unreliable:
                        flags.append(f"visual_digits_unreliable:{still_unreliable}")
                else:
                    region_texts = list(region_visual.texts) if region_visual is not None else []
                    chosen_blocks, position = [], 0
                    for item in ordered:
                        if isinstance(item, str):
                            chosen_blocks.append(Block(type="paragraph", text=item, page_number=page_number,
                                                       metadata={"extraction_method": "native"}))
                            continue
                        seen = region_texts[position].strip() if position < len(region_texts) else ""
                        position += 1
                        if seen:
                            chosen_blocks.append(Block(type="paragraph", text=seen, page_number=page_number,
                                                       metadata={"extraction_method": "image_region"}))
                    if region_visual is not None:
                        flags.append("image_regions_read")
                    raw_text = "\n".join(block.text for block in chosen_blocks).strip()
                    method = "native"
                    if quality.requires_fallback or digits_unreliable:
                        flags.append("fallback_unavailable_native_retained")
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
                    "image_regions": len(regions),
                }
                if should_visualize:
                    metadata["visual_plan"] = {
                        "language": page_language,
                        "image_regions": read_regions,
                        "require_complete": require_complete,
                        # Pages invisible to search without visual reading go first.
                        "priority": 0 if (quality.requires_fallback or digits_unreliable or require_complete)
                        else 1 if read_regions else 2,
                    }
                if whole_visual is not None:
                    metadata.update(
                        {
                            "visual_text": whole_visual.transcription.strip(),
                            "visual_complete": whole_visual.complete,
                            "visual_uncertain_regions": list(whole_visual.uncertain_regions),
                            "visual_sensitive_items": [item.model_dump() for item in whole_visual.items],
                            "visual_model": self.visual_model,
                        }
                    )
                    if whole_visual.uncertain_regions:
                        flags.append("visual_reported_uncertainty")
                if region_visual is not None:
                    metadata.update({"image_region_texts": list(region_visual.texts), "visual_model": self.visual_model})

                pages.append(
                    Page(
                        page_number=page_number,
                        raw_text=raw_text,
                        quality_score=1.0 if use_visual_as_primary else quality.score,
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
                "native_extractor": "docling+pymupdf",
                "visual_provider": self.visual_model if visual_count else None,
                "visual_page_count": visual_count,
                "visual_pending_pages": [page.page_number for page in pages if "visual_pending" in page.quality_flags],
                "page_count": len(pages),
            },
        )
