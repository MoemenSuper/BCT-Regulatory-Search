"""Docling page layout with the PDF's own words (the "Docling hybrid").

Docling (OCR off, TableFormer accurate) finds reading order, tables and figure boxes. The words
come from the PDF text layer, because Docling reorders Arabic words within a line:
  - every box takes the whole PDF lines lying mostly inside it (lam-alef repaired), never
    clipped characters; a text item with none falls back to Docling's text;
  - tables: "row — column: value; ..." when every numeric row matches the PDF lines inside the
    table box, else the PDF lines under the column headers (TableFormer misread a cell);
  - boxes with no PDF text (a picture, a table or text that is an image): {"image": box}, an
    image region the visual reader fills in place;
  - pictures: always an image region; their PDF words (drawing order) go on one IMAGE_WORDS line,
    searchable but never citable;
  - formulas and empty text items: PDF words inside the box;
  - titles, section headers, list items, captions and tables keep Docling's label:
    {"text": ..., "kind": "heading" | "list_item" | "caption" | "table"} (plain paragraphs stay
    strings). The chunker uses it to cut along the structure and to name each piece's section.
Benchmark (2026-09-28): same retrieval as PyMuPDF, 41 vs 33 of 60 targeted answers, mostly tables.

Docling's native code occasionally crashes its process, so it runs in a worker process; a crash
replaces the worker and retries. Models load once per worker.
"""
from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from rag.answer_evidence import IMAGE_WORDS

_NUM = re.compile(r"\d[\d.,]*\d|\d")
_LONE_DIACRITIC = re.compile(r"(?<!\S)[ً-ٰٟ]+(?!\S)")  # PDF glyph artefacts
_ATTEMPTS = 3
_MIN_REGION_PT = 24  # ~8 mm: smaller boxes are bullets and icons
_PAGE_NUMBER = re.compile(r"(?i)(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?")


def _labelled(text: str, kind: str | None):
    """A text item with Docling's label kept; a plain paragraph stays a string."""
    return {"text": text, "kind": kind} if kind else text


def _clean(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", _LONE_DIACRITIC.sub("", text)).strip()


# --- parent side ---------------------------------------------------------------------------

_IDLE_SECONDS = 120  # the worker holds ~4 GB RAM with its models; free it between uploads
_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_idle_timer: threading.Timer | None = None


class NotEnoughMemory(RuntimeError):
    """Too little free memory to start the Docling worker; the PDF waits instead of crashing the server."""


def _memory():
    from rag.hardware import memory_gb

    return memory_gb(), float(os.environ.get("BCT_DOCLING_MIN_FREE_GB", "4"))


def check_memory() -> None:
    """The worker needs ~4 GB beside the API's search models (measured 4.3 GB on a 7.5 GB Docker
    Desktop, where starting it anyway got the API killed). BCT_DOCLING_MIN_FREE_GB changes it."""
    memory, needed = _memory()
    if memory is not None and memory[1] < needed:
        raise NotEnoughMemory(
            f"reading PDF layouts needs about {needed:g} GB of free memory; {int(memory[1] * 10) / 10:.1f} GB free "
            f"of {memory[0]:.1f} GB. Close other programs, or give Docker more memory (Docker Desktop: "
            "Settings > Resources)."
        )


def _worker_start() -> None:
    """In the worker: if memory still runs out, Linux kills this process, not the API."""
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass  # not Linux


def release() -> None:
    """Stop the worker now and give its ~4 GB back (before an index rebuild)."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
        if _idle_timer is not None:
            _idle_timer.cancel()
    if pool is not None:
        pool.shutdown(wait=True)


def _shutdown_if_idle(pool) -> None:
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False)


def page_blocks(pdf_path: str | Path, raw_cache: Path | None = None) -> dict[int, list]:
    """Per page number, in reading order: block texts and {"image": box} regions. Raises when Docling
    fails; the caller keeps the old corpus. raw_cache keeps Docling's own document: reading it back
    skips the models, so a change to how blocks are assembled never needs a new Docling run."""
    global _pool, _idle_timer
    for attempt in range(1, _ATTEMPTS + 1):
        memory, needed = _memory()
        if _pool is not None and memory is not None and memory[1] < needed:
            release()  # it grows with every PDF it reads: restart it before it runs the server out of memory
        with _pool_lock:
            if _idle_timer is not None:
                _idle_timer.cancel()
            if _pool is None:
                import multiprocessing

                check_memory()
                _pool = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_start
                )
            pool = _pool
        try:
            return pool.submit(_convert, str(pdf_path), str(raw_cache) if raw_cache else None).result()
        except BrokenProcessPool:
            with _pool_lock:
                if _pool is pool:
                    _pool = None
            if attempt == _ATTEMPTS:
                raise RuntimeError(
                    f"the layout reader (Docling) stopped {_ATTEMPTS} times on {Path(pdf_path).name}, "
                    "most likely out of memory"
                )
        finally:
            with _pool_lock:
                if _pool is pool:
                    _idle_timer = threading.Timer(_IDLE_SECONDS, _shutdown_if_idle, args=(pool,))
                    _idle_timer.daemon = True
                    _idle_timer.start()
    raise AssertionError("unreachable")


# --- worker side ---------------------------------------------------------------------------

_converter = None


def _pipeline_options():
    """Best device Docling can see (CUDA > MPS > XPU > CPU), all physical cores, VRAM-sized batch.

    https://docling-project.github.io/docling/usage/gpu/ : layout batches on GPU, TableFormer does
    not; on Apple MPS Docling keeps TableFormer on CPU. Speed only: every device gives the same text.
    """
    import os

    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # scanned pages go to the visual backend
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    from rag.hardware import batch_size, gpu_memory_gb

    # AUTO picks the same device order as hardware.torch_device (CUDA > MPS > XPU > CPU), but a GPU
    # under 12 GB is left to the search models: Docling beside them on an 8 GB laptop GPU reset the
    # Windows driver (event 153), which broke the API's GPU until a restart. Same text, slower.
    device = AcceleratorDevice.AUTO if gpu_memory_gb() == 0 or gpu_memory_gb() >= 12 else AcceleratorDevice.CPU
    opts.accelerator_options = AcceleratorOptions(device=device, num_threads=os.cpu_count() or 4)
    opts.layout_batch_size = batch_size(64)
    return opts


def _convert(pdf_path: str, raw_cache: str | None = None) -> dict[int, list]:
    global _converter
    import pymupdf
    from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TableItem, TextItem

    kinds = {DocItemLabel.TITLE: "heading", DocItemLabel.SECTION_HEADER: "heading",
             DocItemLabel.LIST_ITEM: "list_item", DocItemLabel.CAPTION: "caption"}

    raw = Path(raw_cache) if raw_cache else None
    if raw is not None and raw.is_file():
        doc = DoclingDocument.model_validate_json(raw.read_text(encoding="utf-8"))
    else:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if _converter is None:
            _converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options())}
            )
        doc = _converter.convert(pdf_path).document
        if raw is not None:
            raw.write_text(doc.model_dump_json(), encoding="utf-8")
    pages: dict[int, list] = {}

    def add(page_no: int, value) -> None:
        if value:
            pages.setdefault(page_no, []).append(value)

    with pymupdf.open(pdf_path) as pdf:
        page_lines: dict[int, list] = {}
        claimed: dict[int, set[int]] = {}

        def lines_in(page_no: int, bbox) -> list:
            """[(rect, text)] of the page lines inside a box, marked as claimed."""
            if page_no not in page_lines:
                page_lines[page_no] = _page_lines(pdf[page_no - 1])
                claimed[page_no] = set()
            indices = _lines_in(page_lines[page_no], _rect(pdf[page_no - 1], bbox))
            claimed[page_no].update(indices)
            return [page_lines[page_no][i] for i in indices]

        for item, _level in doc.iterate_items():
            if not getattr(item, "prov", None):
                continue
            page_no = item.prov[0].page_no
            box = _rect(pdf[page_no - 1], item.prov[0].bbox)
            lines = lines_in(page_no, item.prov[0].bbox)
            text = _clean(" ".join(line for _r, line in lines))
            if isinstance(item, PictureItem):
                # Always read visually: whatever PDF words a figure has are in drawing order.
                add(page_no, f"{IMAGE_WORDS} {text}" if text else "")
                add(page_no, _region(box))
            elif isinstance(item, TableItem):
                add(page_no, _labelled(_table_text(item, doc, _visual_rows(lines)), "table") if text else _region(box))
            elif item.label == DocItemLabel.FORMULA or (isinstance(item, TextItem) and not item.text.strip()):
                add(page_no, text or _region(box))
            elif isinstance(item, TextItem):
                # A paragraph continued on the next page has one prov per page; citations are per page.
                per_page: dict[int, list[str]] = {}
                for prov in item.prov:
                    per_page.setdefault(prov.page_no, []).extend(line for _r, line in lines_in(prov.page_no, prov.bbox))
                for number, parts in per_page.items():
                    words = _clean(" ".join(parts)) or _clean(item.text)
                    add(number, _labelled(words, kinds.get(item.label)) if words else _region(box))
        # Lines no body box claims: running headers/footers (Docling's furniture: an edition
        # "Bulletin N°17", a period, a glossary of abbreviations) and lines the layout model missed.
        # Kept at the end of their page, as the plain PDF text always had them; bare page numbers dropped.
        for page_no in range(1, pdf.page_count + 1):
            if page_no in page_lines:
                add(page_no, "\n".join(text for i, (_rect_, text) in enumerate(page_lines[page_no])
                                       if i not in claimed[page_no] and not _PAGE_NUMBER.fullmatch(text)))
    return pages


def _page_lines(page) -> list:
    """The page's PDF text lines in reading order, lam-alef repaired: [(rect, text)]."""
    import pymupdf

    from .extract import repair_lam_alef

    lines = []
    for block in page.get_text("rawdict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(repair_lam_alef(span["chars"]) for span in line["spans"]).strip()
            ink = [char["bbox"] for span in line["spans"] for char in span["chars"] if not char["c"].isspace()]
            if text and ink:
                # Extent of the visible characters: PDFs pad lines with spaces far outside the text.
                rect = pymupdf.Rect(ink[0])
                for bbox in ink[1:]:
                    rect |= bbox
                # Text positions are in the unrotated page; Docling boxes are as displayed
                # (a landscape table stored on a rotated page).
                lines.append((rect * page.rotation_matrix, text))
    return lines


def _visual_rows(lines: list) -> list[str]:
    """Regroup a table's PDF lines into its visual rows. A PDF often stores each cell as its own
    line; a row is the cells sharing a height on the page, read left to right."""
    rows: list[list] = []
    for rect, text in sorted(lines, key=lambda line: (line[0].y0 + line[0].y1) / 2):
        middle = (rect.y0 + rect.y1) / 2
        if rows and abs(middle - rows[-1][0]) <= 0.5 * max(rect.height, 1.0):
            rows[-1][1].append((rect.x0, text))
        else:
            rows.append([middle, [(rect.x0, text)]])
    return [" ".join(text for _x, text in sorted(cells)) for _middle, cells in rows]


def _lines_in(lines: list, box) -> list[int]:
    """Indices of the whole PDF lines lying mostly inside a Docling box. Whole lines, never clipped
    characters: a box drawn slightly tight (rotated headers) must not cut the first letters off a word."""
    inside = []
    for index, (rect, _text) in enumerate(lines):
        area = rect.get_area()
        if (rect & box).get_area() >= 0.5 * area if area else box.contains(rect.tl):
            inside.append(index)
    return inside


def _region(box) -> dict | None:
    """A box the PDF text layer cannot fill (picture, scanned table, text in an image): read it
    visually and put the reading back in its place. Tiny boxes (bullets, icons) are skipped."""
    if box.width < _MIN_REGION_PT or box.height < _MIN_REGION_PT:
        return None
    return {"image": [round(value, 1) for value in (box.x0, box.y0, box.x1, box.y1)]}


def _rect(page, bbox):
    """Docling bbox (bottom-left origin) -> PyMuPDF rect (top-left origin)."""
    import pymupdf

    box = bbox.to_top_left_origin(page.rect.height)
    return pymupdf.Rect(box.l, box.t, box.r, box.b)


def rows_match_pdf(table_rows: list[list[str]], pdf_lines: list[str]) -> bool:
    """Every row with 2+ numbers must appear, in order, within <=3 consecutive PDF lines."""
    line_numbers = [_NUM.findall(line) for line in pdf_lines]
    windows = [sum(line_numbers[i:i + k], []) for k in (1, 2, 3) for i in range(len(line_numbers))]
    for numbers in table_rows:
        n = len(numbers)
        if n >= 2 and not any(w[i:i + n] == numbers for w in windows for i in range(len(w) - n + 1)):
            return False
    return True


def column_headers(header_rows: list[list[str]]) -> list[str]:
    """One name per column from Docling's header rows. A spanning cell repeats in every column it
    covers, so repeats are dropped. Levels are joined with " / " (a parent header over its sub-column); docling-core's own
    export_to_dataframe joins with "." and glues years into one number ("2026.2027")."""
    return [" / ".join(dict.fromkeys(text for text in column if text)) for column in zip(*header_rows)]


def _table_text(item, doc, pdf_lines: list[str]) -> str:
    grid = [[(cell.text or "").strip() for cell in row] for row in item.data.grid]
    header_count = 0
    for index, row in enumerate(item.data.grid):
        if not any(cell.column_header and cell.start_row_offset_idx == index for cell in row):
            break
        header_count += 1
    headers = column_headers(grid[:header_count]) if header_count else [""] * len(grid[0] if grid else [])
    rows = grid[header_count:]
    caption = [c] if (c := _clean(item.caption_text(doc) or "")) else []
    # Always name the columns: a form's columns are empty, and "column: value" only names filled cells.
    columns = ["Colonnes: " + " | ".join(h for h in headers if h)] if any(headers) else []
    if rows_match_pdf([[n for cell in row for n in _NUM.findall(cell)] for row in rows], pdf_lines):
        lines = []
        for row in rows:
            cells = [f"{h}: {v}" if h else v for h, v in zip(headers[1:], row[1:]) if v]
            lines.append(f"{row[0]} — " + "; ".join(cells) if cells else row[0])
        return _clean("\n".join(caption + columns + lines))
    return _clean("\n".join(caption + columns + pdf_lines))
