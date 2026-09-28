# Implementation plan: Docling hybrid ingestion, hardware, answer gates (2026-09-28)

Decisions agreed in session on 2026-09-28. Measurements come from the benchmark in
`backend/tests/benchmark/suite_stats/` (question sets) and the 807-question eval.

## Measured basis

| Variant (848 questions, Page@5) | All | FR | AR | Stats tables Doc@5 |
|---|---|---|---|---|
| PyMuPDF (today) | 86.8 | 90.5 | 83.4 | 72.0 |
| PyMuPDF + lam-alef fix | 87.6 | 90.7 | 84.9 | 72.0 |
| Docling (pure) | 86.7 | 90.7 | 82.8 | 84.0 |
| Docling hybrid | 86.7 | 89.8 | 84.0 | 84.0 |

Retrieval differences are not significant (p ≥ 0.30). Answers on 60 table/chart/statistics
questions: **Docling hybrid 41/60 vs PyMuPDF+fix 33/60** (9 wins, 1 loss, p ≈ 0.02); every win
but one is a table.

Pure Docling reorders Arabic words inside a line, which breaks verbatim quote gates. Hence the
hybrid: Docling decides layout; text comes from the PDF text layer.

## 1. Extraction: Docling hybrid (`backend/ingestion/extract.py`)

Replace `_native_blocks` with a Docling-backed page reader. Everything downstream
(`StructuredDocument`, `classify_blocks`, chunking, staging, supersession) is unchanged.

- Docling (OCR off by default, TableFormer accurate) gives reading order, headings, list items,
  tables, pictures, formulas.
- Paragraph text = PDF text layer inside the Docling box (PyMuPDF `rawdict`), with the lam-alef
  swap repair. A paragraph continued on the next page is split per page using Docling's
  per-page `charspan`s.
- Table = one block of rows `label — header: value; …` when every numeric row matches one to
  three consecutive PDF text lines in the table box (row check). Otherwise the table is emitted
  from its PDF text lines under Docling's headers, and each line whose value count equals the
  column count is labelled column by column (removes the wrong-column risk).
- Arabic table cells: text from the PDF text layer inside each cell box (Docling reorders Arabic).
- Pictures and formulas: PDF words inside the box.
- `assess_page_quality` still runs on the result. Unreadable or garbled pages (broken font maps,
  e.g. `dette2024.pdf`) and image-only regions go to OCR (see 3).
- Page headers and footers stay out of the body (Docling furniture), as in the benchmark.

## 2. Docling in its own worker process

Docling crashed with a native segfault twice in about 900 PDF conversions (random, not
file-specific). Run it like the PaddleOCR-VL worker: a child process that exits with its parent.
A crash retries the PDF once; a second crash marks that PDF's quick pass as failed. The API
never loads Docling.

## 3. OCR through Docling where possible

Image regions and scanned pages: Docling's built-in EasyOCR in its default PDF-aware mode, which
OCRs only regions without a text layer, with the document language from `_ar`/`_fr`. This
replaces the separate EasyOCR wrapper for those pages. Pages flagged by the quality check get
full-page OCR. PaddleOCR-VL stays only for raster charts. Gemini VLM stays for the cloud
profile. This changes results, so it gets its own benchmark variant before switching.

## 4. Hardware helper (one module, used everywhere)

`backend/hardware.py`: best device (CUDA > MPS > XPU > CPU, with `BCT_DEVICE` as an override),
physical core count, VRAM. It replaces the duplicated `_torch_device` functions in
`embedding.py`/`reranker.py` and sets Docling's `AcceleratorOptions` and batch sizes. Layout
batch: 64 on CUDA with ≥12 GB free, 16–32 on 8 GB (search models share the card), 16 on
MPS/XPU, 4 on CPU. Speed only, never results. The admin overview shows the detected hardware.
Install notes pick the torch wheel: NVIDIA cu126, Intel XPU, macOS default, CPU.

## 5. Reranker on CPU-only servers (measured before enabling)

int8 reranker plus reciprocal rank fusion with the top-15 rerank, enabled only on CPU after a
benchmark shows no Page@5 loss. The BM25 tokenizer fix (NFKC, elision, accent/Arabic folding,
digit splitting, `bm25s` + `PyStemmer`) comes first, because top-15 depends on it.

## 6. Storage hygiene

- Done: `prune_versions` (keep active + 2 previous) after every committed activation.
- Startup sweep of Chroma segment folders that the database no longer references (Windows
  leaves them behind).
- Page numbering made explicit: `normalize_page` must not infer 0- vs 1-based from whether a
  `pages` key exists.

## 7. Answer gates

First diagnosis (16 human-style questions, every LLM call recorded):

- **10 of 16 questions hit HTTP 413** on the free Groq tier: "Request too large … tokens per
  minute (TPM): Limit 8000, Requested 8347". Selection and drafting prompts are 22–26k characters
  (7–8k tokens). Fixed instructions alone are 8.3k (selection) and 10.7k (drafting) characters.
- Of those 10, only 1 ended `answered` (6 partial, 3 refused). Of the 6 without a 413, 4 ended
  `answered`.
- **The app hides provider failures.** A failed selection call becomes `selection_error`, and the
  answer continues from all retrieved evidence. A failed draft falls through the forced-partial
  ladder. Users see a refusal or a weaker answer, not an error.

Measured so far, much of the reported over-refusal is prompt size versus provider limits plus
silent degradation, not the claim gates. The gate diagnosis resumes once calls fit within the
provider limit (paid key, or a prompt budget).

Fixes (independent of the remaining diagnosis):
1. Provider errors (413/429/5xx) end the turn with an explicit "service unavailable" status that
   is logged as such, never as a refusal of the evidence.
2. Prompt budget: evidence trimmed per record to the question-relevant span (the existing
   `_compact_evidence_for_formulation` logic), and the instruction blocks deduplicated, so each
   call stays well under the provider limit. This changes results and gets benchmarked.

## Order

1. Extraction hybrid + row check + Arabic cells (1), test on fixtures, re-run the benchmark.
2. Docling worker process (2) and hardware helper (4).
3. Answer-gate fixes (7), measured on the human-style set and the 807 eval.
4. OCR through Docling (3), benchmarked separately.
5. BM25 tokenizer, then the CPU reranker (5).
6. Storage hygiene leftovers (6).
