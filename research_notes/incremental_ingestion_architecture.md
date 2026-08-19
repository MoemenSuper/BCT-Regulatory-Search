# Production architecture for incremental BCT PDF ingestion

Date: 2026-08-18

## Decision

Use an **asynchronous, idempotent, per-document ingestion pipeline**. An administrator upload creates an immutable document version and a durable ingestion job; a GPU worker processes that PDF once with Docling, validates it, chunks and embeds it, and then updates Chroma using deterministic chunk IDs. A batch upload is the same operation repeated as one independent job per PDF.

Do **not** run the current full-directory `ingestion.py` after every upload. Do **not** choose one extraction method for an entire PDF based only on native-text count. Mixed PDFs should first go through Docling's standard PDF pipeline with table extraction and PDF-aware OCR; only pages that fail validation should be retried with full-page OCR.

This yields one one-time baseline migration for the existing public corpus, followed by incremental processing only for new or changed PDF bytes.

## 1. Verified requirements from the BCT cahier des charges

The following are explicit requirements, not architectural assumptions:

- The system must load and process public or authorized PDFs, segment them, generate embeddings, retrieve passages semantically, generate answers grounded only in retrieved passages, and expose sources for human verification (`Cahier_des_charges_BCT_VERSION1.pdf`, page 2).
- A *document administrator* adds documents, enters their principal information, and launches their processing. Required document fields include title, date, type, and category (`Cahier_des_charges_BCT_VERSION1.pdf`, page 3).
- Document ingestion is described as PDF import plus textual-content extraction; preprocessing includes cleaning, chunking, and metadata preparation (`Cahier_des_charges_BCT_VERSION1.pdf`, page 3).
- Sources should expose the document, passage, and page when possible. The assistant aids retrieval and summarization; final regulatory interpretation remains human (`Cahier_des_charges_BCT_VERSION1.pdf`, page 3).
- The backend is the priority, with source traceability and evaluation central to the value of the project (`Cahier_des_charges_BCT_VERSION1.pdf`, page 4).
- A functional backend with upload and question-answer endpoints is a stated deliverable. Evaluation includes accuracy, source correctness, retrieval relevance, performance, and confidentiality (`Cahier_des_charges_BCT_VERSION1.pdf`, page 5).

The cahier requires administrator-driven upload and processing. It does not require rebuilding the entire corpus for every upload, nor does it prescribe synchronous processing.

## 2. Verified facts about the current repository

- `ingestion.py` recursively scans every PDF under `documents/`, accumulates every extracted page in memory, chunks the complete result, creates the embedding model, and passes all chunks to `create_vector_store`.
- `load_pdf.py` sends every PDF through one shared `DoclingPdfLoader`.
- `docling_pdf_loader.py` enables OCR and table structure for every conversion. Its normal OCR mode is `pdf_aware_layout_regions`; `force_ocr=True` switches the entire PDF to `full_page`.
- The loader exports one LangChain `Document` per page and records filename, page, language, extraction method, OCR backend, and OCR mode.
- `vector_store.py` uses `Chroma.from_documents(...)` with the local `./chroma_db` persistent directory and collection `bct_regulations`.
- There is no upload endpoint, document registry, ingestion-job table, stable application-assigned document/chunk ID, duplicate detection, document-version model, progress endpoint, update workflow, or deletion workflow in the current code.
- The API currently has only `/health` and `/chat`.

Therefore the current implementation is a corpus-build script. It is not yet the administrator ingestion workflow required by the cahier.

## 3. Why this architecture fits the actual technologies

### Docling

Docling officially supports:

- `convert` for one document and `convert_all` for multiple documents, with one `ConversionResult` per input;
- conversion status, errors, timings, page ranges, file/page limits, and configurable error behavior;
- independent OCR and table-structure controls;
- OCR modes including `full_page`, `layout_regions`, and `pdf_aware_layout_regions`;
- explicit CUDA acceleration;
- a service deployment with asynchronous jobs and Redis-backed workers when conversion must scale independently.

Sources: [DocumentConverter reference](https://docling-project.github.io/docling/reference/document_converter/), [Docling CLI and OCR-mode reference](https://docling-project.github.io/docling/reference/cli/), [CUDA accelerator example](https://docling-project.github.io/docling/_generated/examples/run_with_accelerator/), [Docling Serve REST API](https://docling-project.github.io/docling/usage/api_server/rest_api/), [Docling Serve deployment](https://docling-project.github.io/docling/usage/api_server/deployment/).

This makes one Docling job per uploaded PDF the natural unit of work. For a mixed PDF, `pdf_aware_layout_regions` is the correct first pass: table analysis remains enabled across the document while OCR is applied to supported bitmap/layout regions. Docling's own full-page OCR example describes `full_page` as slower and appropriate when layout extraction is unreliable or pages are scanned, so it should be a fallback rather than the default for every page. Source: [Docling full-page OCR example](https://docling-project.github.io/docling/_generated/examples/full_page_ocr/).

### FastAPI and a job queue

FastAPI supports `UploadFile`, including multiple uploads, and its spooled-file implementation avoids keeping every large upload entirely in memory. Source: [FastAPI request files](https://fastapi.tiangolo.com/tutorial/request-files/).

FastAPI also explicitly recommends a larger queue system such as Celery for heavy computation that can run in other processes or servers. Docling conversion is exactly that kind of work; it should not keep an upload request open or run inside the API process. Source: [FastAPI background-task caveat](https://fastapi.tiangolo.com/tutorial/background-tasks/).

Celery states that tasks should ideally be idempotent because messages may be redelivered, and it provides retry state, backoff, and jitter. Source: [Celery task guide](https://docs.celeryq.dev/en/stable/userguide/tasks.html).

### Chroma

Chroma's collection API supports `upsert` by record ID and deletion by IDs or metadata filters. Deterministic chunk IDs and `document_id` / `document_version_id` metadata therefore provide the required incremental update seam. Source: [Chroma Python Collection reference](https://docs.trychroma.com/reference/python/collection).

Chroma supports metadata filters in queries, which can be used to limit normal retrieval to active documents/versions. Source: [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).

Chroma warns that deletion is irreversible, so deletion must be controlled by the document workflow rather than exposed as an untracked vector-store operation. Source: [Chroma delete-data guide](https://docs.trychroma.com/docs/collections/delete-data).

The Chroma Python reference describes `PersistentClient` as intended for local development/testing and recommends a server-backed Chroma instance for production. The internal BCT deployment should therefore use a separately running Chroma server and `HttpClient`; the current local persistence remains suitable for the prototype. Sources: [Chroma Python client reference](https://docs.trychroma.com/reference/python/client), [Chroma client-server mode](https://docs.trychroma.com/production/chroma-server/client-server-mode).

## 4. Recommended production flow

```text
Administrator
    |
    | POST one PDF or multiple PDFs + title/date/type/category
    v
FastAPI upload boundary
    |-- validate type, size, authorization, metadata
    |-- stream immutable original to document storage
    |-- compute SHA-256 while streaming
    |-- create document version + ingestion job in SQL
    |-- enqueue job
    '-- return HTTP 202 with document_id/job_id
                         |
                         v
               GPU ingestion worker
                 1. Docling PDF-aware OCR + tables
                 2. validate every page/result
                 3. retry bad pages with full-page OCR
                 4. flag unresolved pages for human review
                 5. deterministic chunking
                 6. embeddings
                 7. staged Chroma upsert
                 8. verification and activation
                         |
             +-----------+-----------+
             v                       v
       SQL audit/status          Chroma server
       source of truth           retrieval projection
```

### API behavior

- `POST /documents`: one PDF and its metadata; returns `202 Accepted`, `document_id`, `version_id`, and `job_id`.
- `POST /documents/batch`: several PDFs; creates one job per PDF and returns their IDs. One bad file must not erase successful siblings.
- `GET /ingestion-jobs/{job_id}`: returns current state, progress, page/chunk counts, warnings, and failure details.
- `GET /documents`: supports the administrator's document list.
- `POST /documents/{document_id}/versions`: uploads a corrected/replacement PDF without destroying audit history.
- `DELETE /documents/{document_id}`: should normally be a controlled soft-delete workflow, not a direct Chroma delete.

FastAPI requires file plus metadata fields to use `multipart/form-data`; JSON cannot simultaneously be the request body of the same multipart request. Metadata can be form fields or a serialized validated form field. Source: [FastAPI request forms and files](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/).

### Job state machine

Use explicit durable states, for example:

```text
uploaded -> queued -> converting -> validating -> chunking
         -> embedding -> indexing -> ready
                              |          |
                              +-> failed <-+
```

`ready` is the only state visible to normal retrieval. Warnings such as a partially successful Docling conversion must be recorded separately; a job must not be labeled ready merely because it produced some text.

### Idempotency and incremental-only processing

Calculate SHA-256 from the uploaded bytes and use it as the immutable content-version fingerprint.

- Same hash already `ready`: do not run Docling or embed again; return the existing successful version/job.
- Same hash currently queued/running: return the existing in-flight job.
- New hash for a new regulation: create a new document and version.
- New hash for an existing logical document: create a new version and preserve the previous version for audit/history.
- A retry uses the same version and deterministic chunk IDs, so repeated delivery cannot create duplicate vectors.

Suggested identities:

```text
document_id          = stable UUID for the logical regulation
document_version_id  = stable UUID for one uploaded byte sequence
content_sha256       = SHA-256 of the original bytes
chunk_id             = SHA-256(document_version_id + page + chunk_ordinal
                                + chunker_version + normalized_chunk_text)
```

The hash registry is **not an extraction cache**. It is the durable identity/audit mechanism that prevents the same upload or a retried queue message from creating duplicate vectors.

### Mixed-page extraction policy

Do not begin with a hand-written router that sends some pages to PyPDF and others to Docling. Because processing occurs once at ingestion time, the correctness-first first pass should be:

1. Docling standard PDF conversion for the complete uploaded document.
2. `do_table_structure=True`.
3. OCR enabled with `pdf_aware_layout_regions`.
4. Validate each output page: conversion status/errors, non-empty result where content is expected, page count, suspiciously low content, and retained table structures.
5. Retry only suspect page ranges with `full_page` OCR.
6. If the fallback is still suspicious, record `needs_review` and prevent automatic activation or allow an administrator to approve with the warning visible.

This handles a page containing native text plus a table, a scanned page next to a native page, and a PDF mixing both patterns without selecting one extraction method for the entire file. It also keeps every fallback decision auditable.

### Version replacement without stale chunks

An upsert alone is insufficient when a replacement has fewer chunks: IDs belonging only to the previous version would remain. Use a staged replacement:

1. Preserve the old version as active while conversion, validation, chunking, and embedding of the new version happen.
2. Upsert all new chunks with deterministic new-version IDs and staged metadata.
3. Verify expected IDs/counts and run ingestion quality checks.
4. Under a short serialized index-promotion operation, make the new version retrievable and make the old version non-retrievable.
5. Delete/archive old vectors only after successful activation and according to retention policy.

Chroma documents the underlying upsert, metadata-filter, update, and delete capabilities, but it does not claim a transaction spanning this complete workflow. Therefore SQL must be the authoritative workflow/version ledger, and the implementation must not describe the multi-step Chroma replacement as atomic. For the internship prototype, serialize promotions through one ingestion worker and block retrieval mutations during the short promotion step. For a multi-instance internal deployment, use a shared lock and test failure recovery explicitly.

### Deletion

1. Mark the document/version `deleting` or `inactive` in SQL so new retrieval does not intentionally select it.
2. Delete its deterministic chunk IDs, or use an exact `document_version_id` metadata filter.
3. Verify no matching Chroma records remain.
4. Mark it deleted and retain the audit record/original according to BCT retention policy.

Do not silently hard-delete the only copy of a regulatory source.

### Required audit data

For each upload/version, record at minimum:

- uploader identity and timestamps;
- original filename, storage URI, byte size, MIME result, and SHA-256;
- administrator metadata: title, date, type, category;
- logical document ID, version ID, previous/superseded version, and document status;
- Docling version, pipeline settings, OCR engine/language/mode, accelerator, result status, warnings/errors, and timings;
- page count, per-page extraction outcome, pages retried with full OCR, and review decisions;
- chunker name/version/settings, embedding model/version, expected chunk IDs/count, and Chroma collection;
- job attempts, worker identity, failure traceback, and activation/deletion timestamps.

This is what makes “processed already” a provable fact rather than an assumption based on a filename or the presence of a vector directory.

## 5. Deployment recommendation

### Internship/prototype

- Keep FastAPI as the API.
- Add a small durable SQL database for document/version/job records.
- Use a real queue worker rather than FastAPI `BackgroundTasks`; Redis plus Celery is a conventional fit.
- Run one CUDA ingestion worker for the current 8 GB RTX 4070 and let uploads queue. Increasing concurrency on one GPU should happen only after VRAM and throughput measurements.
- Keep local original-file storage if deployment is single-machine and backed up.
- Local persistent Chroma is acceptable while this remains a prototype.

### Internal BCT deployment

- Store original PDFs in durable object/document storage with access controls and retention policy.
- Use a managed or backed-up relational database for the document/version/job ledger.
- Run FastAPI, queue broker, and GPU ingestion worker as separate services.
- Run Chroma in server-backed mode with persistent storage and backups.
- Scale upload/API and ingestion workers independently. Use dedicated GPU-worker queues and bounded concurrency.
- Add authentication/authorization, upload-size/type/content checks, malware scanning, transport encryption, secrets management, metrics, structured logs, and recovery drills. These controls are architectural recommendations; their exact BCT implementation must follow internal security and retention policies not specified in the supplied cahier.

Docling itself supports a shared service deployment with Redis-backed workers, so an internal deployment may either call Docling Serve from the ingestion worker or embed the Python library in that worker. The library-in-worker approach is simpler for this prototype; Docling Serve becomes attractive when several applications need the same conversion capacity. Source: [Docling server usage and deployment](https://docling-project.github.io/docling/usage/api_server/).

## 6. Migration from the current corpus

The current Chroma database cannot be treated as a sufficient processing ledger: the application has no content hashes, version records, deterministic chunk IDs, or per-document success manifest.

Recommended migration:

1. Freeze the current public corpus and inventory every source PDF.
2. Create one document/version row and content SHA-256 for each PDF.
3. Build a **new** collection with the new deterministic IDs and full audit metadata; do not mutate the only working collection in place.
4. Process every existing public PDF once through the new validated Docling pipeline. This is the final full-corpus migration.
5. Run corpus completeness checks and the existing evaluation suite.
6. Switch retrieval to the new collection only after it passes acceptance criteria.
7. Keep the old collection temporarily for rollback, then retire it deliberately.
8. From that point onward, process only new hashes or new versions submitted through the upload workflow.

Trying to avoid this one-time migration would preserve uncertainty about which source bytes, extractor settings, chunker, and embedding version produced the current records. That would undermine the auditability requested by the cahier.

## 7. Acceptance criteria before calling it production-ready

- Re-uploading identical bytes creates zero new chunks and returns the known job/version.
- Retrying the same job creates no duplicates.
- A mixed native/scanned/table PDF produces one ordered result per source page, with fallback pages recorded.
- A failed or partial conversion never becomes `ready` silently.
- A replacement with fewer chunks leaves no old-version chunks in normal retrieval.
- Deletion removes the intended document and no other document.
- Single and batch uploads expose independent status and failure details.
- Restarting the API or worker does not lose job state.
- Retrieval never cites a staged, failed, deleted, or superseded version.
- Every returned source resolves to the immutable PDF version and correct page.
- The full evaluation suite is compared against the current baseline before promotion.

## Final conclusion

The ideal solution is not faster full-corpus ingestion and not a complicated pre-Docling page router. It is **process each uploaded PDF exactly once as a durable, GPU-backed, validated job; identify it by content hash; write deterministic versioned chunks; and make SQL—not Chroma—the source of truth for document lifecycle and audit history**.

The single most important architectural change is to replace `ingestion.py` as the operational entry point with the administrator upload + job pipeline. Once the existing corpus has undergone one controlled migration, there is no reason to rerun Docling for all public PDFs when one new PDF arrives.
