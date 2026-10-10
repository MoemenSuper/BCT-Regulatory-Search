# Agents

Working tree: this repository root (`BCT-Regulatory-Search/`).
Preserve grounded-answer behavior. Prefer delete/reuse over new layers (ponytail **full**).

## Read first

| When | Open |
| --- | --- |
| Domain wording (instrument vs PDF, evidence vs citation, verified vs in force) | [CONTEXT.md](CONTEXT.md) |
| How to run backend, UI, ingest | [README.md](README.md) |
| Frontend API wiring and evidence panel | [frontend/README.md](frontend/README.md) |

## Layout

`backend/` is split by importance: `rag/` (the core: question → grounded answer), `ingestion/`
(the core: PDF → chunks), `extras/` (relations, accounts, settings, profiles, logs), `cloud/`
(optional cloud profile). Start points (`app.py`, `run_api.py`, `docker_serve.py`, `ingest.py`,
`bake_assets.py`) stay at the top. Imports name the folder: `from rag.answer_draft import …`.

## Contracts (edit here, not around them)

| Concern | Start in |
| --- | --- |
| Answer shape, claim/quote gates, answer statuses | `backend/rag/answer_contract.py` (facade); gates in `answer_gates.py`, draft ladder in `answer_draft.py` |
| Currentness classifiers (`is_temporal_rule_query`, `is_relationship_query`) | `backend/rag/query_currentness.py` |
| JSONL SUPERSEDES pin / ingest merge | `backend/extras/jsonl_supersession.py` (facade); edges IO in `supersession_edges.py`, retrieve pin in `supersession_pin.py` |
| Admin review of relations (approve / reject / add by hand; decisions kept outside the index) | `backend/extras/supersession_review.py`, `/admin/relations` in `app.py`, `frontend/src/components/admin/RelationsPage.tsx` |
| Conversation routing and follow-ups | `backend/rag/conversation.py` |
| Profiles `cloud` / `local_hybrid` / `local` | `backend/extras/runtime_profiles.py` |
| Answer model provider (OpenAI / Claude / Gemini / Groq / Ollama; one key is enough) | `backend/rag/llm.py` (`answer_provider`, `create_llm`) |
| Nginx in front of the container | the server's own Nginx: `deploy/nginx/bct-regulatory-search.conf`; bundled Nginx (`docker compose --profile nginx`): `deploy/nginx/docker/` (same settings: change both) |
| Chunks cut along the page structure, each with a context header (document, title, section, element, page) that every search reads; answers quote the chunk text only | `backend/ingestion/chunk.py` (`build_runtime_chunks`), `backend/rag/bm25.py` (`searchable_text`), Docling labels in `backend/ingestion/docling_layout.py` |
| Retrieval + evidence selection | `backend/rag/runtime_retrieval.py` (local backend + `create_local_backend`), `backend/rag/retrieval_selection.py` |
| Search speed on CPU (one reranker wording per script, `BCT_SPEED_MODE` 8-bit reranker) | `backend/rag/runtime_retrieval.py` (`script_scores`), `backend/rag/reranker.py` |
| PDF resolve, physical page, quote locate | `backend/extras/source_documents.py` |
| Ingest → stage → activate | `backend/ingest.py`, `backend/ingestion/` |
| Upload queue, batched indexing (one asset version per batch), background enrichment (page ledger, chat priority, statuses) | `backend/ingestion/pipeline.py`, `enrichment.py`, `registry.py` |
| Docker volume seeding from the baked index (`SEEDED_FROM.txt`) | `backend/docker_serve.py` |
| Server log file | `backend/extras/server_logging.py` |
| Visual ingest (EasyOCR / PaddleOCR-VL; `VisualPage` in `models.py`) | `backend/ingestion/local_visual.py`, `extract.py` |
| Optional cloud profile (Voyage search + index staging, Gemini page reading) | `backend/cloud/` — only called when the cloud profile, `BCT_INGEST_CLOUD_INDEX=1`, or `BCT_VISUAL_BACKEND=gemini` is on. The shipped index has no Voyage vectors: an admin builds them from the configuration screen (`POST /admin/config/cloud-index`, `cloud/voyage_index.py` `build_cloud_index`); the cloud profile cannot be chosen before, and every later upload keeps them current |
| HTTP surface | `backend/app.py`, `backend/run_api.py` |
| Auth / sessions / roles / admin audit log | `backend/extras/identity.py` (audit rows written by `_audit` in `app.py`) |
| App profile + provider secrets | `backend/extras/app_settings.py` |
| UI turn presentation | `frontend/src/` (`ResearchNote`, `EvidencePanel`, `api/chat.ts`) |
| Login / admin UI | `frontend/src/Root.tsx`, `LoginPage`, `AdminDashboard` |

## Hard rules

1. **Grounded citations** — filenames and pages come from retrieval metadata. Do not invent paths or pages in prompts, fixtures, or UI copy.
2. **Quotations** — claims need verbatim page text. Fail closed to `insufficient_evidence` / `search_results` rather than paraphrase support.
3. **Trusted paths** — resolve PDFs only via trusted document roots or the ingestion ledger.
4. **Staged activation** — new asset versions stage before they go live; a failed activation keeps the old corpus.
5. **JSONL supersession** — optional `supersession_edges.jsonl` pins successor declaring pages. A SUPERSEDES / REPLACE / ABROGATE edge proves a relationship, not “currently in force.”
6. **Language** — use [CONTEXT.md](CONTEXT.md) terms. Disambiguate regulatory note vs research note; instrument vs PDF; evidence vs source vs citation.
7. **Scope** — change this FINAL package. Do not grow a parallel app tree unless the human redirects you.
8. **Checks** — after non-trivial backend logic, leave or run a small focused test under `backend/tests/`.

## Do not

- Weaken answer or currentness validation to “make the demo pass.”
- Treat `search_results` as a confirmed legal answer in copy or status mapping.
- Add ADRs for finished decisions; update CONTEXT / this file instead when language or trust rules change.
- Commit secrets, PDF corpora, or vector stores.
