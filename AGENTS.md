# Agents

Working tree: this repository root (`BCT-Regulatory-Search/`).
Preserve grounded-answer behavior. Prefer delete/reuse over new layers (ponytail **full**).

## Read first

| When | Open |
| --- | --- |
| Domain wording (instrument vs PDF, evidence vs citation, verified vs in force) | [CONTEXT.md](CONTEXT.md) |
| How to run backend, UI, ingest | [README.md](README.md) |
| Frontend API wiring and evidence panel | [frontend/README.md](frontend/README.md) |

## Contracts (edit here, not around them)

| Concern | Start in |
| --- | --- |
| Answer shape, claim/quote gates, answer statuses | `backend/answer_contract.py` (facade); gates in `answer_gates.py`, draft ladder in `answer_draft.py` |
| Currentness classifiers (`is_temporal_rule_query`, `is_relationship_query`) | `backend/query_currentness.py` |
| JSONL SUPERSEDES pin / ingest merge | `backend/jsonl_supersession.py` (facade); edges IO in `supersession_edges.py`, retrieve pin in `supersession_pin.py` |
| Admin review of relations (approve / reject / add by hand; decisions kept outside the index) | `backend/supersession_review.py`, `/admin/relations` in `app.py`, `frontend/src/components/admin/RelationsPage.tsx` |
| Conversation routing and follow-ups | `backend/conversation.py` |
| Profiles `cloud` / `local_hybrid` / `local` | `backend/runtime_profiles.py` |
| Answer model provider (OpenAI / Claude / Gemini / Groq / Ollama; one key is enough) | `backend/llm.py` (`answer_provider`, `create_llm`) |
| Nginx in front of the container | `deploy/nginx/bct-regulatory-search.conf` |
| Retrieval + evidence selection | `backend/runtime_retrieval.py` (local backend + `create_local_backend`), `backend/retrieval_selection.py` |
| Search speed on CPU (one reranker wording per script, `BCT_SPEED_MODE` 8-bit reranker) | `backend/runtime_retrieval.py` (`script_scores`), `backend/reranker.py` |
| PDF resolve, physical page, quote locate | `backend/source_documents.py` |
| Ingest → stage → activate | `backend/ingest.py`, `backend/ingestion/` |
| Upload queue, batched indexing (one asset version per batch), background enrichment (page ledger, chat priority, statuses) | `backend/ingestion/pipeline.py`, `enrichment.py`, `registry.py` |
| Docker volume seeding from the baked index (`SEEDED_FROM.txt`) | `backend/docker_serve.py` |
| Server log file | `backend/server_logging.py` |
| Visual ingest (EasyOCR / PaddleOCR-VL; `VisualPage` in `models.py`) | `backend/ingestion/local_visual.py`, `extract.py` |
| Optional cloud profile (Voyage search + index staging, Gemini page reading) | `backend/cloud/` — only called when the cloud profile, `BCT_INGEST_CLOUD_INDEX=1`, or `BCT_VISUAL_BACKEND=gemini` is on |
| HTTP surface | `backend/app.py`, `backend/run_api.py` |
| Auth / sessions / roles / admin audit log | `backend/identity.py` (audit rows written by `_audit` in `app.py`) |
| App profile + provider secrets | `backend/app_settings.py` |
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
