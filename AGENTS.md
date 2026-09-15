# Agents

Working tree for this package: `Final output/BCT-Regulatory-Search-FINAL/`.
Preserve grounded-answer behavior. Prefer delete/reuse over new layers (ponytail **full**).

## Read first

| When | Open |
| --- | --- |
| Domain wording (instrument vs PDF, evidence vs citation, verified vs in force) | [CONTEXT.md](CONTEXT.md) |
| How to run backend, UI, Graph Lite, ingest | [README.md](README.md) |
| Graph trust gates and relation types | [backend/GRAPH_LITE.md](backend/GRAPH_LITE.md) |
| Frontend API wiring and evidence panel | [frontend/README.md](frontend/README.md) |

## Contracts (edit here, not around them)

| Concern | Start in |
| --- | --- |
| Answer shape, claim/quote gates, answer statuses | `backend/answer_contract.py` |
| Graph payload / currentness wording | `backend/graph_contract.py` |
| Conversation routing and follow-ups | `backend/conversation.py` |
| Profiles `cloud` / `local_hybrid` / `local` | `backend/runtime_profiles.py` |
| Retrieval + evidence selection | `backend/runtime_retrieval.py`, `backend/retrieval_selection.py` |
| PDF resolve, physical page, quote locate | `backend/source_documents.py` |
| Graph Lite runtime / bootstrap | `backend/graph_lite.py`, `backend/regulatory_graph_lite/` |
| Ingest → stage → activate | `backend/ingest.py`, `backend/ingestion/` |
| HTTP surface | `backend/app.py`, `backend/run_api.py` |
| Auth / sessions / roles | `backend/identity.py` |
| App profile + provider secrets | `backend/app_settings.py` |
| UI turn presentation | `frontend/src/` (`ResearchNote`, `EvidencePanel`, `api/chat.ts`) |
| Login / admin UI | `frontend/src/Root.tsx`, `LoginPage`, `AdminDashboard` |

## Hard rules

1. **Grounded citations** — filenames and pages come from retrieval metadata. Do not invent paths or pages in prompts, fixtures, or UI copy.
2. **Quotations** — claims need verbatim page text. Fail closed to `insufficient_evidence` / `search_results` rather than paraphrase support.
3. **Trusted paths** — resolve PDFs only via trusted document roots or the ingestion ledger.
4. **Staged activation** — new asset versions stage before they go live; a failed activation keeps the old corpus.
5. **Graph Lite** — optional sidecar. Edges are `CITES` / `AMENDS` / `REPLACES` / `ABROGATES`. `VERIFIED` means relationship checks passed, not “currently in force.”
6. **Language** — use [CONTEXT.md](CONTEXT.md) terms. Disambiguate regulatory note vs research note; instrument vs PDF; evidence vs source vs citation.
7. **Scope** — change this FINAL package. Do not grow a parallel app tree unless the human redirects you.
8. **Checks** — after non-trivial backend logic, leave or run a small focused test under `backend/tests/`.

## Do not

- Weaken answer or graph validation to “make the demo pass.”
- Treat `search_results` as a confirmed legal answer in copy or status mapping.
- Add ADRs for finished decisions; update CONTEXT / this file / GRAPH_LITE instead when language or trust rules change.
- Commit secrets, PDF corpora, vector stores, or Neo4j data.
