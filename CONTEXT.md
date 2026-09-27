# BCT Regulatory Search

Ubiquitous language for the consolidated prototype: grounded answers over BCT circulars and notes, with inspectable PDF evidence. Glossary only. No implementation detail.

## Corpus

**Instrument**:
A named BCT regulatory act identified by kind, year, and number (canonical id shape `cir:2017:8`).
_Avoid_: document, source, PDF, file, texte

**Circular** (`cir`):
A BCT circulaire, the primary instrument kind.
_Avoid_: document, note, “circular” when you mean any instrument

**Regulatory note** (`note`):
A BCT note réglementaire, distinct from a circular.
_Avoid_: research note, Note de recherche réglementaire, mémoire

**Provision**:
A named article, section, or disposition inside an instrument.
_Avoid_: passage, chunk, claim, rule (when you mean a numbered provision)

**PDF**:
The authoritative file artifact for an instrument (basename + physical pages).
_Avoid_: instrument (when you mean legal identity), document (when identity matters)

**Physical page**:
The 1-based page number as shown in the PDF viewer.
_Avoid_: chunk index, legacy 0-based page, “page” without saying physical

## Retrieval and answers

**Passage**:
A human-facing excerpt of regulatory text (search hit or support text).
_Avoid_: evidence (until selected), chunk, provision, source

**Chunk**:
A page-local indexed unit of text used for retrieval. Not a legal unit.
_Avoid_: passage, evidence, provision

**Evidence**:
A retrieved passage that may support an answer: trusted filename, physical page, and text (labeled `E1`, `E2`, …). Retrieval ranks chunks; the answer layer receives the retrieved page's text (bounded), rebuilt from the same indexed chunks.
_Avoid_: source, citation, quotation, search result (when not yet selected)

**Evidence warning**:
Observable OCR noise in a page header (`source_header_conflict`, `implausible_gregorian_year`). The passage stays usable: identity comes from the trusted filename, quotations stay digit-exact. Distinct from `unusable_reason` (ingestion-detected extraction conflict), which blocks claims.
_Avoid_: unusable, corrupt document, “bad PDF” (the PDF is authoritative; the extraction is noisy)

**Quotation**:
A verbatim contiguous excerpt recoverable from the supplied physical page text. Locating tolerates spacing and small OCR letter noise; digits and negations must match exactly, and the rendered text is always the page's own characters.
_Avoid_: citation, paraphrase, summary, highlight alone

**Citation**:
A numbered UI pointer (`[1]`, `[2]`) built only from retrieval metadata (`file` + `page`).
_Avoid_: quotation, evidence id, Graph `CITES`

**Claim**:
An atomic answered statement backed by one or more quotations from selected evidence.
_Avoid_: answer (whole response), message, unsupported synthesis

**Grounded answer**:
An answer where every factual claim rests on literal quotations from selected evidence.
_Avoid_: “AI summary”, free-form legal advice

**Runtime profile**:
Which retrieval + answer stack is live: `cloud`, `local_hybrid`, or `local`.
_Avoid_: mode, backend, provider (as the profile name); treating a profile as legal qualification

**Answer status**:
Product outcome of a turn: `answered`, `partial_answer`, `insufficient_evidence`, `clarification_needed`, `out_of_scope`, `search_results`.
`partial_answer` may also present a grounded synthesis from the top retrieved pages (still with quotations), listing those pages for inspection and asking for admin/PDF confirmation. When that synthesis uses more than one page, the answer states that the useful elements are spread across multiple passages (after the claims). If drafting/repair still cannot form valid JSON, complete claims are salvaged from truncated model output when possible. Forced partial feeds the writer compact on-topic snippets and requires natural claim sentences (quotes stay literal). `search_results` remains the last resort when no quoted claim can be formed.
Greetings, help, and conversation-summary turns use the router’s `GENERAL_CHAT` path: a short reply with **no retrieval** (often `answered` with empty sources). Broad regulatory briefings still retrieve; the selector/writer prefer a multi-page **quoted** `partial_answer` when possible, without weakening quote checks. An empty model completion is treated as a draft failure (`draft_empty`), not repaired into `insufficient_evidence`. Selector evidence IDs are sanitized (keep valid IDs; map `1`/`e1` → `E1`) instead of discarding a whole selection for one bad ID.
_Avoid_: “success” / “failure” as the only labels; treating `search_results` as a confirmed legal answer; leading with “couldn’t find” when a multi-page synthesis is available; treating `GENERAL_CHAT` as a fake out-of-scope refusal; treating a blank LLM completion as a deliberate abstention; pasting raw page/OCR text as the claim body

## JSONL supersession

**Supersession edges**:
File-backed SUPERSEDES / REPLACE / ABROGATE / AMEND links between instruments (`supersession_edges.jsonl` in the active asset version). Merged on ingest; used at retrieve time to pin successor declaring pages. Not a fourth runtime profile.
_Avoid_: Neo4j / knowledge graph as the main search; “related docs” without a typed edge

**Relation / action**:
Operative actions such as `REPLACE`, `ABROGATE`, `AMEND` (JSONL), surfaced to the answer layer as `temporal_relation` / `graph_guidance`.
_Avoid_: related, link, predecessor/successor as stored types without an action

**Pinned relationship**:
A JSONL edge that matched the query or top hits and contributed a declaring page with `temporal_relation` metadata.
_Avoid_: “in force”, “currently applicable”, perfect legal interpretation

**Relationship only (not provision-resolved)**:
An edge or `temporal_relation` proves instrument succession, not provision-level temporal applicability. When evidence carries `temporal_relation` / `relationship_note` for REPLACE / ABROGATE / AMEND, the answer layer keeps both instruments, marks successor vs superseded, and instructs the writer to state the relationship then follow the successor for conflicted facts.
Absence of a retrieved amending text is not proof that none exists; claims that “no later text modifies” an instrument require a literal quote.
_Avoid_: verified as synonym for current / en vigueur; silently dropping replaced circulars; merging conflicting values from predecessor and successor; treating missing amendment hits as a negative proof

**Historical cutoff vs grandfathering**:
`avant` / `before` / `قبل` demotes a named later instrument only when it directly precedes the instrument reference (“avant la circulaire 2025-13”) or the question says “ancien régime”. The same words do **not** demote when they describe something done before it (“engagements pris avant …”, “avant le 26 mars”, “avant l’entrée en vigueur de …”): those transitional / grandfathering questions answer from the named instrument.
_Avoid_: treating every “avant 2026-04” as a prior-regime retrieval

## Conversation and UI

**Conversation**:
A saved Q&A thread reopenable from the history sidebar.
_Avoid_: session (unless you mean the same thread), research note

**Research note**:
The UI card that presents one regulatory turn (Note de recherche réglementaire).
_Avoid_: regulatory note (`note` instrument)

**Source viewer**:
Right-hand panel with **Preuve** / **Passage** (rendered page + quote highlight) and **Document** (full PDF).
_Avoid_: Evidence panel = evidence object; Document tab = Instrument

**Source** (UI):
A citation row or PDF basename shown to the user.
_Avoid_: evidence (backend object), instrument

## Ingestion and trust

**Asset version**:
A staged snapshot of retrieval assets; the live corpus is the activated version.
_Avoid_: instrument year, PDF revision, deploy

**Staged activation**:
New assets stage first; activation makes them live. Failure keeps the previous corpus.
_Avoid_: upload as already-searchable; “hot reload” without activation

**Quick pass / enrichment**:
Ingest is two-phase. The quick pass (native PyMuPDF text only, seconds) activates a version so the PDF is searchable at once. Pages that need visual reading (scanned / garbled / chart / Arabic risk pages) are queued per page in the ingestion ledger; the background enrichment worker in the API process reads them one at a time (unreadable pages first), checkpoints each result, and re-activates in batches through the same staged activation. Chat requests take priority: the worker pauses between pages while a question is being answered. A page that needs visual reading to be quotable (Arabic mode `all`) has no page text until it is read — never native text standing in for it.
_Avoid_: “fully ingested” for an `enriching` PDF; OCR inside the upload request

**Document status**:
`enriching` (searchable on native text; visual reading in progress, with progress/ETA), `ready` (every planned page read), `ready_degraded` (searchable; some pages could not be read visually after retries — native text only on those pages; admin can re-queue them), `failed` (quick pass failed; nothing activated). A `processing` row left by a killed server is marked `failed` at startup.
_Avoid_: showing `ready_degraded` as `ready`; treating `enriching` as an error

**Trusted document root**:
A configured corpus path (or ingestion ledger entry) allowed to resolve PDFs for viewing and identity.
_Avoid_: arbitrary client file path

**Immutable copy**:
Content-addressed stored PDF used for viewing and search after ingest.
_Avoid_: original path as the only identity after ingest

**Not legal advice**:
Product stance: research aid. The original PDF is authoritative; open the cited page before operational use.
_Avoid_: “qualified legal opinion”, “en vigueur” claims when temporal scope is incomplete

**Document kind (`doc_kind`)**:
Admin tag at ingest: `regulatory` (primary), `statistical`, or `internal` (secondary). One corpus; claim grounding uses kind, not retrieval silos. Works with any search provider (Voyage, Google, local). Hard pages (scans, charts, image notes) use a profile-selected visual backend at ingest: EasyOCR (Arabic) + PaddleOCR-VL (charts/tables/hard pages) for `local` / `local_hybrid`; Gemini VLM (`GEMINI_API_KEY`) for `cloud`. After transcription the active embedder indexes that text.
_Avoid_: treating a bulletin or memo as a binding circulaire

**Query class**:
Turn label for which kinds may prove claims: `regulatory_rule` | `statistical_fact` | `internal_procedure` | `mixed` | `uncertain`. `uncertain` grounds as regulatory; never rejects the question.
_Avoid_: classifier as a refusal gate

**Chart / figure page**:
A PDF page flagged by geometry (image/drawing area). When chart vision is on, the active visual backend always reads that page (during enrichment) and merges chart notes into the native body — even when the page already has rich extractable text (typical stats layout). Quotes still need page text.
_Avoid_: embedding vector as proof of a number
