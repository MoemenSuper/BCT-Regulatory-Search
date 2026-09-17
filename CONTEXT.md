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
`partial_answer` may also present a grounded synthesis from the top retrieved pages (still with quotations), listing those pages for inspection and asking for admin/PDF confirmation. When that synthesis uses more than one page, the answer states that the useful elements are spread across multiple passages. `search_results` remains the last resort when no quoted claim can be formed.
_Avoid_: “success” / “failure” as the only labels; treating `search_results` as a confirmed legal answer; leading with “couldn’t find” when a multi-page synthesis is available

## Graph Lite

**Graph Lite**:
Optional Neo4j sidecar of relationships between instruments (and named provisions). Not a fourth runtime profile.
_Avoid_: knowledge graph as the main search; “related docs” without a typed edge

**Relation type**:
One of `CITES`, `AMENDS`, `REPLACES`, `ABROGATES`.
_Avoid_: related, link, predecessor/successor as stored types

**Verified relationship**:
A live Graph Lite edge that passed deterministic quote and instrument checks (`VERIFIED` / `AUTO_DETERMINISTIC_V1`).
_Avoid_: “in force”, “currently applicable”, perfect legal interpretation

**Verified relationship only**:
Graph proves a relationship edge, not provision-level temporal applicability (`temporal_verification=VERIFIED_RELATIONSHIP_ONLY`).
_Avoid_: verified as synonym for current / en vigueur

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

**Trusted document root**:
A configured corpus path (or ingestion ledger entry) allowed to resolve PDFs for viewing and identity.
_Avoid_: arbitrary client file path

**Immutable copy**:
Content-addressed stored PDF used for viewing and search after ingest.
_Avoid_: original path as the only identity after ingest

**Not legal advice**:
Product stance: research aid. The original PDF is authoritative; open the cited page before operational use.
_Avoid_: “qualified legal opinion”, “en vigueur” claims when temporal scope is incomplete
