# Incremental graph reference ingestion v1

Status: predeclared before implementation

## Objective

Implement the first reusable graph-enrichment stage for one validated uploaded
PDF. The stage detects explicit BCT instrument citations, resolves their stable
instrument identities, preserves exact page evidence, creates external stubs
when a cited instrument is absent, and exposes a fail-closed promotion boundary
before Neo4j persistence.

This is an ingestion seam, not a second full-corpus script. A future upload/job
worker will call it after source hashing and structured page extraction and
before activating the document for GraphRAG retrieval.

## Public seams under test

1. `extract_document_reference_candidates` accepts one immutable source edition
   plus its ordered extracted pages and returns deterministic exact-source
   reference candidates.
2. `promote_reference_candidate` returns a verified graph reference only when
   source hash, source page, rendered-page evidence, and target identity checks
   all pass.
3. `Neo4jGraphWriter.write_bundle` refuses any non-verified instrument reference
   and writes verified reference lineage idempotently through allowlisted typed
   relationships.

## Scope

- French and Arabic explicit BCT circular/note citations with year and number;
- stable target UIDs compatible with corpus instrument identity;
- exact source filename, edition UID, PDF/artifact hash, page, quote, signal,
  offsets, extraction rule, and verification state;
- `EXTERNAL_STUB` targets when the cited BCT instrument is not in the corpus;
- a dedicated `InstrumentReference` graph node rather than a generic
  `RELATED_TO` edge;
- exact evidence lineage from source instrument to reference, target
  instrument, evidence span, and source page.

## Explicit exclusions

- no production upload endpoint, SQL job ledger, Chroma mutation, or merge to
  `main` in this slice;
- no benchmark/gold access;
- no automatic promotion of `AMENDS`, `REPLACES`, `ABROGATES`, or other legal
  effects from nearby keywords;
- no claim that a citation changes the cited instrument;
- no current/as-of state derived from reference candidates.

## Real-source verification anchor

`CB_2017_08_FR.pdf`, page 28, visibly contains Article 63 and the clause
"La presente circulaire abroge et remplace la circulaire n 2013-15...". This
proves that one exact citation can coexist with two legal-action signals. The
citation identity and the legal effect must therefore remain separate domain
facts; this slice implements only the citation fact and keeps legal-effect
promotion outside the automatic path.

## Gate

KEEP only if candidate extraction is deterministic, French and Arabic digits
resolve without reversal, external stubs are explicit, incomplete promotion
fails closed, Neo4j relationship patterns stay allowlisted and auditable,
unchanged writes are idempotent, focused/full tests pass, and the persistent
Checkpoint B graph is not mutated by the development tests.
