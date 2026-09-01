# Connected-reference GraphRAG v1

Status: predeclared before implementation

## Objective

Make the existing local Neo4j graph materially useful for real natural-language
questions by populating verified source-to-instrument citation relationships
across the frozen 439-PDF corpus and by consulting graph connectivity for the
ordinary retrieval seeds instead of requiring a narrow keyword route.

## Design decision

A trained or fine-tuned LLM will not be the primary GraphRAG router in this
slice. The runtime can cheaply ask local Neo4j whether the top-five retrieved
documents have verified relationships and let the existing reranker decide
whether that evidence is relevant to the user's exact wording. This is
language-agnostic, bounded, observable, and does not make a probabilistic model
the authority for legal status.

An LLM classifier remains a possible fallback for future temporal intent and
date normalization. It is not required for ordinary relationship expansion.

## Public seams under test

1. `RegulatoryGraphRetriever.retrieve(query, seed_documents)` checks verified
   relationship evidence for valid retrieval seeds even when the query does
   not match a keyword template. No seeds still means no graph call.
2. `verify_reference_candidate` promotes an explicit BCT circular/note
   citation only when catalog identity, immutable PDF hash, rendered page,
   occurrence count, and exact target all agree.
3. `Neo4jGraphWriter.write_bundle` receives only verified citation facts and
   writes them idempotently with exact source/page evidence.

## Corpus backfill

The frozen structured inventory is scanned without evaluation or gold access.
Every exact French or Arabic BCT circular/note citation is independently
verified against its source PDF. Successful citations are stored as:

`Instrument -[:DECLARES_REFERENCE]-> InstrumentReference
 -[:TARGETS]-> Instrument`

with a separate `EVIDENCED_BY` path to the exact source page and quote.

Targets absent from the 439-PDF corpus remain explicit `EXTERNAL_STUB`
instruments. Self-references and any hash, page, occurrence, rendering, or
catalog mismatch remain `NEEDS_REVIEW`.

## Safety boundary

- An automatically verified citation means only `CITES`.
- It must not automatically become `AMENDS`, `REPLACES`, or `ABROGATES`.
- Ambiguous legal effects and incomplete temporal predecessor lineages remain
  review-gated and fail closed.
- The frozen evaluation questions, answers, expected documents, and expected
  pages are unavailable to candidate extraction, verification, and graph
  writes.
- The persistent graph is not written until the same bundle passes an empty
  disposable Neo4j run, idempotent repeat, evidence retrieval check, and
  cleanup.

## Gates

- all promoted references pass immutable rendered-page verification;
- zero unverified references reach the writer;
- at least 100 corpus documents receive one verified relationship, unless the
  rendered-PDF verifier explicitly accounts for the shortfall;
- repeated writes are idempotent;
- the frozen 50-case runtime evaluation returns graph evidence for at least one
  real case and reports repairs/regressions separately;
- focused and full repository tests pass;
- persistent before/after snapshots and a reproducible receipt are recorded.
