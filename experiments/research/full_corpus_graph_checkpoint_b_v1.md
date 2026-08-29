# Full-corpus graph checkpoint B v1

Status: predeclared before structural ingestion implementation

## Decision to make

Determine whether the unchanged 439-PDF corpus can be represented completely
and idempotently as instruments, source editions, pages, and retained retrieval
chunks in the persistent Checkpoint A Neo4j store.

Checkpoint B is structural only. It does not promote provision boundaries,
cross-references, or legal effects to verified semantic facts.

## Frozen inputs

- source corpus:
  `C:\Users\Moemen Super\BCT-Regulatory-Search\documents`
- current PDFs: 439
- cached structured-ingestion manifest:
  `C:\Users\Moemen Super\AppData\Local\Temp\bct-structured-ingestion-experiment-20260823-v3\ingestion_manifest.json`
- manifest SHA-256:
  `09ABE18E9DCAD9043650EC1B204967642741CE6C2F1EE93F042B83375B59D5BA`
- cached structured artifacts: 439 of 439 present
- current PDF hashes matching cached records: 439 of 439
- cached pages: 1,729
- retained structured baseline chunks:
  `C:\Users\Moemen Super\AppData\Local\Temp\bct-structured-ingestion-ablations-20260824-v1\representations\structured_baseline_chunking\chunks.jsonl`
- chunks SHA-256:
  `1E4300E460A7FF346764633AAE8BBF6A71965B635BBECC6707B2634B79DBECCD`
- chunks: 4,611 across all 439 source filenames and 1,727 pages
- aggregate SHA-256 over the sorted 439 structured-artifact hashes:
  `343F697BEE208FB72165B937A4529E3B0A52FF0F4BC600B9068D9DE9019BF9DC`
- languages: 166 French editions, 273 Arabic editions
- hosted calls: zero
- evaluation gold available to construction: false

The cached manifest is accepted only because every current PDF byte hash still
matches and every referenced artifact exists. Chroma and source files remain
read-only.

## Identity policy

The normal filename grammar identifies 430 instrument keys from authority BCT,
kind, year, and number. French and Arabic editions may share an Instrument only
when those identity fields agree. Language is edition metadata, not identity.

Nine filename variants require deterministic content-supported normalization
or an explicit ambiguity entry; they must not be silently guessed:

1. `CB_2017_08_FR.pdf`
2. `NB-2018_28_1110_fr.pdf`
3. `CI_2022_03_ar.pdf`
4. `Cir202204_fr.pdf`
5. `Cir202205_ar.pdf`
6. `NB_2022_13_ar.pdf`
7. `NB_2022_14_fr.pdf`
8. `NB_2022_15_fr.pdf`
9. `NB_2022_27_ar.pdf`

The first-page extraction visibly supports a candidate kind/year/number for
eight of these. They remain in the review queue until the parser records both
the filename rule and corroborating source text. Any unresolved case receives a
unique local Instrument with an identity-review status; it is never merged with
another edition speculatively.

## Public seams under test

1. A cache loader validates the corpus path, every PDF SHA-256, every artifact
   SHA-256, page counts, source uniqueness, and chunk source/page references.
2. A deterministic identity parser returns either a supported instrument key or
   a review-queue record.
3. A structural bundle builder emits stable UIDs and exact PDF/artifact/text
   hashes for every edition, page, and chunk.
4. A resumable structural writer skips an unchanged edition and confines a
   changed-PDF fixture update to that edition's structural subgraph.
5. A structural coverage receipt is derived from the graph after writing, not
   from intended input counts.

## Predeclared checks

1. 439 current PDFs, 439 source editions, and 1,729 pages are represented.
2. All 4,611 retained chunks are linked to existing pages, with deterministic
   within-page `NEXT_CHUNK` order.
3. Every source/page/chunk property hash agrees with the frozen cache and source
   inventory.
4. Every relationship has an allowlisted endpoint pattern and no endpoint is
   missing.
5. A second unchanged ingestion leaves the graph content hash unchanged.
6. A changed-PDF fixture changes only its isolated edition/page/chunk subgraph.
7. Identity ambiguities and the two pages without retained chunks are reported,
   not hidden.
8. Persistent restart, focused tests, repository tests, JSON validation, and
   `git diff --check` pass.

## Gate

Return `KEEP_CHECKPOINT_B` only if all structural counts and hashes reconcile
and both idempotency checks pass. Otherwise return `REJECT_CHECKPOINT_B`, retain
the failure receipt, and do not begin semantic relationship extraction.
