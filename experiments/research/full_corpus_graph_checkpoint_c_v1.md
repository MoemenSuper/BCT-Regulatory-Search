# Full-corpus semantic graph checkpoint C v1

Status: predeclared before corpus-wide semantic candidate extraction

## Decision to make

Determine how much of the 439-edition corpus can be promoted from structural
provenance into source-evidenced provisions, cross-references, and temporal
change relationships without treating a regex or model proposal as legal fact.

Checkpoint C starts from the kept Checkpoint B graph at commit `1e78054`, graph
SHA-256 `86F24BB172A7E75D5FAF46A79C493D6A218AAFF18A6B3945B557DAA159B614E1`.
It must not consume evaluation questions, expected sources/pages, or answers.

## Frozen inputs

- 439 hash-verified cached StructuredDocument artifacts from the Checkpoint B
  manifest;
- 1,729 graph-linked pages and 4,611 retained chunks;
- graph schema version `regulatory-graph-structural-v1`;
- the existing manually source-verified `Cir_2016_03_fr.pdf` tracer;
- hosted calls: zero for the deterministic candidate pass.

Source PDFs, production Chroma, evaluation gold, and the validated Checkpoint B
graph are read-only during candidate discovery.

## Candidate extraction policy

The corpus-wide first pass detects only observable source signals:

1. explicit French or Arabic article headings;
2. explicit French or Arabic legal-action terms for add, replace, modify,
   abrogate, derogate, complete, or insert;
3. explicit instrument/provision reference forms.

Every record carries filename, immutable edition UID, PDF/artifact hash, page,
exact source substring, source offsets, signal type, deterministic rule, and a
stable content-derived candidate UID. The output is a committed review queue.

All newly detected records start `NEEDS_REVIEW`. Detection alone does not create
`Provision`, `ProvisionVersion`, `ChangeEvent`, `TARGETS`, `INTRODUCES_VERSION`,
`RETIRES_VERSION`, `CURRENT_VERSION`, or cross-reference relationships in
Neo4j. This avoids false ownership when amendment text quotes a provision from
another instrument and avoids inventing predecessor text.

## Promotion contract

A candidate may be promoted to `VERIFIED` only when all applicable checks pass:

- the source PDF hash and page agree with Checkpoint B;
- the exact evidence quote is visibly confirmed on the rendered PDF page;
- source and target instrument identity are unambiguous;
- provision scope and ownership are unambiguous;
- the legal action and affected span are explicit;
- the effective date or unresolved trigger is recorded exactly;
- predecessor text exists when replacement lineage claims it;
- interval overlap and relationship endpoint validation pass locally.

Anything failing a check remains `NEEDS_REVIEW`; malformed Arabic, conflicting
dates, missing predecessor text, and broad/partial replacements are enumerated
rather than normalized.

## Public seams under test

1. A deterministic extractor returns stable source-linked candidates from a
   page and never returns `VERIFIED`.
2. Evidence quotes and offsets round-trip exactly into cached page text.
3. The full-corpus runner writes a deterministic JSONL review queue and summary
   receipt, both content-hashed.
4. A promotion validator fails closed unless source, target, evidence,
   effective-state, and predecessor requirements are satisfied.
5. Neo4j content remains unchanged when only unresolved candidates exist.

## Predeclared reporting

- candidate counts by type, proposed action, language, document, and page;
- documents/pages with and without detected semantic signals;
- verified, candidate, and review-needed counts;
- malformed or source-offset failures;
- unresolved targets, effective dates, predecessor text, and visual checks;
- graph counts/hash before and after the candidate pass;
- focused/repository tests, JSONL validation, and `git diff --check`;
- hosted request count, latency, and cost status.

## Gate

Return `KEEP_CHECKPOINT_C` only if corpus-wide promoted relationships satisfy
the promotion contract and semantic coverage gaps are explicitly bounded.
Otherwise return `REJECT_CHECKPOINT_C_INCOMPLETE_VERIFICATION`, preserve the
review queue and unchanged graph, and do not claim temporal GraphRAG readiness.

Checkpoint D may not treat `NEEDS_REVIEW` records as graph facts.
