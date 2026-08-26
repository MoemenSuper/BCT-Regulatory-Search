# Document identity candidate selection v1

Status: predeclared before implementation or result scoring

## Hypothesis

When a query explicitly identifies a BCT instrument or contains a narrow
document-year phrase, adding a compact canonical source identity to the BGE
reranker input will improve source/version and exact-page selection without
changing retrieval for un-routed queries or regressing routed controls.

This experiment does not infer an unstated document version. Queries without a
runtime-observable identity signal keep the exact current reranker input and
ordering.

## Single changed variable

The only changed retrieval variable is the text passed to
`BAAI/bge-reranker-v2-m3` for routed queries. Each candidate receives a compact
prefix derived from its runtime filename metadata:

`[document kind=<kind>; year=<year>; number=<number>; language=<language>]`

The candidate's original chunk text follows unchanged. Dense retrieval, BM25,
Arabic OCR routing, candidate budgets, exact union/deduplication, reranker
model, and stable source/page diversity remain fixed.

## Runtime-observable routing policy

Source identities are parsed only from candidate filename metadata matching the
existing BCT family form, including `Cir_2018_03_ar.pdf` and
`Note_2021_05_fr.pdf`.

A query is routed only when one of these signals is present:

1. A filename-like or natural-language circular/note reference that supplies a
   kind, four-digit year, and instrument number.
2. Exactly one year in a narrow source-year phrase such as French `en 2018`,
   `type 2017`, `publiee en 2021`, or Arabic `sana 2018`, `li-sana 2018`, or
   `li-am 2018` in their native spellings.

The parser must exclude years that occur only in a law reference, full calendar
date, agricultural season/range, accounting exercise, or generic numeric
literal. Current/future requests are not converted into historical source-year
constraints.

The policy may use query text and candidate filename metadata. It must not use
case IDs, expected source/page, expected answers, evidence quotes, or expected
literals to route or rank candidates. Gold fields are used only after the
routed ranking is frozen, for scoring.

If no signal is present, candidate objects and reranker text must be byte-for-
byte equivalent to the control path. Malformed or unrecognized source names do
not receive an invented identity.

## Controls and fixed inputs

- Development evaluation SHA-256:
  `00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1`
- Native candidate cache SHA-256:
  `A061D88BEE3F4C221EFC8612AC13272C83AA1C8A2EEA475409C2DC6E0CF65B33`
- Native representation manifest SHA-256:
  `920159F84E87353256899370A2E8B7854BAB87FBBBDC05184FED7C3657B439C1`
- Additive Arabic OCR representation manifest SHA-256:
  `8CBD9A468B2D017DB46218252B555231460A64381245786CCA020C84F3CD05AB`
- Frozen additive-OCR result SHA-256:
  `045EC8D869F5D31730187A74F0A9C779910B1A772E23482BB4FB29CF212950A0`
- Targeted development catalog SHA-256:
  `296C670D8CCBB5B46D3830D19AA98337468B0F11931CA9D5C86797A48931B99C`
- Answer-safety suite SHA-256:
  `5E3D840B6DF8FDF3046AA2A5A67B84DF8CC0E0E61D4EC810E5D51446C89917A1`

The temporal-near-duplicate cohort and its deterministic controls are frozen in
the targeted catalog. No provisional answer validation or final-holdout data
will be accessed.

## Tests before implementation

Focused tests must cover:

- French, Arabic, and filename-like full identity parsing;
- narrow year routing and the exclusions above;
- source filename normalization;
- exact no-op behavior for un-routed queries;
- identity-prefix provenance and preservation of the original chunk text;
- malformed source handling;
- fixed candidate/page budgets and stable source/page diversity;
- scoring that uses gold only after routing and ranking are frozen.

## Execution and scoring

Run the control and candidate arms over all 697 frozen development queries. For
each stage, persist candidate count, route reason, parsed query identity,
control and candidate source/exact-page ranks, final top five, repairs,
regressions, and latency. Report overall, French, Arabic, routed-only, and the
frozen temporal-near-duplicate failure/control cohort separately.

No hosted answer calls are allowed unless the retrieval component gate passes.
If it passes, rerun only answer-suite cases whose final evidence changed,
recompose all untouched records from the frozen raw retrieval-first result, and
manually review every changed response. Otherwise record the retrieval result
and stop without spending hosted calls.

## Predeclared component gate

KEEP for answer-combination testing only if all clauses pass:

1. At least two frozen temporal-near-duplicate failure cases are repaired at
   exact Page@5.
2. Zero frozen temporal-near-duplicate control cases regress at exact Page@5.
3. Overall, French, and Arabic exact Page@5 do not decrease.
4. Un-routed queries have identical ranked source/page order and scores.
5. Candidate counts, per-channel budgets, exact union, reranker model, and
   source/page diversity remain fixed.
6. Routing and ranking use no gold fields.

Otherwise REJECT this exact identity-prefix policy. A KEEP remains
development-only and does not authorize validation, production changes, or
wholesale metadata injection into retrieval text.

