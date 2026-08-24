# RAG reliability program - phase 1 baseline

Date: 2026-08-24

Status: retrieval baseline reproduced; evaluation freeze and registry scaffolding implemented; unseen splits and access-enforcement tooling remain pending; no production merge or production-index write.

## Verified current foundation

The retained experimental foundation is:

`StructuredDocument -> page-local sequential ~1000/200 chunks -> multilingual-e5-small dense top 20 + lowercase-whitespace BM25 top 15 -> exact union deduplication -> BGE reranker -> top 5`

The 697-case rerun reproduced the prior result exactly at every recorded rank, Top-5 source/page identity, reranker score, repair/regression label, and paired statistical result.

- Exact Page@5: 87.9536% (606/689 relevant cases)
- Exact Page@20: 93.1785%
- Source@1: 76.4877%
- Source@5: 92.7431%
- Source@20: 97.0972%
- Source MRR: 0.8327
- Repairs/regressions against the canonical hybrid baseline: 22/9, net +13
- Exact two-sided McNemar p-value: 0.0294494
- Original mean/median latency: 1.055s/0.976s
- Reproduction mean/median latency: 0.959s/0.897s

The latency difference is operational variance. It is not an accuracy improvement.

## Stage-aware development metrics

The candidate pool contains a mean 28.44 and median 29 chunks. Since the fixed dense/BM25 generators produce fewer than 50 unique candidates, pool recall is also the maximum available candidate recall for this configuration.

| Metric | Overall | French | Arabic |
|---|---:|---:|---:|
| Relevant cases | 689 | 377 | 312 |
| Candidate source-pool recall | 97.24% | 99.20% | 94.87% |
| Candidate exact-page pool recall | 93.47% | 95.76% | 90.71% |
| Dense exact Page@20 | 88.10% | 89.92% | 85.90% |
| BM25 exact Page@15 | 82.87% | 88.33% | 76.28% |
| Reranked Source@1 | 76.49% | 81.96% | 69.87% |
| Reranked Source@5 | 92.74% | 96.29% | 88.46% |
| Reranked Source@20 | 97.10% | 99.20% | 94.55% |
| Reranked Page@1 | 70.39% | 72.94% | 67.31% |
| Reranked Page@5 | 87.95% | 90.19% | 85.26% |
| Reranked Page@20 | 93.18% | 95.49% | 90.38% |

There are 45 exact-page candidate-pool misses and 38 further cases where the expected page is in the pool but not in the reranked Top 5. Candidate generation/extraction is therefore the first failure for slightly more cases, while ranking remains co-dominant.

## First-stage failure taxonomy

Primary categories are mutually exclusive and assigned at the first detected failure stage.

| Primary failure | Overall | French | Arabic |
|---|---:|---:|---:|
| Evidence missing because of extraction | 25 | 5 | 20 |
| Wrong temporal/document version | 12 | 10 | 2 |
| Chunk boundary/context problem | 2 | 2 | 0 |
| Correct document missing from candidates | 9 | 2 | 7 |
| Correct page missing from candidates | 10 | 6 | 4 |
| Evidence present but ranked below Top 5 | 25 | 12 | 13 |

The analyzer also records the 8 non-relevant cases separately: 2 ambiguous, 2 current-status-unknown, 2 not-in-corpus, and 2 out-of-scope. This retrieval-only phase does not test whether answer generation abstains or clarifies correctly on them.

## Arabic extraction finding

Development evidence supports Arabic extraction quality as an early bottleneck, but not mainly because pages are empty. Generalization remains unverified.

- 20 Arabic development failures were classified as extraction failures, versus 5 French.
- 17 of those 20 Arabic pages used native extraction, had quality score `1.0`, and had no quality flags.
- Those pages often had a high Arabic Unicode ratio while words, dates, digits, or spacing were semantically corrupted.
- The current quality gate measures text amount, broad character coverage, block presence, and obvious mojibake. It cannot detect fluent-looking Arabic corruption or digit transposition.
- Three failed pages from `Cir_2018_03_ar.pdf` had already gone through OCR; the OCR output was short Latin-heavy garbage. A fallback trigger alone is therefore insufficient: fallback outputs also need language-aware validation and native/OCR/VLM comparison.

This supports a controlled language-aware extraction-quality experiment. It does not yet prove that any particular OCR engine or VLM should be adopted.

### Frozen extraction stress screen

A 50-case development stress set was frozen before changing the gate:

- Arabic: 20 extraction failures + 20 unique language/category-matched Top-5 controls
- French: 5 extraction failures + 5 matched controls
- Unmatched controls: 0
- Canonical tracked stress-set SHA-256: `3A83F92AB5DFE0A90EB66281E22DDB22EC956811BB7F4DDAC0DD665807BA9496`

One exploratory Arabic trigger was screened after inspecting development feature distributions:

`current quality score < 0.55 OR Latin-character ratio > 0.20 OR single-Arabic-token ratio >= 0.10`

| Gate | Failure recall | False-positive rate |
|---|---:|---:|
| Current quality gate | 10% (2/20) | 0% (0/20) |
| Proposed trigger | 85% (17/20) | 5% (1/20) |

The continuation criterion was at least +25 percentage points recall with at most 10% false positives. The trigger passes that development screen and is retained only for a controlled fallback comparison. Because the threshold was explored and measured on the same development suite, 85%/5% is a resubstitution estimate and must not be treated as validation performance. It has not been wired into ingestion. The next experiment must establish whether OCR or VLM output is actually more faithful on triggered pages; otherwise the trigger has no downstream value.

## Evaluation protocol decision

The existing 697-case benchmark has been repeatedly inspected and is development data. Repartitioning it and renaming slices would not create a genuinely unseen validation or holdout set.

Before architecture/model selection:

1. Create new, independently verified validation questions and a separate final holdout.
2. Keep final holdout disjoint from both development and validation by conservative document family (`type:number` across years and languages). Validation and development overlap is audited because the legacy development suite already covers all current sources.
3. Hash all three datasets and the corpus manifest with `experiments.evaluation_protocol`.
4. Permit individual development-failure inspection.
5. Use validation periodically and record accesses.
6. Expose only aggregate final-holdout metrics; never tune from individual holdout failures.
7. Add a prospective future-document holdout when new BCT documents become available, because the legacy development suite already touches all 439 current sources.

The freeze tool fails closed for unknown validation/holdout filename conventions and requires an explicit `document_family`. A legacy development/holdout overlap override exists only to document a weaker protocol explicitly; it is disabled by default.

The current tool freezes hashes and leakage policy only. An append-only validation-access ledger and an aggregate-only final-holdout runner still need to be implemented before those datasets are used; policy strings alone do not enforce access discipline.

Until new validation and holdout questions are independently verified and frozen, any additional result is development evidence only and cannot establish generalization.

## Current decisions

- **KEEP:** `StructuredDocument` as canonical extraction/provenance representation.
- **KEEP:** page-local sequential approximately 1000/200 chunks as the current retrieval baseline.
- **KEEP:** fixed hybrid candidate union and BGE reranker as controls for isolating the next experiment.
- **KEEP FOR NEXT EXPERIMENT ONLY:** the language-aware Arabic risk trigger; do not deploy it until fallback fidelity and downstream retrieval improve on frozen validation data.
- **DO NOT CLAIM:** answer correctness, citation correctness, grounding, abstention, or generalization; none has been evaluated by this phase.
- **NEXT:** finish the remaining stress manifests and compare native/OCR/VLM only on a bounded, cached Arabic-heavy page set before any corpus-wide fallback change.

The extraction-quality set is only the first targeted suite. Tables, identifiers, numeric fidelity, temporal/versioning, near-duplicates, long documents, image annexes, context dependence, and ambiguity still require separate frozen manifests.

## Reproduction artifacts

- Evaluation SHA-256: `00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1`
- Canonical hybrid baseline SHA-256: `DC9917CF985250FA7453C7EEF7D5C31443345F7030BE5E20EEFF924E6D8C36C3`
- Structured ingestion manifest SHA-256: `09ABE18E9DCAD9043650EC1B204967642741CE6C2F1EE93F042B83375B59D5BA`
- Frozen candidate cache SHA-256: `A061D88BEE3F4C221EFC8612AC13272C83AA1C8A2EEA475409C2DC6E0CF65B33`
- Reproduced result SHA-256: `A785228676583BE0961F3163E56B3B2D9D33C8B0E9D5F8D5EAB0866A31F9562F`
- Reproduced stage analysis SHA-256: `BDB95A6C18E78FBF54E88C1A1B3B8BFE494829CDADA37764281846D04CB1089E`

Full reproduction directory:

`C:\Users\Moemen Super\AppData\Local\Temp\bct-rag-goal-reproduction-20260824-v1`
