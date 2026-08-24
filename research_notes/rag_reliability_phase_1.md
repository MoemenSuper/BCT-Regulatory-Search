# RAG reliability program - phase 1 baseline

Date: 2026-08-24

Status: retrieval baseline reproduced; evaluation freeze, access enforcement, and registry tooling implemented; unseen splits remain pending; no production merge or production-index write.

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

The access boundary now verifies frozen hashes and exact case coverage, records
successful validation and holdout accesses in an append-only ledger, and rejects
case-level final-holdout output. Actual unseen datasets still need independent
curation before this boundary can be used for a generalization claim.

Until new validation and holdout questions are independently verified and frozen, any additional result is development evidence only and cannot establish generalization.

## Current decisions

- **KEEP:** `StructuredDocument` as canonical extraction/provenance representation.
- **KEEP:** page-local sequential approximately 1000/200 chunks as the current retrieval baseline.
- **KEEP:** fixed hybrid candidate union and BGE reranker as controls for isolating the next experiment.
- **KEEP FOR NEXT EXPERIMENT ONLY:** the language-aware Arabic risk trigger; do not deploy it until fallback fidelity and downstream retrieval improve on frozen validation data.
- **DO NOT CLAIM:** answer correctness, citation correctness, grounding, abstention, or generalization; none has been evaluated by this phase.
- **NEXT:** complete the bounded native/OCR/VLM fidelity comparison, then use the frozen targeted cohorts to isolate versioning, tables, long-document retrieval, context assembly, and abstention.

The extraction-quality and numeric-fidelity sets are supplemented by the
broader catalog below. Some requested categories remain proxies rather than
fully semantic annotations; those limits are preserved in the artifact.

## Phase 2 checkpoint: explicit Arabic OCR replacement

The frozen language-aware gate selected 18 Arabic development cases over 15
unique pages: 17 known extraction failures and one matched control. A cached
comparison held page selection fixed and measured three extraction variants:

- cached native Docling text;
- the current `OcrAutoOptions(force_full_page_ocr=True)` behavior;
- explicit `RapidOcrOptions(lang=["arabic"], backend="onnxruntime",
  force_full_page_ocr=True)`.

The installed Docling 2.120.2 auto selector resolved to RapidOCR's generic
PP-OCRv6 recognizer. The explicit arm resolved to the Arabic PP-OCRv5
recognizer. This is recorded execution behavior for the current environment,
not an assumption based on package names.

| Triggered development metric | Native | Auto OCR | Explicit Arabic OCR |
|---|---:|---:|---:|
| Failure evidence-token coverage (17) | 9.44% | 3.11% | 66.76% |
| Failure critical-number recall (12 numeric cases) | 51.19% | 30.65% | 37.50% |
| False-positive control evidence coverage (1) | 91.30% | 0.00% | 21.74% |
| Mean OCR latency per failure page | 0.00s cached | 3.81s | 2.82s |

The predeclared replacement gate required gains in semantic coverage and
critical-number recall without more than five points of control degradation.
The result is **REJECT**: explicit Arabic OCR fixes much of the Arabic word
recognition, but loses critical digits and damages the false-positive native
control. Auto OCR is worse than native on both evidence coverage and numbers.

This rules out replacing native text with either OCR output. It supports one
new development hypothesis: preserve native text and add explicit Arabic OCR as
a secondary candidate representation, so OCR can contribute readable words
without discarding native digits. That fusion must pass a separate cached
retrieval ablation and still cannot be deployed without unseen validation.

### Corpus-wide additive OCR retrieval

To avoid selecting only known failed pages, the unchanged gate was applied to
all 521 pages in the 273-document Arabic corpus. It selected 110 pages and
produced 196 page-local sequential 1000/200 OCR chunks. The experiment kept the
entire frozen native candidate pool, added at most five dense and five BM25 OCR
candidates only for runtime queries containing Arabic script, and reranked the
exact union with the unchanged BGE reranker.

| Retrieval metric | Current winner | Additive OCR | Delta |
|---|---:|---:|---:|
| Overall Source@1 | 76.49% | 77.65% | +1.16 pp |
| Overall Source@5 | 92.74% | 94.63% | +1.89 pp |
| Overall exact Page@1 | 70.39% | 70.25% | -0.15 pp |
| Overall exact Page@5 | 87.95% | 89.26% | +1.31 pp |
| Overall exact Page@20 | 93.18% | 94.63% | +1.45 pp |
| Arabic Source@5 | 88.46% | 92.63% | +4.17 pp |
| Arabic exact Page@5 | 85.26% | 88.14% | +2.88 pp |
| Arabic exact Page@20 | 90.38% | 93.59% | +3.21 pp |

The paired exact-Page@5 comparison has 12 repairs and 3 regressions (net +9,
two-sided exact McNemar `p=0.03515625`). French metrics are unchanged by design.
Mean/median measured retrieval-plus-reranking latency was 1.011s/0.927s,
compared with 0.959s/0.897s in the reproduction run; this is approximately a
5.4% mean increase, with the usual operational-variance caveat.

At candidate stage, Arabic exact-page pool recall increased from 90.71% to
94.23%. Eleven pages absent from the native candidate pool were recovered: nine
were prior extraction failures and two were prior missing-document failures.
The final reranker converted 12 cases into Top-5 repairs, but three current
rank-4/5 cases fell to rank 6. Multiple same-page native/OCR chunks also appear
in some Top-5 lists, identifying diversity and reranking pressure as the next
development failure seam.

**Decision: KEEP FOR UNSEEN VALIDATION, DO NOT DEPLOY.** This is an exploratory
development selection after failure inspection. It has not crossed a frozen
validation gate, and the small Page@1 decline, three cutoff regressions, numeric
OCR weakness, and duplicate-page pressure remain explicit risks.

### Numeric and identifier fidelity gate

A separate development suite was frozen from all 38 relevant cases whose
verified evidence contains numeric literals and whose expected page was already
selected by the unchanged Arabic fallback gate. Page selection therefore does
not depend on whether OCR preserved the expected number.

| Verified-literal metric | Native | Explicit Arabic OCR | Native + OCR union |
|---|---:|---:|---:|
| Mean critical-number recall | 67.45% | 27.37% | 72.93% |
| Cases with full number recall | 50.00% | 15.79% | 55.26% |
| Latin alphanumeric identifier cases | 4 | 4 | 4 |
| Mean identifier recall on those 4 cases | 100.00% | 100.00% | 100.00% |

OCR number recall improved over native in only 2 cases and regressed in 23.
The four identifier cases are too few to support a broader identifier claim.

This strengthens the architecture decision: OCR may be an additive retrieval
representation, but it is not authoritative evidence for digits. The answer
layer must preserve extraction provenance, prefer source-page/native agreement,
and flag or abstain on unresolved native/OCR numeric conflicts. A VLM fallback
must beat this frozen numeric suite as well as retrieval metrics; readable
Arabic prose alone is not sufficient.

### Frozen targeted development catalog

The remaining high-risk cohorts were frozen without consulting OCR-fusion or
VLM outputs. Selection uses only the 697-case development labels, the exactly
reproduced winner's failure diagnostics, and canonical StructuredDocument
page/block metadata.

| Cohort | Cases | Arabic | French | Selection note |
|---|---:|---:|---:|---|
| Structured table pages | 181 | 122 | 59 | Expected page contains a `table` block |
| Visual review, no table block | 88 | 77 | 11 | Image/visual-annex proxy only |
| Wrong-version diagnostic + controls | 34 | 12 | 22 | 17 failures + 17 unique matched controls |
| Documents with at least 20 pages | 53 | 0 | 53 | Arabic long-document coverage gap |
| Primary context failures + controls | 4 | 0 | 4 | 2 failures + 2 unique matched controls |
| Negative/ambiguous queries | 8 | 4 | 4 | All `relevant=false` cases |
| Latin alphanumeric identifiers | 42 | 20 | 22 | Numeric-only article references excluded |

The visual cohort does not prove that every page is a full-page image or annex,
and table-block presence does not prove the labeled answer lies inside the
table. The versioning cohort is a retrieval-diagnostic sample rather than a
complete amendments/supersession graph. These distinctions prevent the catalog
from claiming annotations it does not contain.

Canonical catalog SHA-256:
`296C670D8CCBB5B46D3830D19AA98337468B0F11931CA9D5C86797A48931B99C`.

### Targeted retrieval results

Applying the cached current-winner and additive-OCR ranks to those fixed
cohorts gives a more specific picture of where the retrieval gain lands:

| Cohort | Current Page@5 | Additive OCR Page@5 | Repairs | Regressions |
|---|---:|---:|---:|---:|
| Structured table pages (181) | 82.87% | 84.53% | 4 | 1 |
| Visual non-table proxy (88) | 70.45% | 76.14% | 7 | 2 |
| Wrong-version diagnostic + controls (34) | 50.00% | 55.88% | 2 | 0 |
| Long documents (53, French only) | 94.34% | 94.34% | 0 | 0 |
| Context failures + controls (4, French only) | 50.00% | 50.00% | 0 | 0 |
| Latin alphanumeric identifiers (42) | 100.00% | 97.62% | 0 | 1 |

French queries bypass additive OCR, so every French slice is unchanged. On
Arabic table cases, Page@5 rises from 88.52% to 90.98% (4 repairs, 1
regression). On Arabic visual non-table cases it rises from 67.53% to 74.03%
(7 repairs, 2 regressions), though Page@1 falls from 45.45% to 42.86%. The two
version-cohort repairs are Arabic extraction rescues; additive OCR does not
solve French version discrimination. Long-document retrieval is already strong
within its French-only development coverage, while the two known context
failures remain unresolved.

The identifier regression is
`note_2021_09_ar_eligibility_or_scope_01`, which moves from rank 5 to 6. This
confirms that a semantic OCR gain can still crowd exact identifiers at the
final cutoff. It strengthens the next controlled hypothesis: page-level
diversity or identifier-aware ranking must preserve the new candidate recall
without allowing redundant OCR/native candidates to displace exact evidence.

Canonical targeted retrieval result SHA-256:
`90A5BADE399E29467AEDC78D901E762DC00D0B4A095A2AB2540C6F3B4F2EF20F`.

### Post-reranker source-page diversity

The full additive-OCR candidate union was reranked again with the unchanged
BGE model. Every undiversified exact-page rank matched the tracked fusion run,
which verifies that page diversity is the isolated change. Stable filtering
then retained only the highest-scored chunk for each case-folded source/start
page.

| Exact-page metric | Additive OCR | Page-diverse OCR | Delta |
|---|---:|---:|---:|
| Overall Page@1 | 70.25% | 70.25% | 0.00 pp |
| Overall Page@5 | 89.26% | 89.40% | +0.15 pp |
| Overall Page@20 | 94.63% | 94.78% | +0.15 pp |
| Arabic Page@1 | 66.99% | 66.99% | 0.00 pp |
| Arabic Page@5 | 88.14% | 88.46% | +0.32 pp |
| Arabic Page@20 | 93.59% | 93.91% | +0.32 pp |

The predeclared gate required no Page@1 or Page@5 regressions overall or by
language and at least one Page@5 repair. It passes with one repair and zero
regressions versus additive OCR. The repair is
`note_2018_31_ar_effective_date_01`, restored from rank 6 to 5 after a duplicate
native/OCR page ahead of it was collapsed. Across all 697 queries, 259 candidate
lists contained page duplicates; stable filtering removed 1,444 duplicate-page
candidates and improved 11 exact-page ranks. French ranks are unchanged.

**Decision: KEEP FOR UNSEEN VALIDATION, DO NOT DEPLOY.** The gain is small and
mechanically monotonic for first-page occurrence. A second chunk from the same
page may still carry useful answer context, so the eventual context builder
should select diverse evidence pages first and then reattach provenance-linked
same-page or neighboring context. This retrieval-only ablation does not prove
answer quality.

Tracked result SHA-256:
`1A0ABFD6D97CDAC7AC102987B1E274DD7958295A9E18D2577B694C9F566CDE40`.

## Reproduction artifacts

- Evaluation SHA-256: `00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1`
- Canonical hybrid baseline SHA-256: `DC9917CF985250FA7453C7EEF7D5C31443345F7030BE5E20EEFF924E6D8C36C3`
- Structured ingestion manifest SHA-256: `09ABE18E9DCAD9043650EC1B204967642741CE6C2F1EE93F042B83375B59D5BA`
- Frozen candidate cache SHA-256: `A061D88BEE3F4C221EFC8612AC13272C83AA1C8A2EEA475409C2DC6E0CF65B33`
- Reproduced result SHA-256: `A785228676583BE0961F3163E56B3B2D9D33C8B0E9D5F8D5EAB0866A31F9562F`
- Reproduced stage analysis SHA-256: `BDB95A6C18E78FBF54E88C1A1B3B8BFE494829CDADA37764281846D04CB1089E`

Full reproduction directory:

`C:\Users\Moemen Super\AppData\Local\Temp\bct-rag-goal-reproduction-20260824-v1`
