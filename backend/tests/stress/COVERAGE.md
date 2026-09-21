# Stress coverage matrix (Suite A — engineering / adversarial)

This folder is **Suite A**. For the human / production-style benchmark, see
[`../benchmark/`](../benchmark/) (**Suite B**).

Executable suite: `backend/tests/stress/`. Failures print `Failure stage = <Stage>`.

| Stage | Question |
| --- | --- |
| Retrieval | Did we find the right document? |
| Coverage | Did we find all relevant pages/passages? |
| Temporal | Did we identify the correct version/applicability? |
| Selection | Did we choose the right evidence? |
| Generation | Did the LLM interpret the evidence correctly? |
| Validation | Did the safety layer accept/reject correctly? |
| Citation | Does the citation actually support the claim? |

## Corpus anchors

`fixtures/corpus_pages.json` extracted from `documents/`:

- `Cir_2025_13_fr.pdf` — export delays 120 / 121–360 / >360
- `Cir_2026_04_fr.pdf` — non-priority deposit + Art.4 exclusions; empty OCR annex p.4–7
- `Cir_2020_02_fr.pdf` — earlier regime (60-day + crédit documentaire distractor)

## Item map (user taxonomy → suite id)

### 1. Retrieval

| Item | Suite | Stage |
| --- | --- | --- |
| 1.A Same vocab / wrong domain | `1.A` | Retrieval |
| 1.B Explicit name ignored | `1.B` | Retrieval |
| 1.C Correct doc, wrong page | `1.C` | Coverage |
| 1.D Correct page, wrong chunk | `1.D` | Coverage |
| 1.E General without exception | `1.E` | Coverage |
| 1.F Exception without general | `1.F` | Coverage |
| 1.G Definition missed | `1.G` | Coverage |
| 1.H Cross-document contamination | `1.H` | Generation |
| 1.I Historical outranks current | `1.I` | Temporal |
| 1.J Newer mention outranks named | `1.J` | Retrieval |
| 1.K Amendment chain not followed | `1.K` | Temporal |
| 1.L Wrong amendment chain (Vu) | `1.L` | Temporal |
| 1.M Numeric near-miss | `1.M` | Validation |
| 1.N Language mismatch | `1.N` | Retrieval |
| 1.O OCR corruption / empty annex | `1.O` | Coverage |
| 1.P Metadata mismatch | `1.P` | Citation |
| 1.Q Duplicate / version confuse | `1.Q` | Retrieval |
| 1.R Query ambiguity | `1.R` | Selection |
| 1.S Under-specified | `1.S` | Validation |
| 1.T Overly broad | `1.T` | Selection |
| 1.U Overly narrow paraphrase | `1.U` | Retrieval |
| 1.V Typo / terminology variant | `1.V` | Validation |
| 1.W Contradictory evidence | `1.W` | Generation |

### 2. Context / evidence assembly

| Item | Suite | Stage |
| --- | --- | --- |
| 2.A Page not assembled | `2.A` | Coverage |
| 2.B Wrong order | `2.B` | Coverage |
| 2.C Truncation | `2.C` | Coverage |
| 2.D Operator dropped | `2.D` | Validation |
| 2.E Exception detached | `2.E` | Generation |
| 2.F Article across pages | `2.F` | Coverage |
| 2.G List/table destroyed | `2.G` | Coverage |
| 2.H Definition detached | `2.H` | Coverage |
| 2.I Relations not marked | `2.I` | Selection |

### 3. Temporal / supersession

| Item | Suite | Stage |
| --- | --- | --- |
| 3.A Citation ≠ amendment | `3.A` | Temporal |
| 3.B Wrong direction | `3.B` | Temporal |
| 3.C Amend/replace/abrogate | `3.C` | Temporal |
| 3.D Vu as operative | `3.D` | Temporal |
| 3.E Later mentions earlier | `3.E` | Retrieval |
| 3.F Earlier exception survives | `3.F` | Validation |
| 3.G Replacement destroys unrelated | `3.G` | Temporal |
| 3.H Effective date | `3.H` | Validation |
| 3.I Transition ignored | `3.I` | Validation |
| 3.J Before entry into force | `3.J` | Generation |
| 3.K Current date substituted | `3.K` | Validation |
| 3.L Future as current | `3.L` | Validation |
| 3.M Partial as complete | `3.M` | Temporal |
| 3.N One article → whole circular | `3.N` | Temporal |

### 4–5. Selection & generation

| Item | Suite | Stage |
| --- | --- | --- |
| 4.A–4.J | `4.A`…`4.J` | Selection / Retrieval / Coverage / Validation / Generation / Temporal |
| 5.A–5.T | `5.A`…`5.T` | Generation / Validation / Selection / Citation / Temporal |

### 6. Validator

| Item | Suite | Stage |
| --- | --- | --- |
| False rejection | `6.false_reject_ok` | Validation |
| False acceptance | `6.false_accept_block` | Validation |
| Quote too short | `6.quote_too_short` | Validation |
| Quote semantically insufficient | `6.quote_semantically_insufficient` | Validation |
| Question facts as claims | `6.question_facts` | Validation |
| Overfit wording | `6.overfit_wording` | Validation |
| Underfit wording | `6.underfit_wording` | Validation |
| Polarity FP | `6.polarity_fp` | Generation |
| Polarity FN | `6.polarity_fn` | Generation |

### 7. Citation

| Item | Suite | Stage |
| --- | --- | --- |
| Wrong PDF | `7.wrong_pdf` | Citation |
| Wrong page | `7.wrong_page` | Citation |
| Wrong article | `7.wrong_article` | Validation |
| Historical vs operative | `7.historical_vs_operative` | Citation |
| Half support | `7.half_support` | Validation |
| Multipart one cite | `7.multipart_one_cite` | Validation |
| Multi-source one supports | `7.multi_source_one_supports` | Generation |

### 8. Current rule

| Item | Suite | Stage |
| --- | --- | --- |
| Old more similar | `8.old_vs_new` | Temporal |
| Partial amendment | `8.partial_amendment` | Temporal |
| Mentions without replace | `8.mention_not_replace` | Temporal |
| Exception in new rule | `8.exception_current` | Coverage |
| Grandfathering | `8.grandfather` | Generation |
| Effective date | `8.effective_date` | Validation |
| Future effective | `8.future_effective` | Validation |

### 9–10. Multi-hop & adversarial

| Item | Suite |
| --- | --- |
| Multi-hop pack / chain | `9.multi_hop_pack`, `9.amendment_chain` |
| Adversarial battery | `10.same_words_wrong_doc` … `10.insufficient` |

Known product gaps: `xfail` in `test_stress_suite.py` (`_KNOWN_GAPS`) with `Failure stage = …`.

## Run

```text
cd backend
python -m pytest tests/stress/ -q
python -m tests.stress.report
```

Live dense/BM25 against Voyage/local indexes needs a non-empty `native.jsonl`; retrieval-stage cases exercise hard-anchor / pin / expand guards on corpus-backed packs until the index is rebuilt.
