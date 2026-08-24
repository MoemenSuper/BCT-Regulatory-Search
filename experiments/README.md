# Disposable structured-ingestion experiment

This branch and worktree are intentionally separate from the production ingestion path and production `chroma_db`.

The experiment keeps these retrieval controls fixed:

- original `evaluation_queries.json` questions;
- `intfloat/multilingual-e5-small` embeddings;
- dense top 20;
- lowercase-whitespace BM25 top 15;
- dense-plus-BM25 candidate union with the production deduplication key;
- `BAAI/bge-reranker-v2-m3` reranker;
- final top 5.

## Stage-aware analysis

Generate overall, French, Arabic, candidate-pool, dense-channel, BM25-channel,
ranking, and failure metrics from frozen artifacts without rerunning models:

```powershell
python -m experiments.retrieval_analysis `
  --evaluation evaluation_queries.json `
  --result "$env:TEMP\experiment\results\configuration.json" `
  --candidate-cache "$env:TEMP\experiment\candidate_caches\configuration.json" `
  --output "$env:TEMP\experiment\stage_analysis.json"
```

## Freeze evaluation sets

The original 697 cases are development data because their failures have already
been inspected. Once newly curated validation and holdout files exist, freeze
their hashes and audit leakage with:

```powershell
python -m experiments.evaluation_protocol `
  --development evaluation_queries.json `
  --validation evaluation_validation.json `
  --holdout evaluation_final_holdout.json `
  --corpus-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --output evaluation_protocol.json
```

Final holdout is required to be disjoint from both development and validation
by conservative type-number family across years and languages. Unknown validation/holdout filename forms
require an explicit `document_family` instead of silently becoming unique.
Validation/development overlap is audited. A legacy development/holdout override
exists for explicitly documenting a weaker protocol, but is disabled by default.
When no new family-disjoint documents exist yet, replace `--holdout` with a
non-empty `--prospective-holdout-reason`. The protocol then records zero
holdout cases, and the evaluation boundary refuses final-holdout execution.
Pass `--validation-review-receipt` to bind the frozen validation file to its
source-review receipt and status.

Audit the page-disjoint validation candidate before any retrieval access:

```powershell
python -m experiments.validation_candidate_audit `
  --development evaluation_queries.json `
  --validation evaluation_validation_candidate_v1.json `
  --structured-manifest "$env:TEMP\structured-ingestion\ingestion_manifest.json" `
  --output experiments/results/validation_candidate_v1_audit.json

python -m experiments.validation_review_packet `
  --validation evaluation_validation_candidate_v1.json `
  --audit experiments/results/validation_candidate_v1_audit.json `
  --output experiments/reviews/validation_candidate_v1_human_review.md
```

The tracked v1 candidate remains pending independent human verification. A
blind second agent reviewed all source pages without model outputs, caught one
unsupported label, and approved all 24 cases after correction. That hash-bound
review and receipt permit provisional validation only; they are not human
adjudication. The audit's `retrieval_access: not_run` records the pre-access
freeze state. The later one-time retrieval access is recorded separately in
`experiments/registry/evaluation_access.jsonl`. No current-corpus final holdout
is claimed: all 439 sources are legacy development, so the family-disjoint
holdout remains prospective.
The audit also fails on normalized query/evidence duplicates and emits lexical
near-duplicate review flags; lexical non-matches are not treated as proof of
semantic independence.

After reviewing the Markdown packet, fill a copy of
`experiments/reviews/validation_candidate_v1_human_decisions.json`. Approval is
fail-closed and produces the validation dataset only when every case is
approved and every independent-review attestation is true:

```powershell
python -m experiments.validation_human_approval approve `
  --candidate evaluation_validation_candidate_v1.json `
  --audit experiments/results/validation_candidate_v1_audit.json `
  --review "$env:TEMP\validation_candidate_v1_human_decisions.json" `
  --output evaluation_validation_v1.json `
  --receipt "$env:TEMP\validation_candidate_v1_approval_receipt.json"
```

Any `correct`, `reject`, or `pending` decision refuses approval and requires a
new candidate/audit version. Keep reviewer identity outside the repository
unless the reviewer explicitly authorizes recording it.

The partially completed Markdown packet is preserved as reviewer work against
the superseded pre-correction hash. Use the refreshed machine-readable decision
template for any future independent human approval of the current candidate.

Verify the completed second-agent review and its provisional-use boundary with:

```powershell
python -m experiments.validation_agent_review `
  --candidate evaluation_validation_candidate_v1.json `
  --audit experiments/results/validation_candidate_v1_audit.json `
  --review experiments/reviews/validation_candidate_v1_agent_double_review.json `
  --receipt "$env:TEMP\validation_candidate_v1_agent_review_receipt.json"
```

Evaluate a frozen split through the access-logged boundary:

```powershell
python -m experiments.frozen_evaluation `
  --protocol evaluation_protocol.json `
  --role validation `
  --result "$env:TEMP\validation-result.json" `
  --output "$env:TEMP\validation-aggregate.json" `
  --ledger experiments/registry/evaluation_access.jsonl `
  --purpose "Architecture selection checkpoint" `
  --accessed-by "reviewer-name" `
  --code-commit "$(git rev-parse HEAD)"
```

Validation case details require the explicit `--include-case-details` switch.
The final-holdout role rejects that switch and emits overall/French/Arabic
aggregates only. Both roles verify the frozen dataset hash and exact result-ID
coverage before recording a successful access.

## Record experiments

Append one schema-checked entry at a time. Entries require dataset hashes,
overall development metrics, validation status/metrics, French and Arabic
metrics, latency/cost, failure distribution, and a KEEP/REJECT decision:

```powershell
python -m experiments.experiment_registry `
  --registry experiments/registry/experiments.jsonl `
  --entry "$env:TEMP\experiment-entry.json"
```

## Freeze the extraction-quality development stress set

```powershell
python -m experiments.stress_suite `
  --evaluation evaluation_queries.json `
  --result "$env:TEMP\experiment\results\structured_baseline_chunking.json" `
  --structured-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --output "$env:TEMP\experiment\extraction_stress_development_v1.json"
```

The frozen set contains all current development extraction failures plus unique,
deterministic same-language and same-category Top-5 controls. It is development
data and must not be reported as validation or holdout evidence.

Screen a proposed Arabic fallback gate without changing ingestion:

```powershell
python -m experiments.arabic_quality_experiment `
  --stress-suite experiments/stress_suites/extraction_development_v1.json `
  --output "$env:TEMP\experiment\arabic_quality_gate_v1.json"
```

Compare the current native text and auto-OCR with an explicit Arabic RapidOCR
recognizer on only the frozen gate-triggered development pages:

```powershell
python -u -m experiments.ocr_fallback_experiment `
  --stress-suite experiments/stress_suites/extraction_development_v1.json `
  --evaluation evaluation_queries.json `
  --structured-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --output-dir "$env:TEMP\bct-arabic-ocr-fallback"
```

The output is resumable from per-page caches. Its KEEP/REJECT screen measures
verified evidence-token and critical-number recall; it does not deploy or alter
the ingestion fallback.

Build and run the corpus-wide additive OCR retrieval ablation from the frozen
native candidate cache:

```powershell
python -u -m experiments.ocr_fusion_retrieval all `
  --evaluation evaluation_queries.json `
  --current-result "$env:TEMP\reproduction\results\structured_baseline_chunking.json" `
  --current-candidates "$env:TEMP\reproduction\candidate_caches\structured_baseline_chunking.json" `
  --structured-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --structured-cache-dir "$env:TEMP\structured" `
  --ocr-cache-dir "$env:TEMP\arabic-ocr-cache" `
  --output-dir "$env:TEMP\arabic-ocr-fusion" `
  --summary-output "$env:TEMP\arabic-ocr-fusion-summary.json"
```

This applies the unchanged gate to every Arabic page, builds a disposable OCR
index, routes by Arabic characters in the runtime query text, and preserves the
native candidate union. The tracked slim result retains all 697 before/after
ranks while omitting duplicated chunk text.

Freeze answer-bearing numeric and Latin alphanumeric identifier fidelity for
the gate-triggered Arabic pages:

```powershell
python -m experiments.numeric_fidelity_stress `
  --evaluation evaluation_queries.json `
  --structured-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --ocr-representation-manifest "$env:TEMP\arabic-ocr-fusion\representation\manifest.json" `
  --ocr-cache-dir "$env:TEMP\arabic-ocr-cache" `
  --output "$env:TEMP\numeric-identifier-development-v1.json"
```

The suite scores only verified literals from development evidence snippets.
It does not treat unrelated page numbers as hallucinations and does not permit
OCR text to override conflicting native digits.

Freeze the broader targeted development catalog from the reproduced baseline
and canonical StructuredDocument artifacts:

```powershell
python -m experiments.targeted_stress_catalog `
  --evaluation evaluation_queries.json `
  --result "$env:TEMP\reproduction\results\structured_baseline_chunking.json" `
  --structured-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --output experiments/stress_suites/targeted_development_catalog_v1.json
```

The catalog fixes table-page, visual non-table, wrong-version, long-document,
primary context-failure, negative/ambiguous, and Latin alphanumeric identifier
cohorts. Wrong-version and context failures receive unique deterministic
same-language/category Top-5 controls. Membership does not consult OCR-fusion
or VLM outputs, and every cohort remains development-only.

Measure the reproduced winner and additive OCR ranks on each frozen cohort:

```powershell
python -m experiments.stress_catalog_benchmark `
  --catalog experiments/stress_suites/targeted_development_catalog_v1.json `
  --retrieval-result experiments/results/arabic_ocr_fusion_retrieval_v1.json `
  --output experiments/results/targeted_stress_retrieval_v1.json
```

This is a cached rank analysis: it performs no retrieval, reranking, OCR, or
hosted-model calls. Negative/ambiguous cases are deliberately marked not
evaluated because retrieval ranks cannot measure abstention.

Rerun the identical additive-OCR candidates and scores, then apply stable
source-page diversity after reranking:

```powershell
python -u -m experiments.candidate_diversity_ablation `
  --evaluation evaluation_queries.json `
  --current-result "$env:TEMP\reproduction\results\structured_baseline_chunking.json" `
  --current-candidates "$env:TEMP\reproduction\candidate_caches\structured_baseline_chunking.json" `
  --fusion-slim-result experiments/results/arabic_ocr_fusion_retrieval_v1.json `
  --representation-manifest "$env:TEMP\arabic-ocr-fusion\representation\manifest.json" `
  --output "$env:TEMP\candidate-page-diversity-full.json" `
  --slim-output experiments/results/candidate_page_diversity_v1.json
```

The ablation fails if its undiversified ranks do not exactly match the tracked
fusion result. Diversity keeps the highest-scored chunk for each source/start
page; context assembly may still reattach other chunks from a selected page.

Freeze the bounded gold-evidence answer-safety development suite:

```powershell
python -m experiments.answer_safety_suite `
  --evaluation evaluation_queries.json `
  --current-result "$env:TEMP\reproduction\results\structured_baseline_chunking.json" `
  --targeted-catalog experiments/stress_suites/targeted_development_catalog_v1.json `
  --numeric-suite experiments/stress_suites/numeric_identifier_development_v1.json `
  --output experiments/stress_suites/answer_safety_development_v1.json
```

The 40 disjoint cases contain 32 relevant questions across eight risk/control
roles and all eight negative/ambiguous questions. Relevant generation receives
only the verified evidence snippet and exact source/page; negatives receive no
regulatory evidence. This isolates generation safety from retrieval quality.

Run the bounded hosted-VLM numeric inventory on the frozen public Arabic pages:

```powershell
python -u -m experiments.vlm_numeric_experiment `
  --numeric-suite experiments/stress_suites/numeric_identifier_development_v1.json `
  --structured-manifest "$env:TEMP\structured\ingestion_manifest.json" `
  --dotenv "C:\path\to\the\existing\.env" `
  --output-dir "$env:TEMP\vlm-numeric-v1" `
  --confirm-public-documents `
  --summary-output experiments/results/vlm_numeric_fidelity_v1.json
```

Per-page outputs are cached by PDF hash, page, model, and prompt version. The
tracked result contains literal-level metrics but no rendered images, base64,
API key, or raw provider errors. A KEEP decision permits only a separate
full-text VLM retrieval ablation; it does not make VLM output authoritative.

Run the current generator on the frozen gold-evidence suite, then apply the
case-level evidence review:

```powershell
python -u -m experiments.gold_evidence_answer_experiment `
  --suite experiments/stress_suites/answer_safety_development_v1.json `
  --dotenv "C:\path\to\the\existing\.env" `
  --output-dir "$env:TEMP\gold-evidence-answer-v1" `
  --confirm-public-documents

python -m experiments.answer_evidence_review `
  --suite experiments/stress_suites/answer_safety_development_v1.json `
  --generated-result "$env:TEMP\gold-evidence-answer-v1\result.json" `
  --review experiments/reviews/gold_evidence_answer_review_v1.json `
  --output experiments/results/gold_evidence_answer_baseline_v1.json
```

Automatic literal checks are diagnostic only. The review artifact expands
case-level answer, citation, grounding, refusal, and clarification labels. Its
labels are an agent evidence review and explicitly do not claim independent
human adjudication.

Test the unchanged answer model with a strict claim-linked response contract:

```powershell
python -u -m experiments.structured_answer_experiment `
  --suite experiments/stress_suites/answer_safety_development_v1.json `
  --dotenv "C:\path\to\the\existing\.env" `
  --output-dir "$env:TEMP\structured-answer-v1" `
  --confirm-public-documents

python -m experiments.answer_evidence_review `
  --suite experiments/stress_suites/answer_safety_development_v1.json `
  --generated-result "$env:TEMP\structured-answer-v1\result.json" `
  --review experiments/reviews/structured_answer_review_v1.json `
  --output experiments/results/structured_answer_prompt_v1.json
```

The contract exposes status, atomic claims, evidence IDs, and structured
citations. Valid links are declared provenance, not proof of entailment; the
same evidence review remains mandatory.

Test the two claim-linked safe abstentions with exact-page context only:

```powershell
python -m experiments.answer_context_expansion `
  --base-suite experiments/stress_suites/answer_safety_development_v1.json `
  --manifest "$env:TEMP\structured-ingestion\ingestion_manifest.json" `
  --case-id cir_2019_02_fr_amount_or_rate_02 `
  --case-id note_2022_16_ar_ceramic_importer_02 `
  --output experiments/stress_suites/answer_context_expansion_development_v1.json

python -u -m experiments.structured_answer_experiment `
  --suite experiments/stress_suites/answer_context_expansion_development_v1.json `
  --dotenv "C:\path\to\the\existing\.env" `
  --output-dir "$env:TEMP\answer-context-expansion-v1" `
  --confirm-public-documents

python -m experiments.answer_evidence_review `
  --suite experiments/stress_suites/answer_context_expansion_development_v1.json `
  --generated-result "$env:TEMP\answer-context-expansion-v1\result.json" `
  --review experiments/reviews/answer_context_expansion_review_v1.json `
  --output experiments/results/answer_context_expansion_v1.json
```

This is a post-hoc two-case context-sufficiency diagnostic. It bypasses
retrieval and cannot establish that full-page context generalizes.

Run the bounded v2 status-policy gate before any full-suite answer call:

```powershell
python -m experiments.answer_status_suite `
  --base-suite experiments/stress_suites/answer_safety_development_v1.json `
  --output experiments/stress_suites/answer_status_development_v2.json

python -u -m experiments.structured_answer_experiment `
  --suite experiments/stress_suites/answer_status_development_v2.json `
  --dotenv "C:\path\to\the\existing\.env" `
  --output-dir "$env:TEMP\answer-status-v2" `
  --confirm-public-documents

python -m experiments.answer_evidence_review `
  --suite experiments/stress_suites/answer_status_development_v2.json `
  --generated-result "$env:TEMP\answer-status-v2\result.json" `
  --review experiments/reviews/answer_status_policy_review_v2.json `
  --output experiments/results/answer_status_policy_v2.json
```

The gate requires all eight expected statuses and non-empty user-facing text.
V2 reached seven of eight, so the full 40-case suite was deliberately not run.

Run the complete experiment with an empty output directory outside the repository:

```powershell
python -u -m experiments.structured_ingestion_benchmark all `
  --documents-dir "C:\Users\Moemen Super\BCT-Regulatory-Search\documents" `
  --output-dir "$env:TEMP\bct-structured-ingestion-experiment-20260823" `
  --evaluation evaluation_queries.json `
  --baseline "$env:TEMP\bct_hybrid_eval_20260821.json"
```

Extraction checkpoints after every PDF, so `ingest` or `all` can resume. Index creation refuses to overwrite an existing experimental index. The final `benchmark_results.json` stores every question's baseline and experimental ranks, retrieved top five with text and structure metadata, failure classifications, and OCR-rescue status. `benchmark_report.md` contains aggregate metrics and the changed/failed question summaries.
