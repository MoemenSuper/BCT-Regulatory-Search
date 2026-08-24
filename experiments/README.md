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

Run the complete experiment with an empty output directory outside the repository:

```powershell
python -u -m experiments.structured_ingestion_benchmark all `
  --documents-dir "C:\Users\Moemen Super\BCT-Regulatory-Search\documents" `
  --output-dir "$env:TEMP\bct-structured-ingestion-experiment-20260823" `
  --evaluation evaluation_queries.json `
  --baseline "$env:TEMP\bct_hybrid_eval_20260821.json"
```

Extraction checkpoints after every PDF, so `ingest` or `all` can resume. Index creation refuses to overwrite an existing experimental index. The final `benchmark_results.json` stores every question's baseline and experimental ranks, retrieved top five with text and structure metadata, failure classifications, and OCR-rescue status. `benchmark_report.md` contains aggregate metrics and the changed/failed question summaries.
