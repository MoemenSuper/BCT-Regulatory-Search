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

Run the complete experiment with an empty output directory outside the repository:

```powershell
python -u -m experiments.structured_ingestion_benchmark all `
  --documents-dir "C:\Users\Moemen Super\BCT-Regulatory-Search\documents" `
  --output-dir "$env:TEMP\bct-structured-ingestion-experiment-20260823" `
  --evaluation evaluation_queries.json `
  --baseline "$env:TEMP\bct_hybrid_eval_20260821.json"
```

Extraction checkpoints after every PDF, so `ingest` or `all` can resume. Index creation refuses to overwrite an existing experimental index. The final `benchmark_results.json` stores every question's baseline and experimental ranks, retrieved top five with text and structure metadata, failure classifications, and OCR-rescue status. `benchmark_report.md` contains aggregate metrics and the changed/failed question summaries.
