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

Run the complete experiment with an empty output directory outside the repository:

```powershell
python -u -m experiments.structured_ingestion_benchmark all `
  --documents-dir "C:\Users\Moemen Super\BCT-Regulatory-Search\documents" `
  --output-dir "$env:TEMP\bct-structured-ingestion-experiment-20260823" `
  --evaluation evaluation_queries.json `
  --baseline "$env:TEMP\bct_hybrid_eval_20260821.json"
```

Extraction checkpoints after every PDF, so `ingest` or `all` can resume. Index creation refuses to overwrite an existing experimental index. The final `benchmark_results.json` stores every question's baseline and experimental ranks, retrieved top five with text and structure metadata, failure classifications, and OCR-rescue status. `benchmark_report.md` contains aggregate metrics and the changed/failed question summaries.
