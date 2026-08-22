# Contextual Retrieval assessment for BCT Regulatory Search

Date: 2026-08-21

## Decision

Do **not** replace the newly measured hybrid search or rebuild the production index yet. Keep dense + BM25 + BGE reranking as the baseline and run Contextual Retrieval as a parallel, reversible experiment. The technique is a credible candidate for the next retrieval improvement, but Anthropic's headline result is not evidence that BCT will gain the same amount.

The safest first experiment is a stratified subset of French and Arabic PDFs, including long documents, tables, OCR-affected pages, and the four hybrid-regression queries. If that succeeds, build a separate contextual Chroma collection and contextual BM25 index for all 5,214 chunks, then run the same 697-case evaluation unchanged.

## What Anthropic actually did

For every original chunk, Anthropic gave Claude the **whole source document** and that chunk, and asked for a short explanation that situates the chunk in the document. The generated context was normally 50–100 tokens. It was prepended to the original chunk both when creating the embedding and when building the BM25 index. This is an ingestion-time transformation, not an LLM call on every user query. Anthropic used Claude 3 Haiku in the published experiment and provides the exact generic prompt in its [official engineering article](https://www.anthropic.com/engineering/contextual-retrieval).

The evidence text must remain distinguishable from generated context. Anthropic explicitly advises testing whether answer generation improves when it receives the contextualized chunk and can distinguish the context from the chunk ([implementation considerations](https://www.anthropic.com/engineering/contextual-retrieval)). For BCT, citations and quoted evidence must always come from the immutable original page text, never from the LLM-generated prefix.

## What the reported improvement means

Anthropic measured **1 minus recall@20**: the fraction of relevant or "golden" chunks not found among the first 20 retrieved chunks. Averaged across codebases, fiction, arXiv papers, and science papers, using its best reported embedding configuration (Gemini Text 004):

- contextual embeddings reduced failure from 5.7% to 3.7%, a **35% relative reduction**;
- contextual embeddings plus contextual BM25 reduced it from 5.7% to 2.9%, a **49% relative reduction**;
- contextual embeddings plus contextual BM25 plus Cohere reranking reduced it from 5.7% to 1.9%, a **67% relative reduction**.

These are relative reductions in missed retrievals, not increases of 35, 49, or 67 percentage points. The 67% result corresponds to recall@20 increasing from 94.3% to 98.1%. It does not measure final-answer correctness, citation accuracy, abstention, or regulatory validity. Anthropic retrieved 150 initial candidates and reranked down to 20 for that result; BCT currently unions only 20 dense and 15 BM25 candidates, then returns five. Sources: [methodology and results](https://www.anthropic.com/engineering/contextual-retrieval), [official appendix with full breakdown and examples](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf).

Anthropic's more focused official cookbook used 737 chunks from nine codebases and 248 queries. In that example, contextual embeddings raised pass@10 from about 87% to about 95%. That is useful corroboration, but it is still code retrieval rather than bilingual regulatory retrieval ([official cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)).

## Cost and caching assumptions

Anthropic's published estimate was a one-time **$1.02 per million source-document tokens**, assuming 800-token chunks, 8,000-token documents, 50 instruction tokens, and 100 generated context tokens. That figure depends on the then-current Claude pricing and prompt caching. The full document is written to a five-minute cache for the first chunk, then read from cache for subsequent chunks from the same document; therefore chunks must be processed document-by-document and quickly enough to keep cache hits. In the cookbook's concrete 737-chunk run, 61.83% of input tokens came from cache and the reported input cost fell from about $9.20 without caching to $2.85 with caching. Sources: [Anthropic cost assumptions](https://www.anthropic.com/engineering/contextual-retrieval), [cookbook cost and cache flow](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide), [current prompt-caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

The $1.02 figure should be treated as historical methodology, not a quote for BCT. Actual cost depends on each PDF's extracted token count, model choice, cache eligibility and hit rate, output length, retries, and embedding rebuild cost. Query-time contextualization adds no LLM latency, although larger stored text can affect BM25, embedding truncation, reranking, storage, and downstream prompt size.

## BCT-specific risks and controls

1. **Generated legal claims can be wrong.** A contextualizer may invent a circular number, scope, date, entity, definition, or relationship. Such text can improve matching while poisoning retrieval. Store `original_page_text` and `generated_retrieval_context` separately, validate required identifiers against metadata, and never display or quote generated context as evidence.
2. **French/Arabic behavior is unproven here.** Anthropic reports other domains and embedding providers, not `multilingual-e5-small`, BGE reranking, Arabic morphology, OCR-corrupted Arabic, or mixed-direction regulatory text. Evaluate languages and extraction conditions separately.
3. **"Whole document" may not be feasible or trustworthy.** Long PDFs can exceed the chosen model's context window, and noisy extraction or tables can distort generated context. Define a documented fallback such as structured document metadata plus nearby pages; do not silently truncate.
4. **Embedding truncation can erase the gain.** Anthropic warns that contextualized chunks may perform worse if the embedding model truncates them. Measure token length after adding context and keep the original chunk intact.
5. **The benchmark has less top-20 headroom than the headline suggests.** BCT hybrid exact-page top-20 is already 92.45% and source top-20 is 96.95%. The meaningful production target is exact-page top-five and end-to-end answer safety, not merely reproducing Anthropic's recall@20 setup.
6. **Context changes two indexes at once.** To attribute gains, compare at least: current hybrid baseline; contextual embeddings with original BM25; and contextual embeddings with contextual BM25. Keep candidate counts, reranker, queries, and evaluation labels fixed.

## Go/no-go evaluation

Promote contextual retrieval only if the parallel experiment shows a statistically and operationally meaningful net gain on the unchanged 697 cases, improves or preserves French and Arabic exact-page top-five separately, repairs more hybrid misses than it creates, and does not worsen negative-query answer behavior. Manually inspect every changed success/regression for generated-context errors. Also measure ingestion cost, cache hit rate, indexing time, query latency, and contextualized-token truncation.

Until those conditions pass, Contextual Retrieval is a **promising proposal**, not a demonstrated BCT improvement.

## Full-corpus experimental result

The parallel experiment was completed on 2026-08-21 without changing the production collection. It generated retrieval-only context for all 5,214 existing chunks and evaluated the unchanged 697-query benchmark (689 relevant queries and 8 negative queries). The production `bct_regulations` collection remained at 5,214 chunks.

### Model screening and safety design

Five available French/Arabic-capable models were screened on a fixed 24-case sample. Groq Qwen 3.6 27B had the strongest embedding proxy but invented legal dates and denominations. GPT-OSS 20B was the strongest safer Groq candidate, but it still invented unsupported institutional framing. The available local Qwen3 4B model also hallucinated when asked for free-form contextual prose. A Qwen3.5 download could not be completed after network access was restricted, so it was not counted as tested.

The full run therefore used local Qwen3 4B with a constrained retrieval-label format instead of free-form legal context. Each label could contain only the exact source identifier, exact page, and a short French or Arabic topic noun phrase. Generated labels were kept out of citations, evidence, and BGE reranking. Long documents that could not fit the 8 GB GPU used an explicit page-neighborhood fallback rather than silent truncation.

All 5,214 outputs were received, with zero language or template failures. The guard accepted 3,934 labels (75.45%) and fell back to original chunk text for 1,280. Rejections comprised 862 incomplete labels, 245 labels with unsupported numbers, and 173 with both. This rejection rate is itself evidence that unguarded contextual text would be unsafe for this corpus.

### Retrieval results

The current hybrid baseline and the two guarded ablations were:

| Method | Source top 5 | Source top 20 | Exact page top 5 | Exact page top 20 |
| --- | ---: | ---: | ---: | ---: |
| Current dense + BM25 + BGE | 92.02% | 96.95% | 86.07% | 92.45% |
| Contextual dense + original BM25 + original-text BGE | 92.02% | 97.82% | 86.07% | 93.76% |
| Contextual dense + contextual BM25 + original-text BGE | 91.73% | 97.97% | 86.21% | 94.05% |

The contextual-BM25 variant produced 9 exact-page top-five repairs and 8 regressions: a net gain of one query out of 689, with no paired statistical evidence of a top-five improvement (`p = 1.0`). Its exact-page top-20 result improved by 11 net queries (15 repairs and 4 regressions), a 1.60 percentage-point gain with a paired exact test of `p = 0.019`. The original-BM25 ablation had 8 top-five repairs and 8 regressions.

The top-20 gain is real but small and does not satisfy the production target. Manual review found useful repairs, but also regressions on exact codes, page-specific requirements, and table-like amounts where a broad or wrong topic label displaced the correct page. The eight negative queries also experienced candidate churn; retrieval metrics alone cannot verify that final-answer abstention remains safe.

### Decision after measurement

Do **not** implement this contextual-retrieval variant before other safety improvements, and do not rebuild the production index with these labels. It adds generation complexity and a new failure surface while leaving exact-page top-five effectively unchanged. Preserve the current hybrid search as the production baseline.

If Contextual Retrieval is revisited, the next experiment should target the observed failure modes rather than repeat the same run: derive document/page descriptors from verified metadata or extractive phrases, improve table and code handling, require an end-to-end abstention benchmark, and set a predeclared minimum top-five gain before promotion.
