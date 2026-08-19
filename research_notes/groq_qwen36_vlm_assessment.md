# Groq Qwen 3.6 27B for BCT PDF extraction

Date: 2026-08-18

## Decision

`qwen/qwen3.6-27b` on Groq is a strong candidate for a **small, measured page-image benchmark**. It is not a safe one-step replacement for Docling, and it is not currently a suitable production dependency by itself.

The recommended role is a remote visual fallback for pages or table regions that local Docling/native extraction and OCR mark as suspicious. Keep the local extraction, compare the model's reading against it, and preserve disagreements for review or abstention.

## What the proposed model actually is

The exact Groq model ID is `qwen/qwen3.6-27b`. Groq describes it as a dense 27-billion-parameter multimodal model with text and image input, text output, OCR/document/chart understanding, multilingual support, reasoning, tools, and JSON Object Mode. Groq lists approximately 500 output tokens per second, a 131,072-token context, 16,384 maximum output tokens, and prices of $0.60 per million input tokens and $3.00 per million output tokens. Source: [Groq Qwen 3.6 27B model page](https://console.groq.com/docs/model/qwen/qwen3.6-27b).

It is currently labelled **Preview**. Groq states that preview models are intended for evaluation rather than production and may be discontinued at short notice. This makes it appropriate for the proposed experiment, but production code must use a provider/model abstraction and must not assume this model ID will remain available. Sources: [Groq supported-model catalog](https://console.groq.com/docs/models), [Groq model-deprecation policy](https://console.groq.com/docs/deprecations).

## Important API constraints

- Groq documents image-URL and base64-image inputs; it does not document native PDF input for this model. Therefore, the application must render a PDF locally into page images or table crops before sending them. This is an inference from the documented input types, not a Groq claim about every possible undocumented request. Source: [Groq vision guide](https://console.groq.com/docs/vision).
- The model-specific page lists a 20 MB maximum file size and three input images. The general vision guide currently says five images. Because the official pages conflict, a conservative client should assume three and verify the live endpoint. Sources: [model page](https://console.groq.com/docs/model/qwen/qwen3.6-27b), [vision guide](https://console.groq.com/docs/vision).
- Qwen 3.6 supports JSON Object Mode, but Groq's strict and best-effort JSON Schema model lists currently contain only GPT-OSS models. JSON syntax does not prove that extracted text or table cells are correct; the response must be schema-validated locally and retried or rejected when incomplete. Sources: [vision JSON example](https://console.groq.com/docs/vision), [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs).
- Groq advertises asynchronous Batch processing with separate limits and a 50% discount, but the current Batch model list does **not** include `qwen/qwen3.6-27b`. The proposed Qwen pipeline would therefore need controlled synchronous concurrency unless Groq later adds it. Batch files and results are retained for up to 30 days. Source: [Groq Batch API](https://console.groq.com/docs/batch).
- Published limits vary by account and tier. The current free-tier table lists 30 RPM, 1,000 RPD, 8,000 TPM, and 200,000 TPD for Qwen 3.6, while the model catalog advertises higher Developer-plan limits. The organization's live Limits page and response headers are authoritative. Sources: [Groq rate limits](https://console.groq.com/docs/rate-limits), [supported-model catalog](https://console.groq.com/docs/models).

The published token price looks inexpensive, but Groq does not provide a page-based price. The honest estimate must come from a representative pilot's returned input/output token usage. For example, every 1,000 output tokens costs $0.003 before image-input tokens; a full transcription can therefore cost materially more than a short table-cell response.

## Security and confidentiality

For ordinary inference, Groq says customer inputs and outputs are not retained by default. It may temporarily log them for reliability or abuse investigation for up to 30 days. Zero Data Retention is available to all customers; retained customer data is stored in US GCP buckets. Groq also says inputs and outputs are not used for model training unless the customer explicitly permits it. Sources: [Your Data in GroqCloud](https://console.groq.com/docs/your-data), [Groq Services Agreement](https://console.groq.com/docs/legal/services-agreement).

Groq documents TLS 1.2 or later, encryption at rest, least-privilege access controls, penetration testing, and independent SOC 2 Type II audits. Those controls are useful, but they do not themselves authorize BCT documents to leave BCT infrastructure. Sources: [Groq security onboarding](https://console.groq.com/docs/production-readiness/security-onboarding), [Groq Data Processing Addendum](https://console.groq.com/docs/legal/customer-data-processing-addendum).

The BCT cahier permits only public, synthetic, or explicitly authorized documents and makes local/API confidentiality compatibility an evaluation criterion (`Cahier_des_charges_BCT_VERSION1.pdf`, page 5). Sending the current public corpus can be considered for the prototype after normal institutional approval. Future internal or merely "authorized" documents need an explicit BCT decision covering external processing, US data location, contractual terms, and ZDR; authorization to use a document is not automatically authorization to transmit it to Groq.

## Why this will not be a quick answer to everything

A larger vision model may outperform the small local Granite models on Arabic reading order, exact digits, and table association. The official capabilities make that plausible, not proven. A generative VLM can still omit rows, normalize digits incorrectly, invent structure, or return confident but wrong text. Model parameter count and speed do not remove the need for source-page verification.

It also does not replace the non-visual parts of Docling:

1. Docling still opens the PDF, identifies pages and layout regions, preserves native text and metadata, and provides the local baseline.
2. Only suspicious pages or regions are rendered and sent to Qwen.
3. Qwen returns exact transcription candidates and table cells, preferably in a small JSON object.
4. Local code validates completeness and compares native, OCR, and VLM values.
5. A confirmed value becomes canonical evidence; an unresolved disagreement is flagged for review or forces the answer layer to abstain.

This is safer and cheaper than sending every page and trusting a single generated transcription. It also allows public documents to use the external fallback while confidential documents remain on an entirely local path.

## The next evidence-producing experiment

Do not integrate it into `ingestion.py` yet. Run an isolated benchmark on the already known hard pages plus a balanced sample of Arabic prose, French prose, scanned pages, mixed pages, and tables. Render one page or one table region per request, request exact transcription without summarization, disable reasoning for extraction, and validate the JSON locally.

Score at least:

- exact critical-number recall;
- date recall;
- table row-to-label association;
- omitted content;
- hallucinated content;
- Arabic reading order;
- latency and returned token usage per page.

Adopt it only if it materially improves the 157 visually verified evaluation cases and reduces unresolved conflicts without damaging prose. The decisive comparison is not "Qwen versus Docling"; it is current Docling, Qwen-only, and Docling plus selective Qwen fallback.
