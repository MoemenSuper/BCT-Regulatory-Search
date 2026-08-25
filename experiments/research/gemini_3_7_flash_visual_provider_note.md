# Gemini 3.7 Flash visual-provider note

Verified on 2026-08-25 against Google-owned documentation before the controlled BCT provider comparison.

- The stable model ID is `gemini-3.7-flash`. It accepts text, image, video, audio, and PDF inputs; produces text; supports structured outputs; and supports low, medium, and high thinking levels. Source: [Gemini 3.7 Flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash).
- Google documents JSON-Schema-constrained output for `gemini-3.7-flash` through `generateContent`, including the REST `generationConfig.responseFormat.text` contract used by this experiment. Source: [Structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).
- Inline data is intended for quick tests and transient processing; Google documents a 100 MB request limit and a 50 MB PDF limit. This experiment sends only the already frozen rendered page PNGs and retains its own cache binding. Source: [File input methods](https://ai.google.dev/gemini-api/docs/file-input-methods).
- Standard paid-tier introductory pricing through 2026-12-31 is USD 0.75 per million input tokens and USD 3.75 per million output tokens, including thinking tokens. Free-tier inputs and outputs are listed as free, but free-tier content may be used to improve Google products while paid-tier content is listed as not used for that purpose. Source: [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Controlled-experiment decision

Change only the visual provider/model from Groq `qwen/qwen3.6-27b` to Google `gemini-3.7-flash`. Preserve the frozen route, rendered-page scale, prompt semantics, strict local output validation, PDF/image/model/prompt cache binding, page budget, downstream answer model, manual PDF review, and predeclared KEEP/REJECT gate. Do not access validation or production indexes.
