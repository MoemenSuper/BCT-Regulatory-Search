# Chat pipeline diagnosis — 2026-09-27

Scope: why the answer layer abstains when retrieved evidence holds the answer, plus other
retrieval/answer defects. Method: temporary Langfuse tracing on every chat turn (route →
query authority → retrieval lanes → selection → draft gates → forced/literal partials →
final status), 22 real questions from the conversation log (French, Arabic, dialect, English,
statistics, three follow-up pairs) run through the real API under `local_hybrid` on
`baked-runtime-assets` (active version `20260927T130331Z`), plus the 31 rows of the
existing refusal log. No validation was changed.

Run result: 11 `answered`, 10 `partial_answer`, 1 `search_results`. Most "abstentions"
now surface as content-free or wrong partials rather than refusals, so both are covered below.

## Findings, ranked by impact

### 1. Supersession pin is never applied in `local_hybrid` / `local`  (bug, high)
`app.create_local_backend()` returns a bare `LocalRetrievalBackend`; only
`create_voyage_backend_from_environment()` calls `maybe_wrap_backend`. The JSONL edges
(146 loaded, `/health` says `ready: true`) never reach retrieval in the local profiles.

Trace evidence: "heures d'ouverture du marché des changes interbancaire" retrieved
`Cir_2016_01 p2` and `Cir_2021_03 p2`; no `temporal_relation`/`graph_role` in the selection
or draft prompt. The selector itself said "the passages differ … partial pending
clarification of the applicable circular", so the answer lists 2016-01 (abrogated by
2021-03 art. 61 and 2021-02 art. 53) as if still current. Offline replay of
`pin_supersession_edges` on the same hits pins `Cir_2021_02 p10` and `Cir_2021_03 p14`
with `ABROGATES` — the pin works, it just is not wired. Same defect explains "Quelle
circulaire réglemente les opérations de change ?" → "2016-01, 2021-03 and 2025-12 all
regulate …".

Fix: wrap the local backend the same way (`maybe_wrap_backend(backend, active_assets)`),
add a test that `create_local_backend()` returns a `SupersessionPinBackend` when edges
exist.

### 2. `unsupported_claim_anchor` rejects correct claims on French participles  (bug, high)
`answer_gates._anchor_on_page` stems by raw prefix. Question word `abroge` vs page word
`abrogée`: "abrog-é-e" does not start with "abrog-e", so the anchor is "not on page",
and `_reject_regime_remapped_claim` fires. Every base-form verb in a question whose
page uses the participle is affected (modifie/modifiée, remplace/remplacée,
autorise/autorisée, applique/appliquée, exige/exigée).

Trace evidence: "La circulaire 2018-13 abroge-t-elle une circulaire antérieure ?" — the
model answered 4 times with the verbatim quote "Article premier : Est abrogée la circulaire
n° 2017-09 …"; all 4 rejected `unsupported_claim_anchor`; final answer became the
placeholder in finding 4. Offline replay (`.tmp/diag/repro_anchor.py`): current gate →
`insufficient_evidence`; accent-folding both sides inside `_anchor_on_page` → `answered`
with no other gate firing. Quote verification is untouched by this fix.

Fix: NFKD-fold + strip combining marks for anchor and page tokens in `_anchor_on_page`;
regression test with the 2018-13 page.

### 3. Groq 413 "Request too large" makes selection and draft fail deterministically  (design, high)
Free-tier `openai/gpt-oss-120b` allows 8 000 tokens per request-minute. Selection and draft
prompts are 22–60 k characters (the draft system prompt alone is ~3.5 k tokens; evidence
is 3–5 whole expanded pages, e.g. 15 780 chars for Conjoncture pages). Key rotation cannot
help — every key has the same per-request ceiling, and 413 is not a rotate condition.

Trace evidence: Brent question — selection `413 … Requested 10485`, both drafts 413,
ladder fell to forced partial. Every statistics turn on Conjoncture pages hit at least one
413 (inflation, Brent, Brent follow-up). The refusal log shows the same (`selection_error:
Error code: 413`, and 429 TPD exhaustion on 2026-09-16/18 because one turn burns 4–9 calls
of ~6–10 k tokens).

Fix options (no validation change): trim evidence to on-topic windows *before* selection
(not only in the forced path), shorten the draft system prompt (much of it is
case-specific rules), and pre-check prompt tokens so an oversize prompt is split instead
of sent.

### 4. Literal partial returns a content-free sentence as `partial_answer`  (bug, high)
`try_literal_evidence_partial` falls back to
"Selon {label}, le passage cité énonce la disposition applicable à la demande." when no
topical pattern matches. That passes the gates (it has a real quote) and is shown as a
partial answer with a citation, but carries no fact. Seen on the 2018-13 and Brent questions.
This is `search_results` dressed as an answer.

Fix: either use the quote itself as the claim text ("Selon {label} : « … »"), or return
`None` so the turn ends as `search_results`.

### 5. Forced-partial compaction drops the answering sentence  (bug, medium)
`_compact_evidence_for_formulation` keeps one 420-char window per page picked by question
anchors. For Brent, page 3 contains "le cours moyen du baril de Brent a augmenté de 19,4 %
en glissement annuel au mois de juin 2026", but the window chosen was the neighbouring
"indice des prix … juin 2026" sentence. The model then correctly said the information was
missing. Fix: rank windows by anchor coverage including rare anchors (Brent) and keep up
to two windows per page.

### 6. Selector "insufficient_evidence" override can yield a confident wrong `answered`  (bug, medium)
When the selector says `insufficient_evidence`, the ladder makes it advisory and drafts
anyway. For "est-ce que la banque peut prendre une commission sur un virement reçu de
l'étranger ?" the selector said "passages discuss commissions on the purchase of foreign
banknotes … do not address an incoming transfer"; the draft then answered `answered` with
the banknote commission (and cited the abrogated 2016-01). The regime gate misses it
because the page shares "commission"/"banque". Fix: when the selection was overridden,
cap the accepted status at `partial_answer` and pass the selector's reason into
Selection limits.

### 7. Query authority is classified on the raw follow-up fragment  (bug, medium)
`conversation.chat` calls `classify_query_authority(llm, message)`. All three follow-ups
("Et combien était-il en juin 2025…", "Et qu'en est-il de son évolution…", "Et pour une
petite entreprise ?") came back `uncertain`; `uncertain` allows only regulatory documents,
so statistical follow-ups fail `authority_mismatch:uncertain:statistical` (trace + refusal
log 2026-09-26). Fix: classify `route_query` (the resolved rewrite), or inherit the prior
turn's class for FOLLOW_UP.

### 8. Follow-up rewrite overrides good LLM rewrites  (behaviour change, medium)
`_enrich_followup_rewrite` replaces the model's rewrite with
"{prior} — suite: {message}" whenever it shares fewer than 3 long tokens with the prior
query, always appends the prior source filename, and carries a hard-coded keyword list
(inflation, chômage, taux…). This is why the pre-existing test
`test_route_message_validates_a_follow_up_rewrite_against_memory` fails: the clean
"current deadline under Circular 2019-07" becomes
"changes made by Circular 2019-07 — suite: What about the deadline? Cir_2019_07_fr.pdf".

### 9. Multipart statistics claim repeats the raw sentence per year  (bug, low)
`try_multipart_stat_answer` computes the per-year value (`_pct`) but writes
`f"En {year}: {snippet}"`. Output for June 2025/2024 inflation: the same sentence twice,
labelled 2024 and 2025, never stating 5,4 % / 7,3 %. Fix: claim text from `_pct`, quote
unchanged. The regex is also bound to one Conjoncture phrasing (juin/mai/avril, "une année
auparavant et … en juin YYYY").

### 10. Other issues
- **Silent empty corpus.** Pointing `--assets` at an asset root whose active snapshot has no
  local Chroma (e.g. `session-20260905/runtime-assets`) starts cleanly, then every turn is
  `no_retrieval_hits` in 2 s. Startup should fail when the local collection or BM25 corpus
  is empty.
- **Answer language.** English question answered in French (`language_of` = `en`, prompt
  asks for question language; nothing checks it). The appended multi-page note was English,
  so the answer mixed languages.
- **Refusal reason noise.** `chat()` appends `query_class:…` to diagnostics before
  `format_refusal_reason`, so every stored refusal reason ends in `| query_class:…`; this is
  the pre-existing `test_chat_keeps_refusal_reason_without_user_facing_diagnostics` failure.
- **Error mapping.** `post_chat` maps any `ValueError` to 503 "Selected runtime is
  unavailable". In the `local` profile an empty Ollama reply raises `ValueError` from the
  draft loop (which only catches `APIError`/`RequestException`), so it surfaces as a 503.
- **Router.** The route model returned `GENERAL_CHAT` for the Brent question; the
  validation fallback corrected it to `NEW_TOPIC`, so no user impact, but it is one
  prompt edit away from sending statistics questions to general chat.
- **Case-specific heuristics.** Literal partial (marchés publics, fiche technique, 60/120
  jours), follow-up carry words and multipart regexes are tuned to specific benchmark
  questions; they help those and silently do nothing elsewhere.

## Refusal log (31 rows, read-only)
29 `search_results`, 2 `out_of_scope` ("test"). Dominant causes: provider limits (429 TPD
and 413 TPM — 13 rows), `schema_invalid` → repair → `draft_abstained:insufficient_evidence`
(cloud profile, 2026-09-16), `authority_mismatch:uncertain:statistical` on follow-ups
(finding 7), `named_instrument_absent:cir:2025-13` (circular not in that corpus), and
`unsupported_claim_anchor` / `quote_not_found` gate rejections.

## Suggested order
1 (wire pin) and 2 (accent fold) are small, safe, and fix wrong/abstained answers directly.
Then 4 and 6 (stop presenting non-answers or wrong answers as answers), 7, 3, 5, 9.

## Fix status (same day)
All findings except 3 (Groq 413, a free-tier limit) are fixed; benchmark-tuned heuristics from
`docs/overfitting-audit-2026-09-27.md` were removed. Tests: `backend/tests` 440 passed.

| Finding | Change |
| --- | --- |
| 1 pin | `create_local_backend()` wraps with `maybe_wrap_backend`; pin simplified (no regime filters, no P@5 lead retention, one demotion rule; named-instrument order kept unless the question is a currentness one via `is_temporal_rule_query`). |
| 2 anchor | `_anchor_on_page` accent-folds both sides. |
| 4 literal partial | `try_literal_evidence_partial` deleted; supersession partial keeps only the edge claim and requires the relation verb inside its quote. |
| 5 compaction | `_verbatim_excerpt` picks the window covering the most question words and keeps end-of-page sentences. |
| 6 selector override | Overridden `insufficient_evidence` caps `answered` → `partial_answer`, passes `selector_doubt` into Selection limits, and stops at `search_results` when the first draft also abstains. |
| 7 authority | Classified on the resolved `route_query`; keyword "mixed" shortcut removed. |
| 8 follow-up | Router's FOLLOW_UP rewrite kept verbatim; misrouted fragments get `{prior query} — {message}`; carry words, filename append, stats word-list rescue and `"كيف "` marker removed. GENERAL_CHAT replies `RETRIEVE` for fact questions so they are searched instead of answered uncited. |
| 9 multipart | `try_multipart_stat_answer` deleted. |
| 10 | Empty local corpus warns at startup; answer language passed as a full name in both draft prompts; refusal reason no longer carries `query_class`; empty Ollama reply is an empty draft, not a 503. |
| Overfit | Regime lexicon/BM25 seeds/`prefer_regime_hits`, newest-regime-year and question-year selection overrides, case-named prompt lines and gold-set instrument ids removed; `avant` cutoff needs an instrument ref. |

Rerun (`local_hybrid`, `baked-runtime-assets`, 23 turns): 2018-13 answered; opening hours answered
from Cir 2021-03 with the 2021-02 abrogation pinned; Brent answered from the compacted sentence;
English answered in English; Arabic commission → `search_results` (selector and draft agree);
no content-free partials. Remaining `search_results` on the two statistics follow-ups are Groq 413.

## Temporary tracing (remove after fixes)
`backend/chat_tracing.py` plus call sites in `app.post_chat`, `conversation.chat`,
`answer_draft.generate_grounded_answer` (select/draft/forced gate events) and
`runtime_retrieval.LocalRetrievalBackend.retrieve` (`retrieval-lanes`). Scratch scripts
and results live in `.tmp/diag/` (not for commit).
