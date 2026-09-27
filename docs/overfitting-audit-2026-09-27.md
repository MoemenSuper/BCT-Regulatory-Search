# Overfitting audit — backend heuristics (2026-09-27)

Read-only audit. No code changed. Scope: `backend/answer_draft.py`, `answer_gates.py`,
`answer_evidence.py`, `retrieval_selection.py`, `conversation.py`, `supersession_pin.py`,
`runtime_retrieval.py`, `query_authority.py`, `source_documents.py` (+ spillover into
`graph_contract.py`, `supersession_edges.py`).

The benchmarks these heuristics match:

- **Suite B** — `backend/tests/benchmark/suite_b/human_questions.jsonl`. Almost every item targets
  three PDFs: `Cir_2025_13_fr.pdf` (export settlement: 120 days, 121–360 days, articles 10/11/12
  nouveau), `Cir_2020_02_fr.pdf` (60 days / 61–360 days) and `Cir_2026_04_fr.pdf` (100% deposit for
  non-priority imports; Art.4 exclusions for marchés publics / entreprises industrielles + fiche
  technique; transitional rule for engagements made before 26 mars 2026). Examples: B-001, B-014,
  B-020, B-027, B-029, B-036, B-037, B-040, B-044, B-045, B-062, B-094, B-096, B-CONV-098..105.
- **Stats stress** — `.tmp/stats_live_stress.py:66-150`: chômage in États-Unis / Japon / Zone Euro,
  "inflation en Tunisie en juin 2026" followed by "Et combien était-il en juin 2025 et en juin
  2024 ?", BCE meeting of 11 juin 2026, US growth forecasts (avril/juillet 2026).
- **Stress gate cases** — `backend/tests/stress/cases.py:843, 995, 1017` (adversarial claim strings
  that some gate regexes copy word for word).

Legend: **Safety** means removing the heuristic would weaken quote verification, polarity, number,
authority or currentness validation. **Nicety** means it only affects recall, ranking or formatting.

---

## Ranked findings (worst overfit / highest risk first)

### 1. `try_literal_evidence_partial` topical templates — DELETE the templates, keep the generic branch
- **Where:** `answer_draft.py:989-1148`. Topic flags at `1000-1007`, topical regexes at `1009-1032`,
  canned legal claim text at `1078-1103`, `status="answered"` for templated claims at `1113-1116`,
  recency bias `score += year/1000` at `1062-1064`.
- **Hard-codes:** four topics — marchés publics/collectivité, entreprises industrielles/fiche
  technique, a historical 60-day or 61–360 band, and a 120-day free window. Each topic gets a
  pre-written French legal conclusion ("les importations dans le cadre de marchés publics sont
  exclues…", "le délai libre sans condition était de 60 jours…", "les ventes jusqu'à 120 jours
  peuvent être réglées librement…").
- **Tuned to:** B-020 / B-CONV-102 (marchés publics), B-027 / B-029 / B-036 / B-CONV-105
  (industrial + fiche technique), B-014 / B-040 / B-CONV-101 (60/61), B-001 / B-011 / B-044 (120).
- **Callers:** `generate_grounded_answer` at `answer_draft.py:1597-1603` (last resort); re-exported by
  `answer_contract.py`; one test file.
- **Safety?** No. The claim text is written by code, not taken from the evidence, and still passes
  through `parse_answer`. But it can emit `answered` with a French-only canned sentence (wrong
  language for AR/EN questions) and a conclusion the quote only partly supports. Removing it is
  strictly safer.
- **Recommendation:** DELETE `wants_*`, `topical_patterns` and the canned claims. Keep only the
  generic anchor-excerpt branch as `partial_answer`, and drop the year recency bonus.

### 2. `try_multipart_stat_answer` + `_STAT_GEOS` — DELETE (or reduce to a prompt hint)
- **Where:** `answer_draft.py:764-934`, `_STAT_GEOS` `937-944`, `_named_stat_geos` `947-949`,
  `_snippet_near_aliases` `952-986`. Call site at `1160-1165`.
- **Hard-codes:** three geographies (US / Japon / Zone Euro); the words `inflat` / `chômage`; the
  months `juin|mai|avril` (`829-836`); the phrasing "X % contre Y % une année auparavant et Z % juin
  YYYY" (`807-813`), where the year of Y is computed as `prior_year + 1` (`816-821`); exclusion words
  `produits alimentaires|transformés` (`843`) and `croissance|prévisions|pib` (`801-804`).
- **Tuned to:** stats stress cases 1, 2+3, 7, 8 (`.tmp/stats_live_stress.py:66-150`) and
  `tests/test_multipart_stat_answer.py`.
- **Callers:** only `generate_grounded_answer` (`answer_draft.py:1161`).
- **Safety?** Yes, and removing it makes the system safer:
  - It runs **before** the named-instrument check (`1166-1169`) and before evidence selection.
  - It forces `query_class="statistical_fact"` (`927`) whatever the classifier said, which widens
    the authority gate in `answer_gates.py:183-191`.
  - It pins a figure to a year by arithmetic ("une année auparavant" → year + 1). The claim's year
    then passes only because the question's years are trusted (`trusted_years`).
- **Recommendation:** DELETE. The selector and draft prompts already tell the model to cover every
  named entity (`476-481`, `1305-1309`). If a deterministic path is still wanted, run it after
  selection, keep the classifier's `query_class`, and drop the year arithmetic.

### 3. Lexical "regulatory regime" system — DELETE
- **Where:** `retrieval_selection.py`:
  - `_EXPORT_SETTLEMENT_QUERY` `290-343` (the comment at `291` admits it covers only "these two"
    regimes);
  - `_NONPRIORITY_IMPORT_QUERY` `345-356`;
  - `_EXPORT_SETTLEMENT_DOC` `359-370` (literally "120 jours", "121 à 360", "article 10/11/12
    nouveau", "librement et sans autorisation");
  - `_NONPRIORITY_DOC` `372-390` (includes bare `100\s*%` and `sont exclues?`, which match many
    unrelated pages);
  - `EXPORT_SETTLEMENT_BM25_QUERY` / `NONPRIORITY_IMPORT_BM25_QUERY` `392-405`;
  - `query_regulatory_regime` `408-424`, `_doc_matches_regime` `427-434`, `*_DOC_STRONG` `437-449`;
  - `prefer_regime_hits` `457-519` (newest-year-first sort at `492-495`; industrial / marchés
    publics promotion at `496-513`), `_partition_newest_pattern` `522-537`.
- **Hard-codes:** the vocabulary and article numbers of Cir 2025-13 and Cir 2026-04.
- **Tuned to:** all of Suite B's export-settlement and non-priority-import items (B-001…B-109).
- **Callers:**
  - `runtime_retrieval._identity_diversified_rank` `:87`;
  - BM25 seed lanes `runtime_retrieval.py:148-167` and `276-294` (duplicated in both backends);
  - `supersession_pin.pin_supersession_edges` `:380, 439`, `_edge_fits_query_regime` `:256-287`,
    `_newer_regime_hit_exists` `:290-305`;
  - `answer_draft.generate_grounded_answer` `:1231-1268`;
  - unused imports in `supersession_edges.py:31-39`;
  - tests in `test_runtime_retrieval.py`.
- **Safety?** Ranking only, apart from the answer_draft use (see #4). The newest-year sort inside a
  bucket (`492-495`) is a recency shortcut with no supersession edge behind it.
- **Recommendation:** DELETE the whole family. Named-instrument lanes, supersession edges and the
  reranker are the general mechanisms. If recall drops, add doc-kind or topic metadata at ingest
  instead of query regexes.

### 4. Newest-regime-year override inside answer drafting — DELETE
- **Where:** `answer_draft.py:1226-1268`. Its comment names "marchés / industrial answers when 2026
  is present".
- **Hard-codes:** after the LLM selector has chosen evidence, the selection is replaced by every
  usable page from the newest year that matches the regime regex.
- **Tuned to:** B-020, B-027, B-029, B-094 (2026-04 beating the 2017/2018 list circulars).
- **Callers:** inline in `generate_grounded_answer`.
- **Safety?** Currentness. It prefers the newest document without a supersession edge, which
  contradicts the prompts' own rule (`514-518`, `1329-1332`) and AGENTS rule 5. Removing it
  strengthens currentness.
- **Recommendation:** DELETE.

### 5. Prompt paragraphs that name benchmark cases — DELETE those lines, keep the general rules
- **Selector prompt** (`answer_draft.py`):
  - `460-467`: payer un fournisseur vs délais de règlement des ventes; envoyer de l'argent vs
    traveler cash-export; bureau de change; Middle-Office.
  - `476-481`: États-Unis / Japon / Zone Euro; inflation 2024/2025/2026.
  - `483-484`: "what about gold".
  - `530-532`: "Avant 2025-13".
  - `539-542`: entreprises industrielles + Art.4 / fiche technique vs 100% deposit.
  - `546-547`: "ancien régime vs new circular, 60 vs 120".
  - `559-575`: an out-of-scope / in-scope list — IS, CNSS, congés, sanctions list, voyage
    d'affaires, paiement anticipé fournisseur, chômage, BCE.
- **Draft prompt:**
  - `1290-1293`: payer un fournisseur / envoyer de l'argent.
  - `1305-1309`: États-Unis / Japon / Zone Euro / BCE.
  - `1316-1317`: "Selon la circulaire 2016-01".
  - `1370`: "La circulaire 2021-03 abroge la circulaire 2016-01". This names a specific edge in a
    prompt — close to breaking hard rule 1.
  - `1387-1389`: "Avant 2025-13".
  - `1393-1394`: "60 vs 120, ancien régime vs 2025-13".
  - `1405-1407`: "industrial importer must deposit 100% … fiche technique / Art.4".
- **Forced-partial prompt:**
  - `1710-1711`: "note 2024-163", "one credit facility".
  - `1725`: example claim "la circulaire 2022-12 … or monétaire".
- **Retry hints:** `1804-1805` ("entreprises industrielles", "tous les importateurs").
- **Tuned to:** Suite B (B-027/029/036/037/040/096/CONV-101), the stats stress cases, and Langfuse
  traces in `.tmp/diag`.
- **Safety?** Niceties. The general rules around them (same audience/operation/regime; keep
  exceptions and conditions; cover every named part; prior-regime vs transitional) carry the value.
- **Recommendation:** DELETE the named cases. Replace instrument ids in examples with neutral
  placeholders such as `circulaire AAAA-NN`.

### 6. Supersession pin tuned for P@5 — GENERALIZE
- **Where** (`supersession_pin.py`):
  - `_retain_classic_lead_hits` `308-363` and its call at `442` ("so P@5 still sees mined PDFs",
    `315-318`, `441`);
  - soft mode of `_demote_fully_superseded` `210-221` ("so P@5 / citation still sees", `212-213`);
  - magic scores `8000/8500/9000` at `217, 244, 247, 359, 419, 428`;
  - regime filters `_edge_fits_query_regime` `256-287` and `_newer_regime_hit_exists` `290-305`,
    used at `386-397`;
  - `_CURRENTNESS_QUERY` `166-171`, which duplicates `graph_contract._EXPLICIT_CURRENT_PATTERNS`
    (`graph_contract.py:76-87`).
- **Tuned to:** the retrieval eval metric (P@5 / document-grounded eval) and B-051 / B-066 /
  B-CONV-104 (2026-04's visa of 2025-13 must not pin as an export edge).
- **Callers:** `pin_supersession_edges` → `SupersessionPinBackend.retrieve` (`457-461`) →
  `maybe_wrap_backend` (`runtime_retrieval.py:436-439`).
- **Safety?** Ranking only. Pinning and demotion never change what a gate accepts. Declaring pages
  are still rendered as relationship evidence.
- **Recommendation:**
  - Carry an explicit `pinned=True` metadata flag instead of score sentinels.
  - Drop `_retain_classic_lead_hits` and the soft/aggressive split; use one demotion rule.
  - Replace `_CURRENTNESS_QUERY` with `graph_contract.is_temporal_rule_query`.
  - Remove the regime filters along with #3. The edge extractor already rejects Vu citations
    (`supersession_edges.py:77+`).

### 7. Conversation routing word lists — GENERALIZE / DELETE
- **`_enrich_followup_rewrite`** `conversation.py:141-180`:
  - carries a fixed list (`inflation, chômage, chomage, taux, tunisie, conjoncture, glissement`,
    `152-162`) into the rewrite;
  - appends the prior PDF filename to the retrieval query (`164-179`).
  - Tuned to stats stress case 3 ("Et combien était-il en juin 2025…").
  - Called by `route_message` `:420`; covered indirectly by
    `test_conversation_followups.py:128,159`.
  - Nicety. GENERALIZE: carry the prior standalone query's content tokens that are missing from
    the fragment (the `prior_tokens` logic at `170-175` already exists), with no fixed list.
- **`_CORPUS_FACT_MARKERS` / `_looks_like_corpus_fact_lookup`** `202-228`, `239-245`:
  - forces GENERAL_CHAT → NEW_TOPIC on substrings like `rapport`, `évolution`, `fed`, `japon`,
    `zone euro`. `fed` matches inside any word and "par rapport" matches `rapport`.
  - Tuned to stats cases 1 and 6 and `test_conversation_followups.py:91,111`.
  - Nicety. DELETE; the router prompt already covers it at `338-341`. If kept, match whole words
    only.
- **`_FOLLOWUP_PREFIXES` / `_FOLLOWUP_MARKERS`** `58-86`:
  - `"كيف "` (Arabic "how") at `68` turns every Arabic "how…" question into a FOLLOW_UP once memory
    exists, overriding the router (`409-426`). `"combien était"` / `"était-il"` are tuned to stats
    case 3.
  - Nicety, but it is a real misrouting bug. GENERALIZE: only override when the router returned
    GENERAL_CHAT/AMBIGUOUS, and drop `"كيف "`.
- **Router prompt examples** `353-362` (acompte 25 pourcent prestataire étranger; chômage US / Japon
  / Zone Euro; BCE 11 juin 2026; inflation juin 2025 / 2024 "suite au taux de juin 2026") and the
  lists at `329-341`:
  - These are the stress questions word for word. Nicety. GENERALIZE with neutral examples.
- **`_prefer_prior_turn_sources`** `119-138`: general mechanism (follow-ups prefer the prior turn's
  PDFs). Its docstring names Balance/Conjoncture. KEEP.

### 8. `_order_evidence_for_question_year` override — GENERALIZE
- **Where:** `answer_draft.py:48-66`, used at `1209-1224`. The comment names "en 2024 →
  Note_2024_*".
- **Hard-codes:** when the question names a year, it replaces the selector's choice with **all**
  same-year evidence.
- **Tuned to:** "note 2024-163" credit-facility questions; the same idea appears at
  `1710-1713`.
- **Safety?** Nicety, but it throws away the selector's audience/regime filtering, which weakens
  the "same regime" discipline.
- **Recommendation:** GENERALIZE: reorder same-year evidence first inside the selector's picks;
  never replace the selection.

### 9. `query_authority` gold set and keyword shortcut — GENERALIZE
- **`_GOLD`** `query_authority.py:40-116`: names benchmark instruments (2026-04 at `42` and `102`,
  2025-13 at `92`, 2016-01 at `57`, note 2024-03 at `82`). Nicety. Replace ids with placeholders.
- **`_looks_mixed_authority`** `198-232`, used at `237-242`:
  - any regulatory word (`plafond`, `délai`, `réglementation`…) combined with any stats word
    (`inflation`, `conjoncture`…) returns `mixed` with high confidence, skipping the LLM.
  - Tuned to stats case 4 ("Selon la réglementation BCT et les publications statistiques…").
  - Safety (authority): `mixed` allows statistical **and** internal evidence to prove claims
    (`allowed_doc_kinds` `119-128`). GENERALIZE: pass it as a hint to the classifier instead of
    short-circuiting, or return `mixed` with `confidence="low"`.

### 10. Claim gates that copy corpus nouns and stress strings — KEEP (safety), generalize wording
- **`_SUPPORT_NARROW_POPULATION`** `answer_gates.py:356-361`: entreprises industrielles, produits
  non prioritaires, marchés publics, collectivités locales, "jusqu'à N jours" (Cir 2026-04 /
  2025-13 nouns).
- **`_CLAIM_UNIVERSAL_SCOPE`** `350-355`: "tous les importateurs", "commerce extérieur mondiaux",
  "un seul moyen impos" (from `tests/stress/cases.py:1017`).
- **`_CLAIM_DROPS_RESERVE`** `376-380`: the literal "toute importation industrielle est exclue"
  (from `cases.py:843`).
- **`_SUPPORT_EXCLUSION`** `267-286`: treats `sous réserve` / `شريطة` / `مع مراعاة` (a condition)
  as *exclusion* polarity, tuned to B-027 / B-036.
- **`_CLAIM_EXCLUSION`** `303-320`: treats `librement` / `sans autorisation` as exclusion, tuned to
  2025-13's "librement et sans autorisation".
- **`_reject_threshold_boundary`** `500-516`: a day-range window `30 <= n <= 400` (`507`), tuned to
  60/61, 120/121 and 360.
- **Callers:** `_validate_claim` `answer_gates.py:252-257` via `parse_answer`.
- **Safety?** Yes — polarity, scope, condition and number gates. Deleting them would weaken fail-closed
  behaviour.
- **Recommendation:** KEEP. GENERALIZE later:
  - Scope: "claim universal quantifier + quote restricted subject" using generic determiners
    (tous/toutes/tout/any/all vs a noun phrase qualified in the quote), not corpus nouns.
  - Drop the literal "toute importation industrielle est exclue".
  - Split `sous réserve` into a "condition" class instead of "exclude".
  - Apply the ±1 boundary check to any integer, not only 30–400.

### 11. `is_historical_cutoff_query` / `_GRANDFATHERING_AVANT` — GENERALIZE
- **Where:** `retrieval_selection.py:540-573`.
- **Hard-codes:** any `avant` / `before` / `قبل` makes the query "historical", unless the
  grandfathering list matches (engagements pris, exécution … avant, financement … avant).
- **Tuned to:** B-040 / B-CONV-101 ("Avant 2025-13") and B-062 / `test_context_and_abstention.py`
  (engagement before 26 mars 2026 under 2026-04).
- **Callers:**
  - `prefer_historical_hits` `608-627`, used by `runtime_retrieval.py:90` and
    `supersession_pin.py:441,445`;
  - `prefer_regime_hits` `493`;
  - `answer_draft.py:1003,1036,1229`.
- **Safety?** Ranking, plus it toggles the currentness preference at `answer_draft.py:1229`. A
  false positive ("avant de payer…") silently switches to historical mode.
- **Recommendation:** GENERALIZE: require `avant` to be followed by an instrument ref or a date
  (`avant\s+(?:la\s+)?(?:circulaire\s+)?\d{4}-\d+|avant\s+(?:le\s+)?\d{1,2}\s+<mois>`).

### 12. `try_supersession_partial_answer` head-of-page fallback — GENERALIZE (grounding weakness, not a benchmark tune)
- **Where:** `answer_draft.py:631-761`. The fallback at `671-677` quotes the first 45 words of the
  page when no abrogation verb is found. The canned claim at `689-693` states "X abroge/remplace/
  modifie … n'est plus en vigueur".
- **Callers:** `generate_grounded_answer` `:1590-1595`; re-exported in `answer_contract.py`.
- **Safety?** Quote verification. The fallback quote does not support the relationship claim, so
  the claim is backed only by edge metadata. The second claim ("énonce la disposition applicable",
  `720-723`) is vacuous.
- **Recommendation:** GENERALIZE: require `abr_re` to match inside the quote (drop the head
  fallback), and drop the vacuous second claim.

### 13. `question_scenario_numbers` — KEEP
- **Where:** `answer_evidence.py:226-260`, used at `answer_gates.py:231-238`.
- **What it does:** lets a claim restate day/year numbers from a calendar date in the question
  without quoting them. Tuned to "avant le 26 mars 2026"
  (`tests/test_context_and_abstention.py:192-196`).
- **Safety?** It slightly loosens the number gate, but it is limited to dates the user wrote.
  KEEP.

### 14. Minor corpus conventions — KEEP
- `supersession_pin.build_page_lookup_from_native` aliases `CB_` ↔ `Cir_`, `_FR` ↔ `_fr`
  (`483-493`). These are filename conventions, not benchmark tuning.
- `retrieval_selection._SOURCE_ID` / `_KIND_ALIASES` (`13-16`, `107-119`): filename identity.
- `source_documents.py`: **no benchmark-specific heuristics found.** Trusted-root resolution,
  quote location and the optional Gemini locator are generic.
- `answer_evidence.py`: generic apart from #13.

---

## Dead code and unused imports

| Location | Finding |
|---|---|
| `graph_contract.py:134-141` | `GraphRetrievalResult` (and `requires_temporal_abstention`) has no references outside its own docstring (`:4`). DELETE. |
| `answer_gates.py:655` | `ANSWER_SCHEMA` is only imported (`answer_draft.py:29`) and re-exported (`answer_contract.py`). Prompts use `ANSWER_SCHEMA_FOR_PROMPT`; no test uses it. DELETE unless an external client imports it. |
| `answer_draft.py:13-40` | Imported but unused: `numeric_literals`, `supported_numbers`, `trusted_years`, `strip_instrument_references`, `claim_asserts_unverified_applicability`, `question_scenario_numbers`, `AnswerDraft`, `ANSWER_SCHEMA`, `_TEMPORAL_LIMITS`, `_SUPPORT_EXCLUSION`, `_PAGE_HAS_RESERVE`, `ARABIC`, `ValidationError`. |
| `runtime_retrieval.py:9-19, 26` | Unused imports: `tempfile`, `time`, `contextmanager`, `ContextVar`, `dataclass`, `Lock`, `requests`, `expand_ranked_pages`. (Cloud client names at `47-57` are re-exported and used by `tests/test_token_quota.py:8` and `test_runtime_retrieval.py:379` — keep.) |
| `supersession_edges.py:23-40` | Unused imports: all `retrieval_selection` names (`parse_source_identity`, `prefer_historical_hits`, `prefer_named_instrument_hits`, `prefer_regime_hits`, `query_instrument_refs`, `query_regulatory_regime`, `_doc_matches_regime`), plus `Document`, `normalize_page`, `lru_cache`, `defaultdict`. |
| `supersession_pin.py:6, 11` | Unused imports: `os`, `Iterable`. |
| `answer_draft.py:1230-1231` | Function-local imports of `Document`, `_doc_matches_regime`, `query_regulatory_regime`. They go away with #3/#4. |

All other audited functions have at least one production caller (verified with `rg` excluding
`tests/`).

---

## Suggested deletion order (smallest safe steps)
1. Remove #4 (newest regime-year override). This is a currentness fix on its own.
2. Remove #2's pre-selection call (`answer_draft.py:1160-1165`), or move it after selection
   without forcing `query_class`.
3. Remove the topical templates from #1.
4. Remove #3 (regime family and BM25 seeds), the regime filters from #6, and the unused imports.
   Re-run Suite B retrieval (`run_live_retrieval.py`) to measure the recall change.
5. Prompt cleanup (#5) and router cleanup (#7).
6. Generalize the gates (#10), the historical detector (#11) and the supersession partial (#12),
   with focused tests under `backend/tests/`.
