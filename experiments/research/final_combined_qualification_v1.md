# Final combined qualification v1

Status: predeclared before checkpoint implementation or result scoring

This bounded program combines only the development components already selected
by prior controlled experiments. It does not change production code, source
PDFs, persistent Chroma data, retrieval candidate budgets, reranker model, or
the provisional/final evaluation boundary.

## Frozen combined retrieval and answer input

The retrieval input is the kept document-identity candidate at commit
`489ed074ab8f037e32dadc5a6eaec0f4a0d4102e`:

`StructuredDocument -> page-local sequential ~1000/200 chunks -> native dense
top-20 + BM25 top-15 -> Arabic additive OCR dense top-5 + BM25 top-5 -> exact
candidate union -> conditional filename-derived identity reranker prefix ->
BAAI/bge-reranker-v2-m3 -> stable source/page-diverse top-5 -> same-page
context -> claim-linked structured answer`

Frozen development hashes:

- answer suite: `5E3D840B6DF8FDF3046AA2A5A67B84DF8CC0E0E61D4EC810E5D51446C89917A1`
- raw retrieval-first answer result: `5E6BE4C8D05A144248821156A03F5789D91D63F2D6928EC599E997905E2DE450`
- identity retrieval result: `4BCBBD54CF1C4D5E3CBE17C8ED29836482E48266BF83BD428B9898866BFCC722`
- identity answer recomposition: `1DAC012D773E94A5C75D512B504C0414668CF4B3EA7ECBE24910F5316F7D87A5`
- Arabic digit-order risk: `4AED0BB5D893D45FA812599F00672403D88A2B3471913BCD1B3D244B3F5FF13C`
- StructuredDocument manifest: `09ABE18E9DCAD9043650EC1B204967642741CE6C2F1EE93F042B83375B59D5BA`

No expected answer, expected source/page, case ID, or verified literal may be
consulted by runtime routing, answer generation, or safety behavior. Gold fields
may be used only after runtime artifacts are frozen, for scoring and manual
review.

## Checkpoint A - page-level Gemini routing and numeric authority

### Hypothesis

When identity-aware retrieval supplies an Arabic numeric/date page, routing
from the actual cited page or the ranked page whose runtime source identity and
page-local representation/risk signals make it answer-bearing can select the
correct page for Gemini verification. A deterministic evidence-authority guard
then prevents the answer generator from changing unsupported digits.

### Single component change

Replace document-wide visual-risk selection with page-level selection. An
Arabic numeric/date page is eligible only when at least one runtime-observable
page signal holds:

1. the exact source/page appears in the frozen digit-order `page_hits`; or
2. the retrieved page is supplied by the additive Arabic OCR representation,
   which is retrieval-only and non-authoritative for numeric/date claims.

For an answered response, consider only claim-linked/cited pages. For an
insufficient-evidence response, rank eligible pages by an exact match between
the parsed runtime query identity and filename metadata, followed by the frozen
retrieval order. Select at most two distinct pages. No document-wide flag makes
an unrelated page eligible.

Gemini remains `gemini-3.7-flash`, low thinking, with the existing strict
transcription contract and maximum two pages per query. New cache keys bind PDF
hash, page, rendered-image hash, model, prompt version, and the complete visual
configuration. Existing pages may be reused only after their containing
artifact configuration and each page binding validate exactly.

Visual text replaces, rather than concatenates with, the selected native/OCR
text. Malformed, unavailable, incomplete, uncertain, or cache-mismatched output
fails closed. After answer generation, every asserted number/date literal must
occur in its linked authoritative evidence and every linked routed-risk page
must carry validated visual provenance; otherwise the response becomes a
non-empty safe abstention.

### Checkpoint-A gate - all clauses required

1. Zero new wrong numeric/date assertions.
2. All four previously verified Arabic corruption repairs remain correct when
   routed.
3. The identity-recovered requested page is selectable without gold fields.
4. Zero routed-control strict-answer or citation regressions.
5. Every affirmative changed claim cites the correct requested source/page.
6. Maximum two pages per query, exact cache binding, and fail-closed tests pass.

A failed clause records `REJECT`; it is not hidden by the later combination.

## Checkpoint B - attribution, completeness, and post-retrieval safety

### Hypothesis

A local post-retrieval validator over the runtime query, retrieved evidence,
claim links, citations, and frozen query-state output can fail closed on wrong
instrument attribution, unsupported numeric claims, visibly incomplete answer
rendering, ambiguity, and unsupported current-status assertions without
preempting evidence-supported historical questions.

### Independent component change

Checkpoint B does not alter retrieval or reranking. It runs after answer and
evidence assembly. It validates citation/evidence identity, requires claims for
an explicitly identified instrument to link to evidence from that instrument,
requires claim numeric literals to occur in linked evidence, and refuses to
mix incompatible versions. A materially unfinished answer field may be
completed only from its already generated, evidence-linked claims; it may not
invent new content.

Ambiguity produces a non-empty clarification only when the frozen query state
reports a missing discriminating detail and post-retrieval evidence does not
establish one unique instrument. Current/in-force/latest wording produces a
non-empty abstention only when retrieved evidence lacks explicit current-status
support. The general query-state classifier is never allowed to preempt an
otherwise validated historical answer.

### Checkpoint-B gate

1. Eight of eight negative/ambiguous cases have safe expected behavior.
2. Zero unsupported current-status claims.
3. Zero query-state false preemptions among the 32 relevant development cases.
4. Zero requested-document attribution regressions.
5. No incomplete affirmative response is presented as complete.

## Final combined development gate and stop rule

Only after A and B have independent frozen artifacts may their composition be
scored over the 40-case development suite. Untouched records are copied from
their frozen input; hosted calls are permitted only for newly routed/changed
cases. Every changed response is reviewed against rendered source PDF pages.

The existing gate remains 32/32 relevant cases strictly correct with correct
requested source/page citation and grounding, 8/8 safe expected negatives,
zero new wrong numeric/date assertions, and zero malformed/conflicting evidence
used affirmatively. Safe relevant abstentions are reported separately and do
not rewrite this gate.

If the strict gate fails, remaining failures are classified as recurring
generalizable defects or bounded pathological legacy-document cases. Further
work is recommended only for a recurring blocker or an unsafe assertion.
Provisional answer validation and the final holdout remain unopened unless the
complete existing gate passes. Once the decision and registry record are
committed and pushed, this program stops without production integration or a
new experiment.
