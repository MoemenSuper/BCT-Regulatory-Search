"""The checkers ("gates"): every claim the answer model writes is tested here before the user
sees it. parse_answer() is the entry point.

The model does not copy quotes; it cites units by ID ("E2.14" = unit 14 of evidence E2). The code
fetches the exact page text of those units, so a quote can never be misquoted. Then each claim
must pass every check, for example:
- every number in the claim appears in the cited text (and was not misread from a chart);
- the claim keeps the text's meaning: "excluded" is not turned into "required", "may" is not
  turned into "must", a condition ("sous réserve") is not dropped;
- the claim comes from the circular the question names, if it names one;
- it does not say a rule is in force today, or that nothing later changed it.

A claim that fails is dropped; if none passes, the user gets the list of search results instead
("fail closed": better no answer than an unsupported one). These checks compare words and
numbers; they do not prove that a claim is legally correct.
"""
import json
import re
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.retrieval_selection import ARABIC, _ascii_fold
from rag.query_currentness import is_temporal_rule_query
from rag.answer_evidence import (
    plain as _plain, unit_spans, numeric_literals, supported_numbers, direct_identity,
    identity_matches, evidence_warning, trusted_years, strip_instrument_references,
    claim_asserts_unverified_applicability, claim_asserts_unsupported_negative_amendment,
    question_scenario_numbers, confirmed_numbers, strip_structure_references,
)

logger = logging.getLogger(__name__)

class Claim(BaseModel):
    """One sentence of the answer, and the units it cites ("E2.14": evidence E2, unit 14)."""

    # The gates read the exact page text of the cited units; the model never retypes evidence.
    # Stray model fields are ignored.
    model_config = ConfigDict(extra="ignore")
    text: str = Field(min_length=1, max_length=1600)
    cites: list[str] = Field(min_length=1, max_length=12)


class AnswerDraft(BaseModel):
    """The answer model's reply, as JSON: a status, an optional message, and the claims."""

    model_config = ConfigDict(extra="ignore")
    status: Literal["answered", "partial_answer", "insufficient_evidence", "clarification_needed", "out_of_scope"]
    message: str = Field(default="", max_length=600)
    # No cap on the number of claims: a broad question ("les règles des bureaux de change")
    # can need ten supported facts, and each one is still checked on its own.
    claims: list[Claim] = Field(default_factory=list)


def _extract_complete_json_dicts(text, *, require_keys=()):
    """Every complete {...} object in a text, even when the text was cut off mid-way. Walks the
    text character by character, counting braces outside of strings."""
    found = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[i : j + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and all(key in obj for key in require_keys):
                        found.append(obj)
                    i = j + 1
                    break
            j += 1
        else:
            break
    return found


def _salvage_answer_payload(text):
    """Rescue a reply that was cut off: keep its status and the claims that are complete."""
    status_match = re.search(
        r'"status"\s*:\s*"(answered|partial_answer|insufficient_evidence|clarification_needed|out_of_scope)"',
        text,
    )
    if not status_match:
        return None
    status = status_match.group(1)
    claims_at = text.find('"claims"')
    if claims_at < 0:
        return None
    claims = _extract_complete_json_dicts(text[claims_at:], require_keys=("text", "cites"))
    if not claims:
        return None
    if status not in {"answered", "partial_answer"}:
        status = "partial_answer"
    return {"status": status, "message": "", "claims": claims}


def _load_answer_payload(content):
    """The reply's JSON as a dict, unwrapping ``` blocks and stray words around it. Missing
    fields are never filled in: a broken reply is rejected, not repaired with guesses."""
    text = str(content or "").strip()
    if not text:
        raise ValueError("schema_invalid")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        payload = None
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            payload = _salvage_answer_payload(text)
        if not isinstance(payload, dict):
            raise
    if not isinstance(payload, dict):
        raise ValueError("schema_invalid")
    # A model sometimes repeats the JSON schema it was given next to its answer: keep the answer.
    if "status" not in payload and "claims" not in payload:
        salvaged = _salvage_answer_payload(text)
        if salvaged is not None:
            return salvaged
    return payload


# Small words only one of the two languages uses. Counting them over the whole question is
# enough to tell French from English ("During Ramadan, what are the hours?" is English,
# "Compare l'inflation en 2023 et 2024" is French) without another model call.
_ENGLISH_WORDS = set(
    "the is are was were what which who whom whose when where why how of for to in at and or "
    "does do did can could should would must this that these those with from by about please "
    "tell me my our their it its be been has have per much many any there "
    "summarise summarize explain give list show circular circulars".split()
)
_FRENCH_WORDS = set(
    "le la les l un une des du de d est sont quel quelle quels quelles que qu qui quoi pour "
    "par avec dans sur au aux et ou en ce cette ces il elle nous vous je j mon ma mes notre nos "
    "peut doit combien comment pourquoi quand selon c s n pas était étaient".split()
)


def language_of(question):
    """Language the answer is written in: "ar", "en" or "fr" (the default)."""
    if ARABIC.search(question):
        return "ar"
    words = re.findall(r"[a-zà-ÿ]+", question.casefold())
    english = sum(word in _ENGLISH_WORDS for word in words)
    french = sum(word in _FRENCH_WORDS for word in words)
    return "en" if english > french else "fr"


_MESSAGES = {
    "fr": {
        "insufficient_evidence": "Les passages disponibles ne permettent pas une réponse suffisamment étayée. Veuillez vérifier le document original.",
        "clarification_needed": "Veuillez préciser la circulaire, la note ou le type d’opération concerné.",
        "out_of_scope": "Cette question sort de mon périmètre : je réponds uniquement à partir des circulaires, notes et rapports de la BCT, avec citations. Exemple : « Quel est le plafond de l’allocation pour études à l’étranger ? »",
    },
    "ar": {
        "insufficient_evidence": "المقاطع المتاحة لا تكفي لتقديم إجابة موثقة. يرجى مراجعة الوثيقة الأصلية.",
        "clarification_needed": "يرجى تحديد المنشور أو المذكرة أو نوع العملية المقصودة.",
        "out_of_scope": "هذا السؤال خارج نطاق عملي: أجيب فقط انطلاقًا من منشورات البنك المركزي التونسي ومذكراته وتقاريره، مع ذكر المصادر. مثال: «ما هو سقف المنحة للدراسة بالخارج؟»",
    },
    "en": {
        "insufficient_evidence": "The available passages do not sufficiently support an answer. Please check the original document.",
        "clarification_needed": "Please specify which circular, note, or type of operation you mean.",
        "out_of_scope": "This question is outside my scope: I only answer from Central Bank of Tunisia circulars, notes and reports, with citations. For example: “What is the maximum allowance for studies abroad?”",
    },
}


def safe_response(question, status="insufficient_evidence"):
    """A fixed reply (no claims, no sources) in the question's language, for a refusal status."""
    return {"status": status, "answer": _MESSAGES[language_of(question)][status], "sources": []}


def _validate_claim(claim, question, by_id, *, temporal_unverified, target, sources, source_numbers, source_excerpts, query_class=None, secondary=None):
    """Check one claim against the units it cites. Returns the claim with its source numbers
    ("... [1] [2]"), and adds its sources to `sources`. Raises ValueError naming the first check
    that fails; parse_answer then drops the claim.

    The checks, in order: no "in force today" claim on a date question, document kind allowed,
    named circular respected, no unsupported "nothing changed it", every number supported, no
    citation marks inside the claim, then the meaning checks (_reject_...).
    """
    if not claim.text.strip():
        raise ValueError("Empty claim")
    if temporal_unverified and claim_asserts_unverified_applicability(claim.text):
        raise ValueError("unverified_applicability_claim_use_document_scoped_wording")
    excerpts = _cited_excerpts(claim.cites, by_id)
    if query_class is not None:
        from rag.query_authority import evidence_kind_allowed

        for record, _excerpt in excerpts:
            # A statistical bulletin or internal memo is not the legal authority for a rule,
            # but it may state the fact (e.g. a policy-rate history). Flag it; parse_answer
            # downgrades to partial with a trusted notice instead of refusing correct evidence.
            if not evidence_kind_allowed(query_class, doc_kind=record.get("doc_kind")) and secondary is not None:
                secondary.add(str(record.get("doc_kind") or "regulatory"))
    numbers = []
    supporting_text = []
    claim_literals = _plain(claim.text)
    for record, excerpt in excerpts:
        if target and not identity_matches(record["source"], target):
            raise ValueError("requested_document_mismatch")
        supporting_text.append(excerpt)
        # The cited PDF's own filename in the claim is not a number that needs quoting.
        stem = re.escape(record["source"].removesuffix(".pdf"))
        claim_literals = re.sub(r"(?<!\w)" + stem + r"(?:\.pdf)?(?!\w)", "", claim_literals, flags=re.I)
        key = record["source"], record["page"]
        if key not in source_numbers:
            source_numbers[key] = len(sources) + 1
            sources.append({"file": record["source"], "page": record["page"],
                            "score": record["score"], "excerpt": excerpt})
            source_excerpts[key] = [excerpt]
        elif _plain(excerpt) not in {_plain(text) for text in source_excerpts[key]}:
            source_excerpts[key].append(excerpt)
            sources[source_numbers[key] - 1]["excerpt"] = "\n…\n".join(source_excerpts[key])
        numbers.append(source_numbers[key])
    if claim_asserts_unsupported_negative_amendment(claim.text):
        excerpt_blob = " ".join(supporting_text)
        if not claim_asserts_unsupported_negative_amendment(excerpt_blob):
            raise ValueError("unsupported_negative_amendment_claim")
    cited = [record for record, _excerpt in excerpts]
    # Claims are written in the question's language, so their numbers are read with its rules
    # (an English "75,966" is 75966); the BCT pages are read with French rules.
    language = language_of(question)
    supported = set().union(*(confirmed_numbers(text, record["text"])
                              for record, text in zip(cited, supporting_text)))
    # A number the user asked about ("les billets de 100 et 500 couronnes") may be repeated when
    # the cited page mentions it, even if the quoted sentence does not. Any other number must be
    # in the quote.
    echoed = numeric_literals(question, language) & set().union(*(numeric_literals(record["text"]) for record in cited))
    # Dates from the user's own situation ("avant le 26 mars 2026") may be repeated without a quote.
    scenario = question_scenario_numbers(question)
    claim_literals = strip_instrument_references(claim_literals, cited)
    # "Selon le tableau 4-1" or "article 5": a table, article or page the cited page really has
    # is a reference, not a fact that needs a quote.
    claim_literals = strip_structure_references(claim_literals, cited)
    claim_numbers = (
        numeric_literals(claim_literals, language)
        - trusted_years(question, cited)
        - echoed
        - scenario
    )
    if claim_numbers - supported:
        raise ValueError("unsupported_claim_number")
    # A page with a misread header (see evidence_warning) cannot prove any digit, even quoted
    # exactly. Numbers written in words ("sept ans", "ثلاث سنوات") may still support a claim.
    if any(record.get("evidence_warning") for record in cited):
        spelled = set().union(*(supported_numbers(text) - numeric_literals(text) for text in supporting_text))
        # Spelling a garbled digit out in words is still stating it.
        claim_words = supported_numbers(claim_literals, language) - numeric_literals(claim_literals, language) - {"0", "1"}
        if (claim_numbers | claim_words) - spelled:
            raise ValueError("digits_unreliable_on_warned_page")
    # The code adds the [n] markers itself; a claim with its own markers, a filename or "..." is broken.
    if re.search(r"\[\d+\]|\.pdf\b|…|\.\.\.", claim_literals, re.I):
        raise ValueError("claim_contains_citation_or_truncation")
    _reject_regime_remapped_claim(question, claim_literals, cited, supporting_text)
    _reject_reversed_legal_polarity(claim_literals, cited, supporting_text)
    _reject_invented_unit(claim_literals, supporting_text)
    _reject_dropped_condition(claim_literals, cited, supporting_text)
    _reject_scope_or_operator_inflation(claim_literals, supporting_text)
    _reject_threshold_boundary(claim_literals, supporting_text, language)
    return claim.text.strip() + " " + " ".join(f"[{n}]" for n in dict.fromkeys(numbers))


_CITE_ID = re.compile(r"^\[?(E\d+)\.(\d+)\]?$")


def _cited_excerpts(cites, by_id):
    """Turn unit IDs ("E2.14") into (evidence, exact page text). Consecutive units of one evidence
    are joined into one excerpt. An ID that does not exist fails the claim."""
    wanted = {}
    for cite in cites:
        match = _CITE_ID.match(str(cite).strip())
        if not match or match.group(1) not in by_id:
            raise ValueError("unknown_citation")
        wanted.setdefault(match.group(1), set()).add(int(match.group(2)) - 1)
    excerpts = []
    for evidence_id, indices in wanted.items():
        record = by_id[evidence_id]
        text = record.get("text") or ""
        spans = unit_spans(text)
        if any(not 0 <= i < len(spans) for i in indices):
            raise ValueError("unknown_citation")
        run = []
        for i in sorted(indices) + [None]:
            if run and (i is None or i != run[-1] + 1):
                excerpts.append((record, text[spans[run[0]][0]:spans[run[-1]][1]]))
                run = []
            if i is not None:
                run.append(i)
    return excerpts


def _token_script(token: str) -> str:
    """"ar" for an Arabic word, "lat" for a Latin-script one."""
    return "ar" if ARABIC.search(token) else "lat"


# "Polarity": does a text include someone in a rule (doivent, sont soumis, تخضع) or exclude them
# (sont exclus, ne s'applique pas, تستثنى)? The _SUPPORT_ patterns read the cited text, the
# _CLAIM_ patterns read the claim; the two must agree. Exclusion is checked first, because
# "ne s'applique pas" also contains "s'applique". The Arabic parts do not use \\b (word
# boundaries are unreliable on Arabic script).
_SUPPORT_EXCLUSION = re.compile(
    r"(?:"
    r"\b(?:sont\s+exclu(?:e|es|s)?|est\s+exclue|"
    r"exclu(?:e|s|es)?\s+du\s+champ|"
    r"hors\s+(?:du\s+)?champ\s+d['']application|"
    r"ne\s+s[''](?:applique|appliquent)\s+pas|"
    r"ne\s+sont\s+pas\s+(?:soumis(?:e|es)?|concern[eé]e?s?|vis[eé]e?s?|applicables?)|"
    r"exempt[eé]e?s?|dispens[eé]e?s?|"
    r"sauf\s+(?:les|lorsque|si|en)|"
    r"sous\s+r[eé]serve)\b"
    r"|"
    r"تستثن[اىي]|يستثن[اىي]|مستثن[اىي]|استثناء|"
    r"لا\s*تنطبق|لا\s*ينطبق|لا\s*تطبق|لا\s*يطبق|لا\s*تسري|لا\s*يسري|"
    r"لا\s*تخضع|لا\s*يخضع|"
    r"خارج\s*(?:نطاق|مجال)\s*(?:التطبيق)?|"
    r"باستثناء|"
    r"مع\s*مراعاة|شريطة|بشرط"
    r")",
    re.I,
)
_SUPPORT_INCLUSION = re.compile(
    r"(?:"
    r"\b(?:obligation\s+s['']impose|doivent|doit|"
    r"sont\s+tenus?|est\s+tenu|"
    r"sont\s+soumis(?:e|es)?|est\s+soumise|"
    r"soumises?\s+[àa]\s+l['']autorisation|"
    r"s[''](?:applique|appliquent)\b(?!\s+pas))"
    r"|"
    r"يتعين\s*على|يجب\s*على|"
    r"(?<!لا)(?<!لا\s)تخضع|(?<!لا)(?<!لا\s)يخضع|"
    r"(?<!لا)(?<!لا\s)تنطبق|(?<!لا)(?<!لا\s)ينطبق|"
    r"(?<!لا)(?<!لا\s)تسري|(?<!لا)(?<!لا\s)يسري|"
    r"ملزم(?:ة)?|واجبة?"
    r")",
    re.I,
)
_CLAIM_EXCLUSION = re.compile(
    r"(?:"
    r"\b(?:ne\s+s[''](?:applique|appliquent)\s+pas|"
    r"ne\s+sont\s+pas\s+(?:concern[eé]e?s?|soumis(?:e|es)?|vis[eé]e?s?|applicables?)|"
    r"n['']est\s+pas\s+(?:concern[eé]e|soumis(?:e)?|vis[eé]e|applicable)|"
    r"sont\s+exclu(?:e|es|s)?|est\s+exclue|hors\s+champ|exempt[eé]e?s?|"
    r"librement|sont\s+libres|est\s+libre|"
    r"sans\s+autorisation|"
    r"ne\s+n[eé]cessitent\s+pas\s+d['']autorisation|"
    r"ne\s+n[eé]cessite\s+pas\s+d['']autorisation)\b"
    r"|"
    r"تستثن[اىي]|يستثن[اىي]|مستثن[اىي]|"
    r"لا\s*تنطبق|لا\s*ينطبق|لا\s*تطبق|لا\s*يطبق|لا\s*تسري|لا\s*يسري|"
    r"لا\s*تخضع|لا\s*يخضع|"
    r"خارج\s*(?:نطاق|مجال)\s*(?:التطبيق)?"
    r")",
    re.I,
)
_CLAIM_INCLUSION = re.compile(
    r"(?:"
    r"\b(?:sont\s+concern[eé]e?s?|est\s+concern[eé]e|"
    r"sont\s+soumis(?:e|es)?|est\s+soumise|"
    r"sont\s+(?:inclus(?:e|es)?|inclues?)|est\s+(?:inclus(?:e)?|inclue)|"
    r"inclus(?:e|es)?\s+dans\s+le\s+champ|"
    r"sont\s+vis[eé]e?s?|est\s+vis[eé]e|"
    r"s[''](?:applique|appliquent)\b(?!\s+pas)|"
    r"doivent|doit|obligatoirement|"
    r"sont\s+applicables?|est\s+applicable|"
    r"aucune\s+exclusion|pas\s+d['']exclusion|"
    r"ne\s+sont\s+pas\s+exclues?|n['']est\s+pas\s+exclue)\b"
    r"|"
    r"يتعين\s*على|يجب\s*على|"
    r"(?<!لا)(?<!لا\s)تخضع|(?<!لا)(?<!لا\s)يخضع|"
    r"(?<!لا)(?<!لا\s)تنطبق|(?<!لا)(?<!لا\s)ينطبق|"
    r"(?<!لا)(?<!لا\s)تسري|(?<!لا)(?<!لا\s)يسري|"
    r"المعني(?:ة|ين|ات)?|"
    r"ملزم(?:ة)?|واجبة?"
    r")",
    re.I,
)

# Claims that say "no exclusion / no other condition": inclusions too, and risky ones.
_CLAIM_DENIES_EXCLUSION = re.compile(
    r"(?i)\b(?:aucune\s+exclusion|pas\s+d['']exclusion|"
    r"ne\s+sont\s+pas\s+exclues?|n['']est\s+pas\s+exclue|"
    r"sans\s+(?:aucune\s+)?(?:autre\s+)?condition|"
    r"sans\s+autre\s+condition)\b"
)
# "all importers", "exclusively": a claim widening a rule that the text limits to a narrow group.
_CLAIM_UNIVERSAL_SCOPE = re.compile(
    r"(?i)\b(?:tous\s+les\s+importateurs|toutes\s+les\s+importations|"
    r"tout(?:e)?\s+importation|tous\s+les\s+contrats|"
    r"commerce\s+ext[eé]rieur\s+mondiaux?|sans\s+aucune\s+limite|"
    r"exclusivement|un\s+seul\s+moyen\s+impos)\b"
)
_SUPPORT_NARROW_POPULATION = re.compile(
    r"(?i)\b(?:entreprises?\s+industrielles?|produits?\s+non[- ]?prioritaires?|"
    r"marches?\s+publics?|collectivit[eé]s?\s+locales?|"
    r"n['']importe\s+quel\s+moyen|peuvent\s+[eê]tre\s+r[eé]gl[eé]s|"
    r"jusqu['’]?[àa]\s+\d+\s+jours)\b"
)
# "business days" when the text only says "days": a different deadline.
_INVENTED_DAY_UNIT = re.compile(
    r"(?i)\bjours?\s+ouvrables?\b|\bbusiness\s+days?\b|\bأيام\s*عمل\b"
)
_SUPPORT_PLAIN_DAYS = re.compile(
    r"(?i)\b(?:\d+\s*)?jours?\b(?!\s+ouvrables?)|\b(?:\d+\s*)?days?\b(?!\s+business)"
)
# "must" in the claim where the text only says "may".
_CLAIM_MUST = re.compile(r"(?i)\b(?:doivent|doit|obligatoirement|exclusivement)\b")
_SUPPORT_MAY = re.compile(
    r"(?i)\b(?:peuvent|peut|facultatif(?:ve)?|n['']importe\s+quel)\b"
)
# A condition on the page ("sous réserve", "شريطة") that the claim must not drop.
_PAGE_HAS_RESERVE = re.compile(
    r"(?i)\bsous\s+r[eé]serve\b|شريطة|بشرط|مع\s*مراعاة"
)
_CLAIM_DROPS_RESERVE = re.compile(
    r"(?i)\b(?:sans\s+(?:aucune\s+)?(?:autre\s+)?condition|"
    r"sans\s+r[eé]serve|automatiquement|"
    r"toute\s+importation\s+industrielle\s+est\s+exclue)\b"
)


def _excerpt_context_window(page_text: str, excerpt: str, *, before: int = 400, after: int = 120) -> str:
    """The page text around an excerpt: a quoted list item often depends on the sentence above
    the list ("Sont exclus du champ d'application : ...")."""
    page = page_text or ""
    needle = excerpt or ""
    if not page or not needle:
        return needle
    pos = page.find(needle)
    if pos < 0:
        collapsed_page = " ".join(page.split())
        collapsed_excerpt = " ".join(needle.split())
        pos = collapsed_page.find(collapsed_excerpt)
        if pos < 0:
            return needle
        start = max(0, pos - before)
        return collapsed_page[start : pos + len(collapsed_excerpt) + after]
    start = max(0, pos - before)
    return page[start : pos + len(needle) + after]


def _support_polarity(cited_records, supporting_excerpts) -> str | None:
    """Does the cited text include or exclude: "include", "exclude", or None if it says neither.

    Words in the excerpt itself count first, then the lines just around it, then a wider window:
    a page often states a general rule first and lists the exclusions after it.
    """
    for record, excerpt in zip(cited_records, supporting_excerpts):
        excerpt_text = excerpt or ""
        if _SUPPORT_EXCLUSION.search(excerpt_text):
            return "exclude"
        if _SUPPORT_INCLUSION.search(excerpt_text):
            return "include"
        immediate = _excerpt_context_window(
            record.get("text") or "", excerpt_text, before=120, after=60
        )
        if _SUPPORT_EXCLUSION.search(immediate):
            return "exclude"
        if _SUPPORT_INCLUSION.search(immediate):
            return "include"
    windows = []
    for record, excerpt in zip(cited_records, supporting_excerpts):
        windows.append(_excerpt_context_window(record.get("text") or "", excerpt))
    blob = " ".join(windows)
    has_ex = bool(_SUPPORT_EXCLUSION.search(blob))
    has_in = bool(_SUPPORT_INCLUSION.search(blob))
    if has_ex and not has_in:
        return "exclude"
    if has_in and not has_ex:
        return "include"
    if has_ex and has_in:
        return "exclude"
    return None


def _claim_polarity(claim_text: str) -> str | None:
    """Does the claim include or exclude: "include", "exclude", or None."""
    text = claim_text or ""
    # Strip negated applicability first so "ne s'appliquent pas" is not also inclusion.
    if _CLAIM_EXCLUSION.search(text):
        return "exclude"
    if _CLAIM_DENIES_EXCLUSION.search(text) or _CLAIM_INCLUSION.search(text):
        return "include"
    return None


_CLAIM_FIELD_EXCLUSION = re.compile(
    r"(?i)\b(?:sont\s+exclu(?:e|es|s)?|est\s+exclue|"
    r"exclu(?:e|s|es)?\s+du\s+champ|hors\s+champ|exempt[eé]e?s?|"
    r"ne\s+s[''](?:applique|appliquent)\s+pas)\b"
)


def _reject_reversed_legal_polarity(claim_literals, cited, supporting_excerpts) -> None:
    """Reject a claim that turns an exclusion into an obligation, or the reverse."""
    support = _support_polarity(cited, supporting_excerpts)
    claim = _claim_polarity(claim_literals)
    if support and claim and support != claim:
        raise ValueError("unsupported_claim_polarity")
    # Field-exclusion claims need an exclusion operator in the excerpt/lead-in.
    # (Do not apply this to librement/sans autorisation settlement wording.)
    if _CLAIM_FIELD_EXCLUSION.search(claim_literals) and support != "exclude":
        raise ValueError("unsupported_claim_polarity")


def _reject_invented_unit(claim_literals: str, supporting_excerpts) -> None:
    """Reject "business days" when the quoted text only says "days"."""
    if not _INVENTED_DAY_UNIT.search(claim_literals):
        return
    blob = " ".join(supporting_excerpts)
    if _INVENTED_DAY_UNIT.search(blob):
        return
    if _SUPPORT_PLAIN_DAYS.search(blob) or supported_numbers(blob):
        raise ValueError("unsupported_claim_unit")


def _reject_dropped_condition(claim_literals: str, cited, supporting_excerpts) -> None:
    """Reject a claim that says "without condition" when the page sets one ("sous réserve")."""
    if not _CLAIM_DROPS_RESERVE.search(claim_literals):
        return
    page = " ".join(record.get("text") or "" for record in cited)
    excerpts = " ".join(supporting_excerpts)
    if _PAGE_HAS_RESERVE.search(page) and not _PAGE_HAS_RESERVE.search(claim_literals):
        # Excerpt may omit the reserve clause; the page still requires it.
        if not _PAGE_HAS_RESERVE.search(excerpts) or _CLAIM_DROPS_RESERVE.search(claim_literals):
            raise ValueError("unsupported_claim_condition")


def _reject_scope_or_operator_inflation(claim_literals: str, supporting_excerpts) -> None:
    """Reject a claim that widens the text: "all" for a narrow group, or "must" for "may"."""
    blob = " ".join(supporting_excerpts)
    if _CLAIM_UNIVERSAL_SCOPE.search(claim_literals) and _SUPPORT_NARROW_POPULATION.search(
        blob
    ):
        if not _CLAIM_UNIVERSAL_SCOPE.search(blob):
            raise ValueError("unsupported_claim_scope")
    if _CLAIM_MUST.search(claim_literals) and _SUPPORT_MAY.search(blob):
        if not _CLAIM_MUST.search(blob):
            raise ValueError("unsupported_claim_operator")


def _reject_threshold_boundary(claim_literals: str, supporting_excerpts, language="fr") -> None:
    """Reject a deadline off by one ("91 jours" when the text says 90): a classic slip."""
    excerpt_nums = set().union(*(supported_numbers(text) for text in supporting_excerpts))
    claim_nums = numeric_literals(claim_literals, language)
    dayish = {
        token
        for token in claim_nums
        if token.isdigit() and 30 <= int(token) <= 400
    }
    for token in dayish:
        value = int(token)
        if token in excerpt_nums:
            continue
        neighbors = {str(value - 1), str(value + 1)} & excerpt_nums
        if neighbors:
            raise ValueError("unsupported_claim_number")


def _anchor_on_page(anchor: str, support_plain: str) -> bool:
    """Is this question word on the page, allowing small variations ("billet" / "billets")?"""
    # "abroge" must match "abrogée": fold accents before the prefix stem compare.
    anchor, support_plain = _ascii_fold(anchor), _ascii_fold(support_plain)
    if re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", support_plain):
        return True
    # Plural / light stemming: "billet" ↔ "billets".
    tokens = re.findall(
        r"[0-9A-Za-zÀ-ÖØ-öø-ÿ’']+|[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+",
        support_plain,
    )
    for token in tokens:
        token = token.replace("’", "'").strip("'").casefold()
        if len(token) < 5:
            continue
        if token.startswith(anchor) or anchor.startswith(token):
            return True
    return False


def _reject_regime_remapped_claim(question, claim_literals, cited, supporting_excerpts) -> None:
    """Reject a claim that pins the question's subject on a text about something else.

    Example: asked about export companies, the model quotes a rule for banks and writes "export
    companies must ...". The subject words of the claim must appear on the cited page; if they do
    not, and the quoted excerpt is clearly about another subject, the claim is rejected.

    The subject is looked for on the whole page (a table row often lacks the topic word), while
    "another subject" is judged on the excerpt only. Words can only be compared within one
    language: an English question answered from a French page shares no words ("growth" vs
    "croissance"), so such pages are skipped.
    """
    cited = [record for record in cited if record.get("language", "fr") == language_of(question)]
    if not cited:
        return
    page_plain = _plain(" ".join(record["text"] for record in cited)).casefold()
    excerpt_plain = _plain(" ".join(supporting_excerpts)).casefold()
    claim_plain = claim_literals.casefold()
    question_plain = _plain(question).casefold()
    anchors = [anchor for anchor in _question_anchors(question) if len(anchor) >= 5]
    # On-topic page: shared scenario anchors mean other question-only framing
    # words in the claim are not evidence of regime remapping.
    # ponytail: one shared token ≥5 chars; tighten if remapping false-negatives appear.
    if any(_anchor_on_page(anchor, page_plain) for anchor in anchors):
        return
    missing = []
    for anchor in anchors:
        if anchor not in claim_plain:
            continue
        if not _anchor_on_page(anchor, page_plain):
            missing.append(anchor)
    if not missing:
        return
    page_only = []
    for token in re.findall(
        r"[0-9A-Za-zÀ-ÖØ-öø-ÿ’']+|[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+",
        excerpt_plain,
    ):
        token = token.replace("’", "'").strip("'")
        if len(token) < 5 or token in question_plain:
            continue
        if _anchor_on_page(token, question_plain):
            continue
        if token.isdigit() or re.fullmatch(r"cir[_\-.]?\d+.*", token):
            continue
        page_only.append(token)
    page_only = [token for token in page_only if len(token) >= 6]
    if len(page_only) < 2:
        return
    if not any(
        _token_script(miss) == _token_script(page)
        for miss in missing
        for page in page_only
    ):
        return
    raise ValueError("unsupported_claim_anchor")


def parse_answer(content, question, evidence, *, temporal_unverified=None, diagnostics=None, query_class=None):
    """Turn the answer model's reply into the final answer: {"status", "answer", "sources"}.

    Each claim is checked on its own; failing claims are dropped. If some were dropped, the status
    becomes partial_answer with a notice. If nothing survives, or the reply is malformed, the
    result is the fixed insufficient_evidence reply (and the reason goes into diagnostics).
    """
    try:
        if temporal_unverified is None:
            temporal_unverified = is_temporal_rule_query(question)
        draft = AnswerDraft.model_validate(_load_answer_payload(content))
        if draft.status not in ("answered", "partial_answer"):
            if draft.claims:
                raise ValueError("Non-answer contains claims")
            return safe_response(question, draft.status)
        if not draft.claims:
            raise ValueError("Answer contains no supported claims")
        by_id = {record["evidence_id"]: record for record in evidence}
        if len(by_id) != len(evidence):
            raise ValueError("duplicate_evidence_id")
        target = direct_identity(question)
        sources, source_numbers, source_excerpts, lines = [], {}, {}, []
        dropped = []
        secondary = set()
        for claim in draft.claims:
            # Snapshot citation state so a rejected claim cannot leave orphan sources.
            snap_sources = list(sources)
            snap_numbers = dict(source_numbers)
            snap_excerpts = {key: list(value) for key, value in source_excerpts.items()}
            try:
                lines.append(_validate_claim(
                    claim, question, by_id,
                    temporal_unverified=temporal_unverified,
                    target=target,
                    sources=sources,
                    source_numbers=source_numbers,
                    source_excerpts=source_excerpts,
                    query_class=query_class,
                    secondary=secondary,
                ))
            except (ValueError, TypeError, KeyError) as error:
                sources[:] = snap_sources
                source_numbers.clear()
                source_numbers.update(snap_numbers)
                source_excerpts.clear()
                source_excerpts.update(snap_excerpts)
                dropped.append(str(error).strip() or type(error).__name__)
        if not lines:
            reason = dropped[0] if len(dropped) == 1 else ("no_supported_claims:" + ",".join(dict.fromkeys(dropped)) if dropped else "Answer contains no supported claims")
            raise ValueError(reason)
        status = draft.status
        # Dropped claims, or evidence that is not a regulatory text, make the answer partial. A date
        # or "in force" question is always partial, with its own notice at the top.
        if dropped or secondary:
            status = "partial_answer"
        if temporal_unverified:
            status = "partial_answer"
        if temporal_unverified:
            limits = _TEMPORAL_LIMITS if is_temporal_rule_query(question) and not _TEMPORAL_SCOPE_HINT.search(question) else _HISTORICAL_LIMITS
            lines.insert(0, limits[language_of(question)])
        if dropped and not temporal_unverified:
            # The model's free "message" field is never shown: it is not checked against any quote.
            lines.append(_PARTIAL_LIMITS[language_of(question)])
        if secondary:
            lines.append(_SECONDARY_SOURCE_NOTICE[language_of(question)])
        return {"status": status, "answer": "\n\n".join(lines), "sources": sources}
    except (ValidationError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        reason = "schema_invalid" if isinstance(error, (ValidationError, json.JSONDecodeError)) else str(error)
        if diagnostics is not None:
            diagnostics.append(reason)
        logger.info("answer_validation_rejected reason=%s", reason)
        return safe_response(question)


# The answer format shown to the model: a short example instead of the full schema (fewer tokens,
# and reasoning models gave more empty drafts with the full one). Validation still uses AnswerDraft.
ANSWER_SCHEMA_FOR_PROMPT = (
    '{"status":"answered|partial_answer|insufficient_evidence|clarification_needed|out_of_scope",'
    '"message":"","claims":[{"text":"string","cites":["E1.3","E1.4"]}]}'
)

# A question about a past date or period ("avant", "en 2020"): parse_answer then uses the
# "applicability at that date" notice instead of the "currently in force" one.
_TEMPORAL_SCOPE_HINT = re.compile(
    r"\b(?:as\s+of|before|after|au\s+\d{1,2}(?:er)?|avant|après|en\s+\d{4})\b|قبل|بعد",
    re.I,
)
# The fixed notices added to answers, in the three languages.
_HISTORICAL_LIMITS = {
    "fr": "Les passages cités étayent les éléments ci-dessous, mais leur applicabilité à la date demandée n’a pas pu être pleinement confirmée.",
    "ar": "تدعم المقاطع المستشهد بها المعلومات التالية، لكن تعذر تأكيد انطباقها في التاريخ المطلوب بشكل كامل.",
    "en": "The cited passages support the following information, but applicability at the requested date could not be fully confirmed.",
}
_SECONDARY_SOURCE_NOTICE = {
    "fr": "Cette réponse s’appuie sur une publication statistique ou une note interne, pas sur un texte réglementaire. Vérifiez la circulaire ou la note applicable.",
    "ar": "تستند هذه الإجابة إلى نشرية إحصائية أو مذكرة داخلية وليس إلى نص ترتيبي. يرجى التثبت من المنشور أو المذكرة المنطبقة.",
    "en": "This answer relies on a statistical publication or internal memo, not a regulatory text. Check the applicable circular or note.",
}
_PARTIAL_LIMITS = {
    "fr": "Ces passages ne permettent de répondre qu’à une partie de la demande. Veuillez préciser le point restant ou vérifier le document original.",
    "ar": "تتيح هذه المقاطع الإجابة عن جزء من الطلب فقط. يرجى توضيح النقطة المتبقية أو مراجعة الوثيقة الأصلية.",
    "en": "These passages support only part of the request. Please clarify the remaining point or check the original document.",
}

_TEMPORAL_LIMITS = {
    "fr": (
        "Voici la dernière valeur étayée par les passages cités, distincte d’une "
        "confirmation qu’elle est actuellement en vigueur. La validité présente "
        "de ces dispositions n’a pas pu être pleinement confirmée."
    ),
    "ar": (
        "فيما يلي آخر قيمة تدعمها المقاطع المستشهد بها، وهي ليست تأكيداً بأنها "
        "سارية حالياً. تعذر تأكيد الصلاحية الحالية لهذه الأحكام بشكل كامل."
    ),
    "en": (
        "This is the latest value supported by the cited passages, not a "
        "confirmation that it is currently in force. Present validity could "
        "not be fully confirmed."
    ),
}

def _question_anchors(question):
    """The question's meaningful words (stop words like "le", "what", "في" removed), plus simple
    variants (without a plural s, without the Arabic article "ال")."""
    stop = {
        "est", "ce", "que", "qui", "quoi", "dont", "les", "des", "une", "un", "le", "la",
        "du", "de", "d", "et", "ou", "au", "aux", "en", "y", "a", "à", "pour", "par", "sur",
        "dans", "avec", "sans", "sous", "chez", "vers", "plus", "moins", "très", "tres",
        "sont", "été", "ete", "être", "etre", "avoir", "fait", "faire", "peut",
        "peuvent", "doit", "doivent", "quel", "quelle", "quels", "quelles", "comment",
        "pourquoi", "quand", "où", "si", "ne", "pas", "tout", "tous", "toute",
        "toutes", "cette", "cet", "ces", "son", "sa", "ses", "leur", "leurs", "mon", "ma",
        "mes", "ton", "ta", "tes", "nos", "votre", "vos", "the", "an", "of", "to",
        "in", "on", "for", "is", "are", "was", "were", "be", "do", "does", "did", "what",
        "which", "who", "when", "where", "why", "how", "can", "could", "would", "should",
        "please", "tell", "me", "about", "from", "with", "without", "into", "over",
        "était", "etait", "étaient", "etaient", "restaient", "restait", "étaient-ils",
        "aujourd", "aujourdhui", "aujourd'hui", "vigueur", "actuel", "actuelle", "actuellement",
        "current", "today", "latest", "plafond", "taux", "durée", "duree", "montant", "limite",
        "délai", "delai", "conditions", "condition", "règles", "regles", "règle", "regle",
        "obligation", "obligations", "autorisation", "autorisations",
        "circulaires", "circulaire", "notes", "note", "documents", "document",
        "mentionnent", "mentionne", "mentionner", "parlent", "parle", "exister", "existe",
        "il", "elle", "nous", "vous", "ils", "elles",
        "هل", "ما", "ماذا", "من", "في", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه",
        "ذلك", "تلك", "التي", "الذي", "اللذان", "اللتان", "الذين", "اللواتي", "كان",
        "كانت", "يكون", "تكون", "أن", "ان", "إن", "لا", "لم", "لن", "قد", "كل", "بعض",
        "أي", "او", "أو", "و", "ف", "ب", "ك", "ل", "ال", "هو", "هي", "هم", "هن", "نحن",
        "أنتم", "أنتن", "كيف", "متى", "أين", "اين", "لماذا", "كم", "هناك", "هنا",
        "المنشور", "المنشورات", "المذكرة", "المذكرات", "الوثيقة", "الوثائق",
        "تذكر", "يذكر", "تذكرون", "يذكرون", "توجد", "يوجد", "بشأن", "حول",
    }
    token_re = re.compile(
        r"[0-9A-Za-zÀ-ÖØ-öø-ÿ’']+|[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+"
    )
    tokens = token_re.findall(_plain(question).casefold())
    anchors = []
    for token in tokens:
        token = token.replace("’", "'").strip("'")
        token = token.strip("؟?!.،,;؛:\"'«»…")
        if len(token) < 2 or token in stop:
            continue
        if token not in anchors:
            anchors.append(token)
        if len(token) > 3 and token.endswith("s") and not ARABIC.search(token):
            stem = token[:-1]
            if stem not in stop and stem not in anchors:
                anchors.append(stem)
        if ARABIC.search(token) and token.startswith("ال") and len(token) > 3:
            bare = token[2:]
            if bare not in stop and bare not in anchors:
                anchors.append(bare)
    return anchors


