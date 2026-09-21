"""Claim gates: fail-closed validation of grounded answer drafts.

Literal/quote checks do not prove semantic entailment or legal correctness.
"""
import json
import re
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from retrieval_selection import ARABIC, parse_source_identity
from source_metadata import normalize_page
from graph_contract import is_temporal_rule_query
from answer_evidence import (
    plain as _plain, source_quote, numeric_literals, supported_numbers, direct_identity,
    identity_matches, evidence_problem, evidence_warning, trusted_years, strip_instrument_references,
    claim_asserts_unverified_applicability, question_scenario_numbers,
)

logger = logging.getLogger(__name__)

class Quote(BaseModel):
    # Ignore stray model fields; literal quote/number gates remain the real control.
    model_config = ConfigDict(extra="ignore")
    evidence_id: str = Field(min_length=1, max_length=16)
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(min_length=1, max_length=1600)
    quotes: list[Quote] = Field(min_length=1, max_length=5)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["answered", "partial_answer", "insufficient_evidence", "clarification_needed", "out_of_scope"]
    message: str = Field(default="", max_length=600)
    claims: list[Claim] = Field(default_factory=list, max_length=8)


def _extract_complete_json_dicts(text, *, require_keys=()):
    """Pull complete {...} objects from possibly truncated model output."""
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
    """Recover status + complete claims from truncated answer JSON (no LLM)."""
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
    claims = _extract_complete_json_dicts(text[claims_at:], require_keys=("text", "quotes"))
    if not claims:
        return None
    if status not in {"answered", "partial_answer"}:
        status = "partial_answer"
    return {"status": status, "message": "", "claims": claims[:8]}


def _load_answer_payload(content):
    """Accept raw JSON or common chat wrappers; never invent claim fields."""
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
    # Repair sometimes echoes the full JSON Schema beside the answer fields.
    if "status" not in payload and "claims" not in payload:
        salvaged = _salvage_answer_payload(text)
        if salvaged is not None:
            return salvaged
    return payload


def language_of(question):
    if ARABIC.search(question):
        return "ar"
    # This is intentionally a tiny deterministic router rather than another
    # model call. Cover the common imperative/query starters used in the UI so
    # English safe fallbacks are not accidentally rendered in French.
    if re.search(
        r"^\s*(?:what|which|who|when|where|how|why|please|tell|hello|can|could|would|"
        r"is|are|was|were|does|do|did|give|show|find|explain|describe|compare|list|"
        r"summarize|summarise|identify|check|has|have|should|may|must)\b",
        question,
        re.I,
    ):
        return "en"
    return "fr"


_MESSAGES = {
    "fr": {
        "insufficient_evidence": "Les passages disponibles ne permettent pas une réponse suffisamment étayée. Veuillez vérifier le document original.",
        "clarification_needed": "Veuillez préciser la circulaire, la note ou le type d’opération concerné.",
        "out_of_scope": "Je peux vous aider à rechercher des informations dans les documents réglementaires de la BCT.",
    },
    "ar": {
        "insufficient_evidence": "المقاطع المتاحة لا تكفي لتقديم إجابة موثقة. يرجى مراجعة الوثيقة الأصلية.",
        "clarification_needed": "يرجى تحديد المنشور أو المذكرة أو نوع العملية المقصودة.",
        "out_of_scope": "يمكنني مساعدتك في البحث عن المعلومات في الوثائق التنظيمية للبنك المركزي التونسي.",
    },
    "en": {
        "insufficient_evidence": "The available passages do not sufficiently support an answer. Please check the original document.",
        "clarification_needed": "Please specify which circular, note, or type of operation you mean.",
        "out_of_scope": "I can help you find information in BCT regulatory documents.",
    },
}


def safe_response(question, status="insufficient_evidence"):
    return {"status": status, "answer": _MESSAGES[language_of(question)][status], "sources": []}


def _validate_claim(claim, question, by_id, *, temporal_unverified, target, sources, source_numbers, source_quotes):
    """Validate one claim. Raises ValueError/KeyError on literal or identity failure."""
    if not claim.text.strip():
        raise ValueError("Empty claim")
    if temporal_unverified and claim_asserts_unverified_applicability(claim.text):
        raise ValueError("unverified_applicability_claim_use_document_scoped_wording")
    numbers = []
    supporting_text = []
    claim_literals = _plain(claim.text)
    for quote in claim.quotes:
        record = by_id[quote.evidence_id]
        if target and not identity_matches(record["source"], target):
            raise ValueError("requested_document_mismatch")
        original_quote = source_quote(quote.quote, record["text"])
        problem = record.get("unusable_reason") or evidence_problem(record)
        if problem:
            raise ValueError(problem)
        supporting_text.append(original_quote)
        # A literal name of the cited document is trusted metadata, not
        # a numeric rule that must also occur in the extracted quote.
        stem = re.escape(record["source"].removesuffix(".pdf"))
        claim_literals = re.sub(r"(?<!\w)" + stem + r"(?:\.pdf)?(?!\w)", "", claim_literals, flags=re.I)
        key = record["source"], record["page"]
        if key not in source_numbers:
            source_numbers[key] = len(sources) + 1
            sources.append({"file": record["source"], "page": record["page"],
                            "score": record["score"], "excerpt": original_quote})
            source_quotes[key] = [original_quote]
        elif _plain(quote.quote) not in {_plain(text) for text in source_quotes[key]}:
            source_quotes[key].append(original_quote)
            sources[source_numbers[key] - 1]["excerpt"] = "\n…\n".join(source_quotes[key])
        numbers.append(source_numbers[key])
    supported = set().union(*(supported_numbers(text) for text in supporting_text))
    cited = [by_id[quote.evidence_id] for quote in claim.quotes]
    # A number the user asked about ("les billets de 100 et 500 couronnes")
    # may be restated when the cited page itself mentions it, even if the
    # supporting quote is only the answering sentence. Numbers absent from
    # the page, or not asked about, must be in the quote.
    echoed = numeric_literals(question) & set().union(*(numeric_literals(record["text"]) for record in cited))
    # Scenario framing from the question (e.g. "26 mars 2026") may be restated
    # without appearing in the quotation; the legal consequence still needs quotes.
    scenario = question_scenario_numbers(question)
    claim_literals = strip_instrument_references(claim_literals, cited)
    claim_numbers = (
        numeric_literals(claim_literals)
        - trusted_years(question, cited)
        - echoed
        - scenario
    )
    if claim_numbers - supported:
        raise ValueError("unsupported_claim_number")
    # A garbled header (reversed or impossible digits) means this page's
    # digits cannot be trusted even when quoted verbatim. Numbers written
    # as words ("sept ans", "ثلاث سنوات") may still support a claim.
    if any(record.get("evidence_warning") for record in cited):
        spelled = set().union(*(supported_numbers(text) - numeric_literals(text) for text in supporting_text))
        # Spelling a garbled digit out in words is still stating it.
        claim_words = supported_numbers(claim_literals) - numeric_literals(claim_literals) - {"0", "1"}
        if (claim_numbers | claim_words) - spelled:
            raise ValueError("digits_unreliable_on_warned_page")
    if re.search(r"\[\d+\]|\.pdf\b|…|\.\.\.", claim_literals, re.I):
        raise ValueError("claim_contains_citation_or_truncation")
    _reject_regime_remapped_claim(question, claim_literals, cited, supporting_text)
    _reject_reversed_legal_polarity(claim_literals, cited, supporting_text)
    _reject_invented_unit(claim_literals, supporting_text)
    _reject_dropped_condition(claim_literals, cited, supporting_text)
    _reject_scope_or_operator_inflation(claim_literals, supporting_text)
    _reject_threshold_boundary(claim_literals, supporting_text)
    return claim.text.strip() + " " + " ".join(f"[{n}]" for n in dict.fromkeys(numbers))


def _token_script(token: str) -> str:
    return "ar" if ARABIC.search(token) else "lat"


# Legal applicability polarity for FR + AR circulars/notes (exclu/تستثنى ↔ concerné/تخضع).
# Arabic avoids \\b (unreliable on Arabic script). Exclusion is checked before inclusion.
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

_CLAIM_DENIES_EXCLUSION = re.compile(
    r"(?i)\b(?:aucune\s+exclusion|pas\s+d['']exclusion|"
    r"ne\s+sont\s+pas\s+exclues?|n['']est\s+pas\s+exclue|"
    r"sans\s+(?:aucune\s+)?(?:autre\s+)?condition|"
    r"sans\s+autre\s+condition)\b"
)
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
_INVENTED_DAY_UNIT = re.compile(
    r"(?i)\bjours?\s+ouvrables?\b|\bbusiness\s+days?\b|\bأيام\s*عمل\b"
)
_SUPPORT_PLAIN_DAYS = re.compile(
    r"(?i)\b(?:\d+\s*)?jours?\b(?!\s+ouvrables?)|\b(?:\d+\s*)?days?\b(?!\s+business)"
)
_CLAIM_MUST = re.compile(r"(?i)\b(?:doivent|doit|obligatoirement|exclusivement)\b")
_SUPPORT_MAY = re.compile(
    r"(?i)\b(?:peuvent|peut|facultatif(?:ve)?|n['']importe\s+quel)\b"
)
_PAGE_HAS_RESERVE = re.compile(
    r"(?i)\bsous\s+r[eé]serve\b|شريطة|بشرط|مع\s*مراعاة"
)
_CLAIM_DROPS_RESERVE = re.compile(
    r"(?i)\b(?:sans\s+(?:aucune\s+)?(?:autre\s+)?condition|"
    r"sans\s+r[eé]serve|automatiquement|"
    r"toute\s+importation\s+industrielle\s+est\s+exclue)\b"
)


def _quote_context_window(page_text: str, quote: str, *, before: int = 400, after: int = 120) -> str:
    """Page text around a quote so list-item excerpts keep their exclusion lead-in."""
    page = page_text or ""
    needle = quote or ""
    if not page or not needle:
        return needle
    pos = page.find(needle)
    if pos < 0:
        collapsed_page = " ".join(page.split())
        collapsed_quote = " ".join(needle.split())
        pos = collapsed_page.find(collapsed_quote)
        if pos < 0:
            return needle
        start = max(0, pos - before)
        return collapsed_page[start : pos + len(collapsed_quote) + after]
    start = max(0, pos - before)
    return page[start : pos + len(needle) + after]


def _support_polarity(cited_records, supporting_quotes) -> str | None:
    """Return 'exclude', 'include', or None from quote + nearby page context.

    Quote-local language wins over distant Article-premier obligations on the
    same page (exclusion lists often sit after a general rule).
    """
    for record, quote in zip(cited_records, supporting_quotes):
        quote_text = quote or ""
        if _SUPPORT_EXCLUSION.search(quote_text):
            return "exclude"
        if _SUPPORT_INCLUSION.search(quote_text):
            return "include"
        immediate = _quote_context_window(
            record.get("text") or "", quote_text, before=120, after=60
        )
        if _SUPPORT_EXCLUSION.search(immediate):
            return "exclude"
        if _SUPPORT_INCLUSION.search(immediate):
            return "include"
    windows = []
    for record, quote in zip(cited_records, supporting_quotes):
        windows.append(_quote_context_window(record.get("text") or "", quote))
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


def _reject_reversed_legal_polarity(claim_literals, cited, supporting_quotes) -> None:
    """Fail when the claim flips exclusion/exemption into inclusion/applicability (or vice versa)."""
    support = _support_polarity(cited, supporting_quotes)
    claim = _claim_polarity(claim_literals)
    if support and claim and support != claim:
        raise ValueError("unsupported_claim_polarity")
    # Field-exclusion claims need an exclusion operator in the quote/lead-in.
    # (Do not apply this to librement/sans autorisation settlement wording.)
    if _CLAIM_FIELD_EXCLUSION.search(claim_literals) and support != "exclude":
        raise ValueError("unsupported_claim_polarity")


def _reject_invented_unit(claim_literals: str, supporting_quotes) -> None:
    """Reject day-unit inventions (jours ouvrables) absent from the quotes."""
    if not _INVENTED_DAY_UNIT.search(claim_literals):
        return
    blob = " ".join(supporting_quotes)
    if _INVENTED_DAY_UNIT.search(blob):
        return
    if _SUPPORT_PLAIN_DAYS.search(blob) or supported_numbers(blob):
        raise ValueError("unsupported_claim_unit")


def _reject_dropped_condition(claim_literals: str, cited, supporting_quotes) -> None:
    """Reject claims that erase a page-level 'sous réserve' / condition."""
    if not _CLAIM_DROPS_RESERVE.search(claim_literals):
        return
    page = " ".join(record.get("text") or "" for record in cited)
    quotes = " ".join(supporting_quotes)
    if _PAGE_HAS_RESERVE.search(page) and not _PAGE_HAS_RESERVE.search(claim_literals):
        # Quote may omit the reserve clause; the page still requires it.
        if not _PAGE_HAS_RESERVE.search(quotes) or _CLAIM_DROPS_RESERVE.search(claim_literals):
            raise ValueError("unsupported_claim_condition")


def _reject_scope_or_operator_inflation(claim_literals: str, supporting_quotes) -> None:
    """Reject universal/mandatory restatements of narrow/permissive source wording."""
    blob = " ".join(supporting_quotes)
    if _CLAIM_UNIVERSAL_SCOPE.search(claim_literals) and _SUPPORT_NARROW_POPULATION.search(
        blob
    ):
        if not _CLAIM_UNIVERSAL_SCOPE.search(blob):
            raise ValueError("unsupported_claim_scope")
    if _CLAIM_MUST.search(claim_literals) and _SUPPORT_MAY.search(blob):
        if not _CLAIM_MUST.search(blob):
            raise ValueError("unsupported_claim_operator")


def _reject_threshold_boundary(claim_literals: str, supporting_quotes) -> None:
    """Reject off-by-one day thresholds when the quote only supports N, not N±1."""
    quote_nums = set().union(*(supported_numbers(text) for text in supporting_quotes))
    claim_nums = numeric_literals(claim_literals)
    dayish = {
        token
        for token in claim_nums
        if token.isdigit() and 30 <= int(token) <= 400
    }
    for token in dayish:
        value = int(token)
        if token in quote_nums:
            continue
        neighbors = {str(value - 1), str(value + 1)} & quote_nums
        if neighbors:
            raise ValueError("unsupported_claim_number")


def _anchor_on_page(anchor: str, support_plain: str) -> bool:
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


def _reject_regime_remapped_claim(question, claim_literals, cited, supporting_quotes) -> None:
    """Fail when the claim restates a question actor absent from the cited page
    while the supporting quotes are clearly about a different subject regime.

    Actor presence is checked on the full page (tables often omit the topic word
    from the numeric quote). Alternate-regime detection uses the quotes only, so
    a same-page but wrong excerpt cannot launder a remapped claim.

    Restating question-scenario framing words is allowed when the cited page
    already shares substantive anchors with the question (on-topic evidence).
    """
    page_plain = _plain(" ".join(record["text"] for record in cited)).casefold()
    quote_plain = _plain(" ".join(supporting_quotes)).casefold()
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
        quote_plain,
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


def parse_answer(content, question, evidence, *, temporal_unverified=None, diagnostics=None):
    """Fail closed on malformed output; keep any claims that pass literal gates."""
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
        sources, source_numbers, source_quotes, lines = [], {}, {}, []
        dropped = []
        for claim in draft.claims:
            # Snapshot citation state so a rejected claim cannot leave orphan sources.
            snap_sources = list(sources)
            snap_numbers = dict(source_numbers)
            snap_quotes = {key: list(value) for key, value in source_quotes.items()}
            try:
                lines.append(_validate_claim(
                    claim, question, by_id,
                    temporal_unverified=temporal_unverified,
                    target=target,
                    sources=sources,
                    source_numbers=source_numbers,
                    source_quotes=source_quotes,
                ))
            except (ValueError, TypeError, KeyError) as error:
                sources[:] = snap_sources
                source_numbers.clear()
                source_numbers.update(snap_numbers)
                source_quotes.clear()
                source_quotes.update(snap_quotes)
                dropped.append(str(error).strip() or type(error).__name__)
        if not lines:
            reason = dropped[0] if len(dropped) == 1 else ("no_supported_claims:" + ",".join(dict.fromkeys(dropped)) if dropped else "Answer contains no supported claims")
            raise ValueError(reason)
        status = draft.status
        if dropped or temporal_unverified:
            status = "partial_answer"
        if temporal_unverified:
            limits = _TEMPORAL_LIMITS if is_temporal_rule_query(question) and not _TEMPORAL_SCOPE_HINT.search(question) else _HISTORICAL_LIMITS
            lines.insert(0, limits[language_of(question)])
        if status == "partial_answer" and not temporal_unverified:
            # The free-form message is not quoted evidence. Never render a second,
            # unvalidated legal answer through this field.
            lines.append(_PARTIAL_LIMITS[language_of(question)])
        return {"status": status, "answer": "\n\n".join(lines), "sources": sources}
    except (ValidationError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        reason = "schema_invalid" if isinstance(error, (ValidationError, json.JSONDecodeError)) else str(error)
        if diagnostics is not None:
            diagnostics.append(reason)
        logger.info("answer_validation_rejected reason=%s", reason)
        return safe_response(question)


ANSWER_SCHEMA = json.dumps(AnswerDraft.model_json_schema(), ensure_ascii=False)
# Full pydantic schema in the system prompt burns tokens and, with reasoning models,
# correlates with empty drafts. Prompts use this compact shape; validation still uses AnswerDraft.
ANSWER_SCHEMA_FOR_PROMPT = (
    '{"status":"answered|partial_answer|insufficient_evidence|clarification_needed|out_of_scope",'
    '"message":"","claims":[{"text":"string","quotes":[{"evidence_id":"E1","quote":"exact substring"}]}]}'
)

# Disclaimer branch in parse_answer: as-of / after / en YYYY (broader than cutoff demotion).
_TEMPORAL_SCOPE_HINT = re.compile(
    r"\b(?:as\s+of|before|after|au\s+\d{1,2}(?:er)?|avant|après|en\s+\d{4})\b|قبل|بعد",
    re.I,
)
_HISTORICAL_LIMITS = {
    "fr": "Les passages cités étayent les éléments ci-dessous, mais leur applicabilité à la date demandée n’a pas pu être pleinement confirmée.",
    "ar": "تدعم المقاطع المستشهد بها المعلومات التالية، لكن تعذر تأكيد انطباقها في التاريخ المطلوب بشكل كامل.",
    "en": "The cited passages support the following information, but applicability at the requested date could not be fully confirmed.",
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
    """Content tokens from the question used to keep formulation snippets on-topic."""
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


