"""Executable stress cases mapped to the full failure taxonomy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stages import Stage


@dataclass(frozen=True)
class Case:
    id: str
    stage: Stage
    title: str
    kind: str
    data: dict[str, Any]


# Every taxonomy cell gets at least one executable Case (kind + data).
# IDs match the user's list (1.A … 10.*). Multiple cases may share a prefix.

CASES: list[Case] = [
    # --- 1. Retrieval ---
    Case(
        "1.A",
        Stage.RETRIEVAL,
        "Same vocabulary, different domain: import dépôt vs export crédit documentaire",
        "prefer_named_domain",
        {
            "query": (
                "Selon la circulaire 2026-04, le crédit documentaire pour l'importation "
                "de produits non prioritaires exige-t-il un dépôt de 100% ?"
            ),
            "named": ("Cir_2026_04_fr.pdf", 2),
            "distractor": ("Cir_2020_02_fr.pdf", 4),
            "named_score": 0.4,
            "distractor_score": 0.99,
            "expect_top": "Cir_2026_04_fr.pdf",
        },
    ),
    Case(
        "1.B",
        Stage.RETRIEVAL,
        "Explicit document name is a hard retrieval anchor",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, quelles sont les règles d'exportation ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2026_04_fr.pdf", 1),
            "named_score": 0.35,
            "distractor_score": 0.98,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "1.C",
        Stage.COVERAGE,
        "Correct document, wrong page — exception on adjacent page",
        "expand_adjacent",
        {
            "hit": ("Cir_2026_04_fr.pdf", 2),
            "must_include_page": 3,
            "must_contain": "entreprises industrielles",
        },
    ),
    Case(
        "1.D",
        Stage.COVERAGE,
        "Correct page, wrong chunk — full page assembly restores operator",
        "page_assembly_operator",
        {
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "must_contain": "ne peuvent mettre à la",
        },
    ),
    Case(
        "1.E",
        Stage.COVERAGE,
        "General rule retrieved, exception must also enter pack",
        "expand_adjacent",
        {
            "hit": ("Cir_2026_04_fr.pdf", 2),
            "must_include_page": 3,
            "must_contain": "Sont exclues",
        },
    ),
    Case(
        "1.F",
        Stage.COVERAGE,
        "Exception alone without general rule is incomplete for dépôt questions",
        "pack_needs_both",
        {
            "general": ("Cir_2026_04_fr.pdf", 2),
            "exception": ("Cir_2026_04_fr.pdf", 3),
            "general_must": "fonds propres",
            "exception_must": "entreprises industrielles",
        },
    ),
    Case(
        "1.G",
        Stage.COVERAGE,
        "Defined term 'non prioritaires' points to annex list not on operative pages",
        "definition_elsewhere",
        {
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "term_marker": "liste figurant en",
            "annex_pages": [4, 5, 6, 7],
        },
    ),
    Case(
        "1.H",
        Stage.GENERATION,
        "Cross-document contamination — conflicting export delay rules",
        "franken_reject",
        {
            "question": (
                "Quel est le délai de règlement libre sans condition pour les ventes "
                "à l'exportation ?"
            ),
            "sources": [
                ("Cir_2020_02_fr.pdf", 3),
                ("Cir_2025_13_fr.pdf", 2),
            ],
            "bad_claim": (
                "Les ventes peuvent être réglées librement jusqu'à 120 jours sans "
                "condition, comme sous le régime antérieur de 60 jours consolidé."
            ),
            "bad_quote_source": "Cir_2025_13_fr.pdf",
            "bad_quote": "délais de règlement allant jusqu’à 120 jours",
        },
    ),
    Case(
        "1.I",
        Stage.TEMPORAL,
        "Historical document must not outrank named current amendment for applicability",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, le délai libre sans condition est-il de 120 jours ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2020_02_fr.pdf", 3),
            "named_score": 0.3,
            "distractor_score": 0.95,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "1.J",
        Stage.RETRIEVAL,
        "Newer document that only mentions older rule must not outrank named older instrument",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, quelles sont les conditions de 121 à 360 jours ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2026_04_fr.pdf", 1),
            "named_score": 0.4,
            "distractor_score": 0.99,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "1.K",
        Stage.TEMPORAL,
        "Operative amendment language creates followable SUPERSEDES edge",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
        },
    ),
    Case(
        "1.L",
        Stage.TEMPORAL,
        "Vu citation 'telle que modifiée par' is not a successor edge",
        "extract_empty",
        {
            "source": "Cir_2026_04_fr.pdf",
            "page": 1,
        },
    ),
    Case(
        "1.M",
        Stage.VALIDATION,
        "Near-identical numeric distractor 120 vs 121 is rejected",
        "parse_reject",
        {
            "question": "Selon la circulaire 2025-13, jusqu'à combien de jours sans condition ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Les ventes peuvent être réglées librement jusqu'à 121 jours sans condition.",
            "quote": "délais de règlement allant jusqu’à 120 jours à compter de la date d’expédition",
        },
    ),
    Case(
        "1.N",
        Stage.RETRIEVAL,
        "Arabic instrument identity parses for bilingual retrieval routing",
        "identity",
        {
            "query": "ما هو موضوع المنشور عدد 6 لسنة 2026؟",
            "kind": "cir",
            "year": 2026,
            "number": 6,
        },
    ),
    Case(
        "1.O",
        Stage.COVERAGE,
        "Annex pages with empty OCR cannot supply the non-priority product list",
        "annex_empty",
        {
            "source": "Cir_2026_04_fr.pdf",
            "annex_pages": [4, 5, 6, 7],
        },
    ),
    Case(
        "1.P",
        Stage.CITATION,
        "Citation must keep filename metadata, not invent another circular",
        "citation_identity",
        {
            "question": "Selon la circulaire 2025-13, quel est le délai sans condition ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Le délai libre sans condition est de 120 jours.",
            "quote": "jusqu’à 120 jours",
            "expect_file": "Cir_2025_13_fr.pdf",
            "expect_page": 2,
            "temporal_unverified": False,
        },
    ),
    Case(
        "1.Q",
        Stage.RETRIEVAL,
        "Duplicate-looking hits still resolve to one named instrument identity",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, article 12 ?",
            "named": ("Cir_2025_13_fr.pdf", 3),
            "distractor": ("Cir_2020_02_fr.pdf", 4),
            "named_score": 0.5,
            "distractor_score": 0.9,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "1.R",
        Stage.SELECTION,
        "Ambiguous deictic 'cette obligation' without topic → clarification path",
        "identity_none",
        {
            "query": "Cette obligation s'applique-t-elle encore ?",
        },
    ),
    Case(
        "1.S",
        Stage.VALIDATION,
        "Under-specified question with empty evidence must fail closed",
        "parse_insufficient",
        {
            "question": "Quel est le plafond exact pour les voyages d'affaires en 2099 ?",
            "evidence": [],
        },
    ),
    Case(
        "1.T",
        Stage.SELECTION,
        "Overly broad query must not invent a confirmed legal answer from topical noise",
        "parse_reject_topical",
        {
            "question": "Quelles sont les règles pour le financement des importations ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 1,
            "claim": (
                "Toutes les importations tunisiennes exigent un dépôt de 100% des fonds propres."
            ),
            "quote": "produits non prioritaires",
        },
    ),
    Case(
        "1.U",
        Stage.RETRIEVAL,
        "Paraphrase still resolves explicit year-number identity",
        "identity",
        {
            "query": "Pour la circulaire BCT n° 2025-13, délai libre max sans condition ?",
            "kind": "cir",
            "year": 2025,
            "number": 13,
        },
    ),
    Case(
        "1.V",
        Stage.VALIDATION,
        "Terminology variants banque non-résidente / crédit doc must stay quote-faithful",
        "parse_accept",
        {
            "question": (
                "Selon la circulaire 2025-13, une vente à 200 jours est-elle libre si "
                "assortie d'une garantie d'une banque non résidente ?"
            ),
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Oui, une vente peut être effectuée librement si elle est assortie d'une "
                "garantie de paiement émise par une banque non-résidente."
            ),
            "quote": "garantie de paiement émise par une banque non-résidente",
            "temporal_unverified": False,
        },
    ),
    Case(
        "1.W",
        Stage.GENERATION,
        "Contradictory delay regimes must not be merged into one rule",
        "franken_reject",
        {
            "question": "Quel délai libre sans condition s'applique aux ventes à l'export ?",
            "sources": [
                ("Cir_2020_02_fr.pdf", 3),
                ("Cir_2025_13_fr.pdf", 2),
            ],
            "bad_claim": (
                "Le délai libre sans condition est de 60 jours ou 120 jours selon les textes."
            ),
            "bad_quote_source": "Cir_2025_13_fr.pdf",
            "bad_quote": "jusqu’à 120 jours",
        },
    ),
    # --- 2. Context / evidence assembly ---
    Case(
        "2.A",
        Stage.COVERAGE,
        "Relevant page retrieved but adjacent exception assembled",
        "expand_adjacent",
        {
            "hit": ("Cir_2026_04_fr.pdf", 2),
            "must_include_page": 3,
            "must_contain": "fiche technique",
        },
    ),
    Case(
        "2.B",
        Stage.COVERAGE,
        "Adjacent pages keep page-order metadata",
        "expand_order",
        {
            "hit": ("Cir_2026_04_fr.pdf", 2),
        },
    ),
    Case(
        "2.C",
        Stage.COVERAGE,
        "Full page text keeps long Article premier deposit rule",
        "page_assembly_operator",
        {
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "must_contain": "Article premier",
        },
    ),
    Case(
        "2.D",
        Stage.VALIDATION,
        "Compression that drops legal operator is rejected",
        "parse_reject",
        {
            "question": (
                "Les importations avec engagement déjà entamé avant l'entrée en vigueur "
                "de 2026-04 restent-elles soumises à l'article premier ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Oui, ces importations restent soumises aux nouvelles restrictions.",
            "quote": "les importations ayant donné lieu, préalablement à la date",
        },
    ),
    Case(
        "2.E",
        Stage.GENERATION,
        "Exception detached from lead-in polarity",
        "polarity_reject",
        {
            "question": (
                "Les importations réalisées par les entreprises industrielles sont-elles "
                "soumises à l'article premier de la circulaire 2026-04 ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "claim": (
                "Oui, les importations réalisées par les entreprises industrielles "
                "sont soumises à l'article premier."
            ),
            "quote": "Sont exclues du champ d'application",
        },
    ),
    Case(
        "2.F",
        Stage.COVERAGE,
        "Article 4 continues from p.2 onto p.3",
        "expand_adjacent",
        {
            "hit": ("Cir_2026_04_fr.pdf", 2),
            "must_include_page": 3,
            "must_contain": "industrie",
        },
    ),
    Case(
        "2.G",
        Stage.COVERAGE,
        "Bullet membership for 121-360 conditions preserved on page text",
        "page_assembly_operator",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "must_contain": "lettre de crédit stand-by",
        },
    ),
    Case(
        "2.H",
        Stage.COVERAGE,
        "Definition of non-priority products is annex-referenced, not on p.2",
        "definition_elsewhere",
        {
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "term_marker": "annexe",
            "annex_pages": [4, 5, 6, 7],
        },
    ),
    Case(
        "2.I",
        Stage.SELECTION,
        "Multiple passages need relationship — exception modifies article premier",
        "pack_needs_both",
        {
            "general": ("Cir_2026_04_fr.pdf", 2),
            "exception": ("Cir_2026_04_fr.pdf", 3),
            "general_must": "Article premier",
            "exception_must": "entreprises industrielles",
        },
    ),
    # --- 3. Temporal / supersession ---
    Case(
        "3.A",
        Stage.TEMPORAL,
        "Citation mistaken for amendment",
        "extract_empty",
        {"source": "Cir_2026_04_fr.pdf", "page": 1},
    ),
    Case(
        "3.B",
        Stage.TEMPORAL,
        "Wrong direction — 2025-13 replaces 94-14, not the reverse",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
            "expect_source_instrument": "cir:2025:13",
        },
    ),
    Case(
        "3.C",
        Stage.TEMPORAL,
        "AMEND vs REPLACE distinguished for 2025-13 operative clause",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
        },
    ),
    Case(
        "3.D",
        Stage.TEMPORAL,
        "Historical Vu recital is not operative supersession",
        "extract_empty",
        {"source": "Cir_2026_04_fr.pdf", "page": 1},
    ),
    Case(
        "3.E",
        Stage.RETRIEVAL,
        "Later document mentions earlier rule but does not govern export delays",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, autorisation préalable au-delà de 360 jours ?",
            "named": ("Cir_2025_13_fr.pdf", 3),
            "distractor": ("Cir_2026_04_fr.pdf", 1),
            "named_score": 0.45,
            "distractor_score": 0.97,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "3.F",
        Stage.VALIDATION,
        "Earlier exception incorrectly surviving — reject inventing survival without quote",
        "parse_reject",
        {
            "question": "L'exception industrielle de 2018-01 s'applique-t-elle encore sous 2026-04 ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "L'ancienne exception de 2018-01 demeure inchangée et s'applique telle quelle.",
            "quote": "entreprises industrielles",
        },
    ),
    Case(
        "3.G",
        Stage.TEMPORAL,
        "Replacement of named 94-14 articles does not invent unrelated abrogation of 2026-04",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
            "forbid_target": "cir:2026:4",
        },
    ),
    Case(
        "3.H",
        Stage.VALIDATION,
        "Effective date: 2025-13 enters into force on publication date",
        "parse_accept",
        {
            "question": "Quand la circulaire 2025-13 entre-t-elle en vigueur ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": (
                "La circulaire 2025-13 entre en vigueur à compter de la date de sa publication."
            ),
            "quote": "entre en vigueur à compter de la date de sa publication",
            "temporal_unverified": False,
        },
    ),
    Case(
        "3.I",
        Stage.VALIDATION,
        "Invented transition period is rejected",
        "parse_reject",
        {
            "question": "Y a-t-il un délai de six mois pour se conformer à 2025-13 ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": "Les opérateurs disposent de six mois pour se conformer.",
            "quote": "entre en vigueur à compter de la date de sa publication",
        },
    ),
    Case(
        "3.J",
        Stage.GENERATION,
        "Before entry into force grandfathering must keep exclusion polarity",
        "polarity_reject",
        {
            "question": (
                "Une importation dont l'exécution de l'engagement avait commencé avant "
                "l'entrée en vigueur de 2026-04 reste-t-elle dans le champ de l'article premier ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, ces importations restent dans le champ d'application de l'article premier."
            ),
            "quote": (
                "les importations ayant donné lieu, préalablement à la date d'entrée "
                "en vigueur de la présente circulaire"
            ),
        },
    ),
    Case(
        "3.K",
        Stage.VALIDATION,
        "Current calendar date must not be substituted into the rule quote",
        "parse_reject",
        {
            "question": "Selon 2026-04, à partir de quand les exclusions s'apprécient-elles ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Les exclusions s'apprécient par rapport au 20 septembre 2026, date du jour."
            ),
            "quote": "préalablement à la date d'entrée en vigueur de la présente circulaire",
        },
    ),
    Case(
        "3.L",
        Stage.VALIDATION,
        "Future rule treated as currently applicable without support is rejected",
        "parse_reject",
        {
            "question": "Le régime de 2099-01 est-il déjà applicable ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": "Oui, le régime futur 2099-01 s'applique déjà aujourd'hui.",
            "quote": "entre en vigueur à compter de la date de sa publication",
        },
    ),
    Case(
        "3.M",
        Stage.TEMPORAL,
        "Partial article replacement must not be read as whole-circular abrogate of unrelated text",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
        },
    ),
    Case(
        "3.N",
        Stage.TEMPORAL,
        "One article amended — edge targets 94-14, not a blanket circular wipe",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
        },
    ),
    # --- 4. Selection ---
    Case(
        "4.A",
        Stage.SELECTION,
        "Selector must accept sufficient grandfathering evidence",
        "parse_accept",
        {
            "question": (
                "Une importation non prioritaire avec engagement déjà en cours d'exécution "
                "avant l'entrée en vigueur de 2026-04 est-elle exclue de l'article premier ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, les importations ayant donné lieu à des engagements dont "
                "l'exécution a été effectivement entamée sont exclues."
            ),
            "quote": "dont l'exécution a été effectivement entamée",
            "temporal_unverified": False,
        },
    ),
    Case(
        "4.B",
        Stage.SELECTION,
        "Merely topical 'exportations' page does not answer import dépôt question",
        "regime_reject",
        {
            "question": (
                "Les importateurs de produits non prioritaires doivent-ils déposer 100% "
                "selon 2026-04 ?"
            ),
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Oui, les importateurs doivent déposer 100% de la valeur sur fonds propres."
            ),
            "quote": "règlement financier des importations et des exportations",
        },
    ),
    Case(
        "4.C",
        Stage.SELECTION,
        "Selecting only general rule while exception exists — polarity on industrial",
        "polarity_reject",
        {
            "question": (
                "Une entreprise industrielle doit-elle déposer 100% au titre de 2026-04 "
                "article premier ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, toute entreprise industrielle doit constituer le dépôt de 100%."
            ),
            "quote": "entreprises industrielles",
        },
    ),
    Case(
        "4.D",
        Stage.RETRIEVAL,
        "Newer less-relevant document loses to named instrument",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, police d'assurance-crédit à l'export ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2026_04_fr.pdf", 2),
            "named_score": 0.42,
            "distractor_score": 0.91,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "4.E",
        Stage.COVERAGE,
        "Complementary general + exception both required in pack",
        "pack_needs_both",
        {
            "general": ("Cir_2026_04_fr.pdf", 2),
            "exception": ("Cir_2026_04_fr.pdf", 3),
            "general_must": "ne peuvent mettre à la",
            "exception_must": "entreprises industrielles",
        },
    ),
    Case(
        "4.F",
        Stage.VALIDATION,
        "High lexical similarity but wrong legal conclusion rejected",
        "parse_reject",
        {
            "question": "Selon 2025-13, le délai libre sans condition est-il de 60 jours ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Oui, le délai libre sans condition est de 60 jours.",
            "quote": "délais de règlement allant jusqu’à 120 jours",
        },
    ),
    Case(
        "4.G",
        Stage.TEMPORAL,
        "Selector/history: Vu text must not become operative edge",
        "extract_empty",
        {"source": "Cir_2026_04_fr.pdf", "page": 1},
    ),
    Case(
        "4.H",
        Stage.GENERATION,
        "Layered general+exception is not unresolved conflict",
        "parse_accept",
        {
            "question": (
                "Les entreprises industrielles sont-elles exclues de l'article premier "
                "de 2026-04 sous réserve d'une fiche technique ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, les importations réalisées par les entreprises industrielles sont "
                "exclues sous réserve de la production d'une fiche technique spéciale."
            ),
            "quote": (
                "les importations réalisées par les entreprises industrielles, sous "
                "réserve de la production par lesdites entreprises, d'une fiche technique spéciale"
            ),
            "temporal_unverified": False,
        },
    ),
    Case(
        "4.I",
        Stage.VALIDATION,
        "User scenario facts need not be repeated on the page",
        "parse_accept",
        {
            "question": (
                "Une entreprise importe un produit non prioritaire; l'intermédiaire avait "
                "déjà pris un engagement et l'exécution avait commencé avant l'entrée en "
                "vigueur de 2026-04. Les restrictions de l'article premier s'appliquent-elles ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Non, ces opérations sont exclues du champ d'application de l'article premier."
            ),
            "quote": "dont l'exécution a été effectivement entamée",
            "temporal_unverified": False,
        },
    ),
    Case(
        "4.J",
        Stage.SELECTION,
        "Too much unrelated evidence must not allow off-topic confirmed answer",
        "regime_reject",
        {
            "question": "Le dépôt de 100% de 2026-04 s'applique-t-il aux exportations ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Oui, les exportateurs doivent aussi déposer 100% sur fonds propres.",
            "quote": "vente",
        },
    ),
    # --- 5. Generation ---
    Case(
        "5.A",
        Stage.GENERATION,
        "Polarity reversal excluded→included",
        "polarity_reject",
        {
            "question": "Les opérations listées à l'article 4 de 2026-04 sont-elles exclues ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "claim": "Non, ces opérations sont incluses dans le champ de l'article premier.",
            "quote": "Sont exclues du champ d'application",
        },
    ),
    Case(
        "5.B",
        Stage.GENERATION,
        "Negation loss on authorization",
        "parse_reject",
        {
            "question": (
                "Selon 2025-13, une vente >360 jours est-elle libre sans autorisation BCT ?"
            ),
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, ces ventes sont libres et ne nécessitent pas d'autorisation préalable."
            ),
            "quote": "sont soumises à l’autorisation préalable de la Banque Centrale de Tunisie",
        },
    ),
    Case(
        "5.C",
        Stage.GENERATION,
        "Exception ignored — industrial still forced to deposit",
        "polarity_reject",
        {
            "question": "Une entreprise industrielle est-elle exclue de l'article premier 2026-04 ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Non, aucune exclusion n'existe pour les entreprises industrielles.",
            "quote": "entreprises industrielles",
        },
    ),
    Case(
        "5.D",
        Stage.VALIDATION,
        "Condition dropped (A and B → A only) rejected when quote lacks reserve",
        "parse_reject",
        {
            "question": (
                "L'exclusion industrielle de 2026-04 s'applique-t-elle sans fiche technique ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, toute importation industrielle est exclue sans autre condition."
            ),
            "quote": "les importations réalisées par les entreprises industrielles",
        },
    ),
    Case(
        "5.E",
        Stage.VALIDATION,
        "Condition invented",
        "parse_reject",
        {
            "question": "Quand 2025-13 entre-t-elle en vigueur ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": (
                "Elle entre en vigueur 30 jours après publication au JORT uniquement."
            ),
            "quote": "entre en vigueur à compter de la date de sa publication",
        },
    ),
    Case(
        "5.F",
        Stage.VALIDATION,
        "Threshold off-by-one 360→361",
        "parse_reject",
        {
            "question": "Selon 2025-13, au-delà de combien de jours faut-il une autorisation ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": "Les ventes prévoyant des délais supérieurs à 361 jours sont soumises à autorisation.",
            "quote": "délais de règlement supérieurs à 360 jours",
        },
    ),
    Case(
        "5.G",
        Stage.VALIDATION,
        "Unit error — invent working days",
        "parse_reject",
        {
            "question": "Selon 2025-13, le délai de 120 jours est-il en jours ouvrables ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Oui, il s'agit de 120 jours ouvrables.",
            "quote": "jusqu’à 120 jours à compter de la date d’expédition des marchandises",
        },
    ),
    Case(
        "5.H",
        Stage.GENERATION,
        "Scope expansion industrial → all importers",
        "polarity_reject",
        {
            "question": "Qui bénéficie de l'exclusion de l'article 4 de 2026-04 pour l'industrie ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Tous les importateurs sont exclus du champ de l'article premier.",
            "quote": "entreprises industrielles",
        },
    ),
    Case(
        "5.I",
        Stage.GENERATION,
        "Scope contraction — only industrial, ignoring other exclusions",
        "parse_accept",
        {
            "question": (
                "Outre les entreprises industrielles, 2026-04 exclut-il aussi les engagements "
                "déjà en cours d'exécution ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, sont aussi exclues les importations ayant donné lieu à des engagements "
                "dont l'exécution a été effectivement entamée avant l'entrée en vigueur."
            ),
            "quote": "dont l'exécution a été effectivement entamée",
        },
    ),
    Case(
        "5.J",
        Stage.VALIDATION,
        "Actor confusion bank vs importer",
        "regime_reject",
        {
            "question": "Selon 2026-04, qui doit constituer le dépôt sur fonds propres ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "L'exportateur résident doit constituer le dépôt de 100%.",
            "quote": "exportateur résident",
        },
    ),
    Case(
        "5.K",
        Stage.GENERATION,
        "Transaction-direction reversal import↔export",
        "regime_reject",
        {
            "question": "Selon 2026-04, le dépôt de 100% concerne-t-il les exportations ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "claim": (
                "Oui, le dépôt de 100% s'applique aux exportations de marchandises."
            ),
            "quote": "importation de produits considérés non prioritaires",
        },
    ),
    Case(
        "5.L",
        Stage.CITATION,
        "Document identity drift 2025-13→2026-04 blocked by named identity",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, stand-by letter of credit ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2026_04_fr.pdf", 2),
            "named_score": 0.4,
            "distractor_score": 0.95,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "5.M",
        Stage.GENERATION,
        "Franken-rule from two circulars",
        "franken_reject",
        {
            "question": "Quel est le régime unique de délai libre à l'export ?",
            "sources": [
                ("Cir_2020_02_fr.pdf", 3),
                ("Cir_2025_13_fr.pdf", 2),
            ],
            "bad_claim": (
                "Le régime unique combine 60 jours (2020-02) et 120 jours (2025-13)."
            ),
            "bad_quote_source": "Cir_2025_13_fr.pdf",
            "bad_quote": "120 jours",
        },
    ),
    Case(
        "5.N",
        Stage.TEMPORAL,
        "Hallucinated abrogation 2026-04 of 2025-13",
        "extract_empty",
        {"source": "Cir_2026_04_fr.pdf", "page": 1},
    ),
    Case(
        "5.O",
        Stage.VALIDATION,
        "Question-fact overvalidation avoided for scenario date",
        "parse_accept",
        {
            "question": (
                "Engagement commencé avant le 26 mars 2026: les restrictions 2026-04 "
                "article premier s'appliquent-elles ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Non, les importations avec engagements déjà en cours d'exécution avant "
                "l'entrée en vigueur sont exclues."
            ),
            "quote": "dont l'exécution a été effectivement entamée",
        },
    ),
    Case(
        "5.P",
        Stage.VALIDATION,
        "Answer too broad beyond quote",
        "parse_reject",
        {
            "question": "Que dit 2025-13 sur le délai de 120 jours ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Tous les contrats de commerce extérieur mondiaux peuvent être réglés "
                "librement jusqu'à 120 jours sans aucune limite."
            ),
            "quote": "jusqu’à 120 jours à compter de la date d’expédition des marchandises",
        },
    ),
    Case(
        "5.Q",
        Stage.VALIDATION,
        "Answer too narrow still accepted if literally supported",
        "parse_accept",
        {
            "question": "2026-04 prévoit-il une exclusion pour les entreprises industrielles ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Oui, il existe une exclusion pour les entreprises industrielles.",
            "quote": "entreprises industrielles",
        },
    ),
    Case(
        "5.R",
        Stage.SELECTION,
        "Unnecessary abstention when evidence is sufficient",
        "parse_accept",
        {
            "question": "2025-13 remplace-t-il l'article 10 de la circulaire 94-14 ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Oui, les dispositions de l'article 10 de la circulaire 94-14 sont "
                "abrogées et remplacées."
            ),
            "quote": (
                "Les dispositions de l’article 10, du paragraphe premier de l’article 11 "
                "et de l’article 12 de la circulaire n° 94-14"
            ),
            "temporal_unverified": False,
        },
    ),
    Case(
        "5.S",
        Stage.VALIDATION,
        "False confidence without evidence fails closed",
        "parse_insufficient",
        {
            "question": "Quel est le taux exact de la taxe fantôme 2026-99 ?",
            "evidence": [],
        },
    ),
    Case(
        "5.T",
        Stage.GENERATION,
        "Legal operator paraphrase changes meaning peut→doit style",
        "parse_reject",
        {
            "question": "Selon 2025-13 article 10, le règlement est-il obligatoire par un seul moyen ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Les prix des ventes doivent être réglés exclusivement par un seul moyen imposé."
            ),
            "quote": "peuvent être réglés par n'importe quel moyen de règlement",
        },
    ),
    # --- 6. Validator ---
    Case(
        "6.false_reject_ok",
        Stage.VALIDATION,
        "Correct grounded answer accepted",
        "parse_accept",
        {
            "question": "Selon 2025-13, jusqu'à combien de jours sans condition ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Le délai libre sans condition est de 120 jours.",
            "quote": "jusqu’à 120 jours",
            "temporal_unverified": False,
        },
    ),
    Case(
        "6.false_accept_block",
        Stage.VALIDATION,
        "Incorrect answer rejected",
        "parse_reject",
        {
            "question": "Selon 2025-13, jusqu'à combien de jours sans condition ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Le délai libre sans condition est de 60 jours.",
            "quote": "jusqu’à 120 jours",
        },
    ),
    Case(
        "6.quote_too_short",
        Stage.VALIDATION,
        "Quote lacking operator insufficient for opposite claim",
        "parse_reject",
        {
            "question": "Les opérations de l'article 4 2026-04 sont-elles exclues ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Ces opérations sont pleinement soumises à l'article premier.",
            "quote": "article",
        },
    ),
    Case(
        "6.question_facts",
        Stage.VALIDATION,
        "Question facts are not regulatory claims",
        "parse_accept",
        {
            "question": (
                "Engagement démarré avant l'entrée en vigueur de 2026-04: exclusion ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Oui, ces opérations sont exclues du champ de l'article premier.",
            "quote": "dont l'exécution a été effectivement entamée",
            "temporal_unverified": False,
        },
    ),
    Case(
        "6.polarity_fp",
        Stage.GENERATION,
        "Page has excluded and other populations — wrong polarity still rejected",
        "polarity_reject",
        {
            "question": "Les entreprises industrielles sont-elles exclues de 2026-04 art.1 ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "claim": "Non, elles ne sont pas exclues.",
            "quote": "Sont exclues du champ d'application",
        },
    ),
    Case(
        "6.polarity_fn",
        Stage.GENERATION,
        "Negation via 'soumises à autorisation' vs 'libres'",
        "parse_reject",
        {
            "question": "Ventes >360 jours selon 2025-13: libres ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": "Oui, elles sont effectuées librement.",
            "quote": "sont soumises à l’autorisation préalable",
        },
    ),
    Case(
        "6.quote_semantically_insufficient",
        Stage.VALIDATION,
        "Quote present but lacks the operator that supports the claim",
        "parse_reject",
        {
            "question": "Selon 2026-04, les engagements déjà entamés sont-ils exclus ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Oui, ils sont exclus du champ de l'article premier.",
            "quote": "engagements pris par l'intermédiaire agréé",
        },
    ),
    Case(
        "6.overfit_wording",
        Stage.VALIDATION,
        "Correct paraphrase using near-synonyms still accepted when numbers/operators match",
        "parse_accept",
        {
            "question": "Selon 2025-13, délai libre max sans condition ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Le délai maximal sans condition particulière est de 120 jours.",
            "quote": "jusqu’à 120 jours",
            "temporal_unverified": False,
        },
    ),
    Case(
        "6.underfit_wording",
        Stage.VALIDATION,
        "Bad paraphrase sharing keywords but wrong number is rejected",
        "parse_reject",
        {
            "question": "Selon 2025-13, délai libre max sans condition ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Le délai de règlement libre sans condition est de 360 jours.",
            "quote": "jusqu’à 120 jours",
        },
    ),
    # --- 7. Citation ---
    Case(
        "7.wrong_pdf",
        Stage.CITATION,
        "Correct text attributed to wrong PDF rejected via identity",
        "named_only",
        {
            "question": "Selon la circulaire 2025-13, délai de 120 jours ?",
            "wrong_source": "Cir_2026_04_fr.pdf",
            "right_source": "Cir_2025_13_fr.pdf",
            "page": 2,
        },
    ),
    Case(
        "7.wrong_page",
        Stage.CITATION,
        "Citation page must match evidence page used",
        "citation_identity",
        {
            "question": "Selon 2025-13 article 12, autorisation préalable ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": (
                "Les ventes prévoyant des délais de règlement supérieurs à 360 jours sont "
                "soumises à l'autorisation préalable de la Banque Centrale de Tunisie."
            ),
            "quote": (
                "supérieurs à 360 jours sont soumises à l’autorisation préalable de la "
                "Banque Centrale de Tunisie"
            ),
            "expect_file": "Cir_2025_13_fr.pdf",
            "expect_page": 3,
            "temporal_unverified": False,
        },
    ),
    Case(
        "7.half_support",
        Stage.VALIDATION,
        "Multi-part claim only half-supported fails or stays partial",
        "parse_reject",
        {
            "question": "Selon 2025-13, 120 jours libres et 50% acompte à l'import ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Le délai libre est de 120 jours et l'acompte à l'import est plafonné à 50%."
            ),
            "quote": "jusqu’à 120 jours",
        },
    ),
    Case(
        "7.wrong_article",
        Stage.VALIDATION,
        "Wrong article number in claim vs quoted article",
        "parse_reject",
        {
            "question": "Selon 2025-13, que dit l'article 12 sur les délais >360 jours ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": (
                "L'article 11 impose l'autorisation préalable pour les délais "
                "supérieurs à 360 jours."
            ),
            "quote": "Article 12 (nouveau) : Les ventes dont les contrats",
        },
    ),
    Case(
        "7.historical_vs_operative",
        Stage.CITATION,
        "Vu historical citation must not outrank the operative named instrument",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, délai libre sans condition ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2026_04_fr.pdf", 1),
            "named_score": 0.4,
            "distractor_score": 0.95,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "7.multipart_one_cite",
        Stage.VALIDATION,
        "One citation cannot carry an unsupported second legal number",
        "parse_reject",
        {
            "question": "Selon 2025-13, 120 jours et autorisation dès 200 jours ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "Le délai libre est de 120 jours et l'autorisation est exigée dès 200 jours."
            ),
            "quote": "jusqu’à 120 jours",
        },
    ),
    Case(
        "7.multi_source_one_supports",
        Stage.GENERATION,
        "Two sources cited but merged conclusion must not invent a hybrid rule",
        "franken_reject",
        {
            "question": "Quel délai libre unique s'applique ?",
            "sources": [
                ("Cir_2020_02_fr.pdf", 3),
                ("Cir_2025_13_fr.pdf", 2),
            ],
            "bad_claim": (
                "Le délai libre unique est de 90 jours, moyenne de 60 et 120."
            ),
            "bad_quote_source": "Cir_2025_13_fr.pdf",
            "bad_quote": "120 jours",
        },
    ),
    # --- 8. Current rule ---
    Case(
        "8.old_vs_new",
        Stage.TEMPORAL,
        "Current delay rule is 2025-13 (120), not 2020-02 (60), when named",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, délai libre sans condition ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2020_02_fr.pdf", 3),
            "named_score": 0.33,
            "distractor_score": 0.96,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "8.partial_amendment",
        Stage.TEMPORAL,
        "2025-13 replaces named 94-14 articles only — not an unrelated circular wipe",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
            "forbid_target": "cir:2026:4",
        },
    ),
    Case(
        "8.mention_not_replace",
        Stage.TEMPORAL,
        "2026-04 mentions 2025-13 without replacing it",
        "extract_empty",
        {"source": "Cir_2026_04_fr.pdf", "page": 1},
    ),
    Case(
        "8.exception_current",
        Stage.COVERAGE,
        "Current 2026-04 exception page assembled with general rule",
        "expand_adjacent",
        {
            "hit": ("Cir_2026_04_fr.pdf", 2),
            "must_include_page": 3,
            "must_contain": "Sont exclues",
        },
    ),
    Case(
        "8.grandfather",
        Stage.GENERATION,
        "Grandfathered transactions excluded",
        "parse_accept",
        {
            "question": (
                "Opération dont l'exécution avait commencé avant l'entrée en vigueur "
                "de 2026-04: article premier applicable ?"
            ),
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": "Non, ces opérations sont exclues du champ de l'article premier.",
            "quote": "dont l'exécution a été effectivement entamée",
            "temporal_unverified": False,
        },
    ),
    Case(
        "8.effective_date",
        Stage.VALIDATION,
        "Publication-based entry into force",
        "parse_accept",
        {
            "question": "2026-04 entre-t-elle en vigueur à sa publication ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "claim": (
                "Oui, la circulaire entre en vigueur à compter de la date de sa publication."
            ),
            "quote": "entre en vigueur à compter de la date de sa publication",
            "temporal_unverified": False,
        },
    ),
    Case(
        "8.future_effective",
        Stage.VALIDATION,
        "Future instrument treated as already applicable is rejected",
        "parse_reject",
        {
            "question": "La circulaire 2099-01 est-elle déjà applicable aujourd'hui ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 3,
            "claim": "Oui, la circulaire 2099-01 s'applique déjà aujourd'hui.",
            "quote": "entre en vigueur à compter de la date de sa publication",
        },
    ),
    # --- 9. Multi-hop ---
    Case(
        "9.multi_hop_pack",
        Stage.COVERAGE,
        "A→amendment B→exception C requires general+exception pack",
        "pack_needs_both",
        {
            "general": ("Cir_2026_04_fr.pdf", 2),
            "exception": ("Cir_2026_04_fr.pdf", 3),
            "general_must": "Article premier",
            "exception_must": "préalablement à la date",
        },
    ),
    Case(
        "9.amendment_chain",
        Stage.TEMPORAL,
        "Operative chain 2025-13 → 94-14 articles",
        "extract_action",
        {
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "expect_action": "REPLACE",
            "expect_target": "cir:1994:14",
        },
    ),
    # --- 10. Adversarial ---
    Case(
        "10.same_words_wrong_doc",
        Stage.RETRIEVAL,
        "Same words, wrong document",
        "prefer_named_domain",
        {
            "query": (
                "Selon la circulaire 2026-04 relative aux produits non prioritaires, "
                "crédit documentaire et dépôt ?"
            ),
            "named": ("Cir_2026_04_fr.pdf", 2),
            "distractor": ("Cir_2020_02_fr.pdf", 4),
            "named_score": 0.4,
            "distractor_score": 0.99,
            "expect_top": "Cir_2026_04_fr.pdf",
        },
    ),
    Case(
        "10.same_number_wrong_article",
        Stage.VALIDATION,
        "Article 11 vs Article 1 numeric near-miss",
        "parse_reject",
        {
            "question": "Selon 2025-13, que dit l'article 11 sur 121-360 jours ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": (
                "L'article 1 fixe seul les conditions de 121 à 360 jours avec garantie."
            ),
            "quote": "Article 11 : Paragraphe premier",
        },
    ),
    Case(
        "10.newer_irrelevant",
        Stage.RETRIEVAL,
        "Newer irrelevant circular",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13…",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2026_04_fr.pdf", 1),
            "named_score": 0.4,
            "distractor_score": 0.99,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "10.older_similar",
        Stage.RETRIEVAL,
        "Older more similar circular loses to named newer rule",
        "prefer_named",
        {
            "query": "Selon la circulaire 2025-13, 120 jours ?",
            "named": ("Cir_2025_13_fr.pdf", 2),
            "distractor": ("Cir_2020_02_fr.pdf", 3),
            "named_score": 0.3,
            "distractor_score": 0.95,
            "expect_top": "Cir_2025_13_fr.pdf",
        },
    ),
    Case(
        "10.general_without_exception",
        Stage.COVERAGE,
        "General without exception fails pack_needs_both invariant",
        "pack_needs_both",
        {
            "general": ("Cir_2026_04_fr.pdf", 2),
            "exception": ("Cir_2026_04_fr.pdf", 3),
            "general_must": "Article premier",
            "exception_must": "entreprises industrielles",
        },
    ),
    Case(
        "10.double_negation",
        Stage.GENERATION,
        "Double negation / exclusion polarity",
        "polarity_reject",
        {
            "question": "N'est-il pas vrai que les industrielles ne sont pas exclues ?",
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "claim": "Exact, les entreprises industrielles ne sont pas exclues.",
            "quote": "Sont exclues du champ d'application",
        },
    ),
    Case(
        "10.boundary_number",
        Stage.VALIDATION,
        "Boundary 120 vs 121",
        "parse_reject",
        {
            "question": "121 jours sans condition selon 2025-13 ?",
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "claim": "Oui, 121 jours restent sans condition comme 120 jours.",
            "quote": "jusqu’à 120 jours",
        },
    ),
    Case(
        "10.wrong_named_doc",
        Stage.RETRIEVAL,
        "Question names wrong document — still anchors to named id",
        "identity",
        {
            "query": "Selon la circulaire 2099-01, dépôt 100% ?",
            "kind": "cir",
            "year": 2099,
            "number": 1,
        },
    ),
    Case(
        "10.right_named_doc",
        Stage.RETRIEVAL,
        "Question names right document",
        "identity",
        {
            "query": "Selon la circulaire 2025-13…",
            "kind": "cir",
            "year": 2025,
            "number": 13,
        },
    ),
    Case(
        "10.ambiguous",
        Stage.SELECTION,
        "Ambiguous question has no explicit instrument identity",
        "identity_none",
        {"query": "Est-ce toujours valable aujourd'hui ?"},
    ),
    Case(
        "10.insufficient",
        Stage.VALIDATION,
        "Insufficient evidence abstains",
        "parse_insufficient",
        {
            "question": "Quel est le taux de la taxe inexistante 1234-99 ?",
            "evidence": [],
        },
    ),
]
