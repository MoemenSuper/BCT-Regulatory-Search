"""False-abstention and adjacent-page assembly regressions."""
from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from answer_contract import generate_grounded_answer, parse_answer
from retrieval_selection import (
    expand_adjacent_instrument_pages,
    expand_answer_pages,
    page_chunks,
)


TEST3_QUESTION = (
    "Une entreprise importe un produit non prioritaire, mais son intermediaire agree "
    "avait deja pris un engagement pour lui accorder un concours financier avant le "
    "26 mars 2026 et l'execution de cet engagement avait deja commence. Les nouvelles "
    "restrictions de la circulaire n°2026-04 s'appliquent-elles ?"
)

TEST3_EXCEPTION = (
    "Sont exclues du champ d'application des dispositions de l'article premier de la "
    "presente circulaire, les operations suivantes : les importations ayant donne lieu, "
    "prealablement a la date d'entree en vigueur de la presente circulaire, a des "
    "engagements pris par l'intermediaire agree pour l'octroi de concours financiers, "
    "dont l'execution a ete effectivement entamee ; les importations realisees par les "
    "entreprises industrielles, sous reserve de la production par lesdites entreprises, "
    "d'une fiche technique speciale."
)

TEST5_QUESTION = (
    "Une entreprise industrielle tunisienne importe un produit classe non prioritaire. "
    "Doit-elle obligatoirement deposer 100 % de la valeur de l'importation sur ses "
    "fonds propres aupres de l'intermediaire agree ?"
)

PAGE2_GENERAL = (
    "Article premier : les importateurs constituent, sur leurs fonds propres, des "
    "depots couvrant la totalite de la valeur des importations envisagees. Cette "
    "obligation s'impose independamment du mode de reglement financier desdites "
    "importations."
)

PAGE3_INDUSTRIAL = (
    "Article 4 : Sont exclues du champ d'application des dispositions de l'article "
    "premier de la presente circulaire, les operations suivantes : les importations "
    "realisees par les entreprises industrielles, sous reserve de la production par "
    "lesdites entreprises, d'une fiche technique speciale destinee a l'intermediaire "
    "agree, delivree par les services competents du ministere de l'industrie."
)


def _parse(draft, evidence, question):
    return parse_answer(json.dumps(draft, ensure_ascii=False), question, evidence)


def test_test3_exception_in_pack_accepts_scenario_date_and_legal_quote():
    """Legal consequence quoted; scenario date from the question need not be on the page."""
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "score": 0.9,
            "text": TEST3_EXCEPTION,
        }
    ]
    draft = {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": (
                    "Non, les nouvelles restrictions ne s'appliquent pas aux importations "
                    "ayant deja donne lieu a des engagements de concours financiers dont "
                    "l'execution a ete entamee avant le 26 mars 2026."
                ),
                "quotes": [
                    {
                        "evidence_id": "E1",
                        "quote": (
                            "les importations ayant donne lieu, prealablement a la date "
                            "d'entree en vigueur de la presente circulaire, a des "
                            "engagements pris par l'intermediaire agree pour l'octroi de "
                            "concours financiers, dont l'execution a ete effectivement entamee"
                        ),
                    }
                ],
            }
        ],
    }
    result = _parse(draft, evidence, TEST3_QUESTION)
    assert result["status"] in {"answered", "partial_answer"}
    assert "engagements" in result["answer"].casefold()
    assert result["sources"][0]["file"] == "Cir_2026_04_fr.pdf"


def test_unsupported_legal_number_still_rejected_when_not_in_question():
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "score": 0.9,
            "text": PAGE2_GENERAL,
        }
    ]
    draft = {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": "L'importateur doit deposer seulement 50 % de la valeur.",
                "quotes": [
                    {
                        "evidence_id": "E1",
                        "quote": "des depots couvrant la totalite de la valeur des importations envisagees",
                    }
                ],
            }
        ],
    }
    result = _parse(draft, evidence, TEST5_QUESTION)
    assert result["status"] == "insufficient_evidence"


def test_regime_remap_still_rejected_for_off_topic_export_page():
    page = (
        "Les prix des ventes peuvent etre regles par n'importe quel moyen de reglement, "
        "lorsque les contrats y afferents prevoient des delais de reglement allant jusqu'a "
        "60 jours a compter de la date d'expedition des marchandises."
    )
    draft = {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": (
                    "Une entreprise peut regler un fournisseur etranger par n'importe quel "
                    "moyen de paiement lorsque le contrat prevoit un delai de 60 jours."
                ),
                "quotes": [{"evidence_id": "E1", "quote": page[:120]}],
            }
        ],
    }
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Cir_2020_02_fr.pdf",
            "page": 2,
            "score": 0.9,
            "text": page,
        }
    ]
    result = _parse(
        draft,
        evidence,
        "Est-ce qu'une entreprise peut payer un fournisseur a l'etranger ?",
    )
    assert result["status"] == "insufficient_evidence"


def test_generate_grounded_answer_does_not_abstain_when_test3_exception_selected():
    page3 = Document(
        page_content=TEST3_EXCEPTION,
        metadata={"source": "Cir_2026_04_fr.pdf", "page": 3, "pages": [3]},
    )
    page2 = Document(
        page_content=PAGE2_GENERAL,
        metadata={"source": "Cir_2026_04_fr.pdf", "page": 2, "pages": [2]},
    )
    draft = {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": (
                    "Non: les importations ayant deja donne lieu a des engagements de "
                    "concours financiers dont l'execution a ete effectivement entamee "
                    "avant l'entree en vigueur sont exclues des nouvelles restrictions."
                ),
                "quotes": [
                    {
                        "evidence_id": "E1",
                        "quote": (
                            "les importations ayant donne lieu, prealablement a la date "
                            "d'entree en vigueur de la presente circulaire, a des "
                            "engagements pris par l'intermediaire agree pour l'octroi de "
                            "concours financiers, dont l'execution a ete effectivement entamee"
                        ),
                    }
                ],
            }
        ],
    }
    selection = {
        "decision": "answer",
        "answer_intent": "conditions",
        "reason": "exception on page 3",
        "evidence_ids": ["E1", "E2"],
    }
    calls = {"n": 0}

    def respond(_prompt):
        calls["n"] += 1
        # First call is evidence selection; second is the answer draft.
        if calls["n"] == 1:
            return AIMessage(content=json.dumps(selection, ensure_ascii=False))
        return AIMessage(content=json.dumps(draft, ensure_ascii=False))

    result = generate_grounded_answer(
        RunnableLambda(respond),
        TEST3_QUESTION,
        [(page3, 0.95), (page2, 0.9)],
    )
    assert result["status"] in {"answered", "partial_answer"}
    assert result["status"] != "search_results"
    assert any(s["page"] == 3 for s in result["sources"])


def test_adjacent_page_expansion_includes_industrial_exception():
    hit = Document(
        page_content=PAGE2_GENERAL,
        metadata={
            "source": "Cir_2026_04_fr.pdf",
            "page": 2,
            "pages": [2],
            "page_label": 2,
            "chunk_index": 0,
        },
    )
    neighbor = Document(
        page_content=PAGE3_INDUSTRIAL,
        metadata={
            "source": "Cir_2026_04_fr.pdf",
            "page": 3,
            "pages": [3],
            "page_label": 3,
            "chunk_index": 0,
        },
    )
    other = Document(
        page_content="autre circulaire sans lien",
        metadata={
            "source": "Cir_2025_13_fr.pdf",
            "page": 2,
            "pages": [2],
            "page_label": 2,
            "chunk_index": 0,
        },
    )
    pages = page_chunks([hit, neighbor, other])
    expanded = expand_answer_pages([(hit, 0.9)], pages)
    sources_pages = {(d.metadata["source"], int(d.metadata["page"])) for d, _ in expanded}
    assert ("Cir_2026_04_fr.pdf", 2) in sources_pages
    assert ("Cir_2026_04_fr.pdf", 3) in sources_pages
    assert ("Cir_2025_13_fr.pdf", 2) not in sources_pages
    joined = " ".join(d.page_content for d, _ in expanded)
    assert "entreprises industrielles" in joined


def test_adjacent_expansion_does_not_cross_documents():
    hit = Document(
        page_content=PAGE2_GENERAL,
        metadata={"source": "Cir_2026_04_fr.pdf", "page": 2, "pages": [2], "page_label": 2},
    )
    foreign = Document(
        page_content=PAGE3_INDUSTRIAL,
        metadata={"source": "Cir_2025_13_fr.pdf", "page": 3, "pages": [3], "page_label": 3},
    )
    pages = page_chunks([hit, foreign])
    expanded = expand_adjacent_instrument_pages([(hit, 0.9)], pages)
    assert all(d.metadata["source"] == "Cir_2026_04_fr.pdf" for d, _ in expanded)
    assert len(expanded) == 1
