"""Authority tags, query class, and multimodal page handling."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_authority import authority_for_kind, normalize_doc_kind, resolve_doc_kind
from query_authority import (
    allowed_doc_kinds,
    classify_query_authority,
    evidence_kind_allowed,
    select_demonstrations,
)
from answer_gates import parse_answer


def test_doc_kind_maps_to_authority():
    assert normalize_doc_kind("bulletin") == "statistical"
    assert authority_for_kind("statistical") == "secondary"
    assert authority_for_kind("regulatory") == "primary"
    assert resolve_doc_kind(filename="Stats_export_2024_fr.pdf") == "statistical"
    assert resolve_doc_kind(filename="Cir_2026_04_fr.pdf") == "regulatory"


def test_allowed_kinds_never_reject_uncertain():
    assert allowed_doc_kinds("uncertain") == frozenset({"regulatory"})
    assert allowed_doc_kinds("statistical_fact") == frozenset({"statistical", "regulatory"})
    assert evidence_kind_allowed("regulatory_rule", doc_kind="statistical") is False
    assert evidence_kind_allowed("statistical_fact", doc_kind="statistical") is True


def test_dynamic_demos_cover_hard_negative_taux():
    demos = select_demonstrations("Quel est le taux applicable selon la circulaire ?")
    labels = {label for _q, label, _n in demos}
    assert "regulatory_rule" in labels
    assert len(demos) >= 4


def test_classify_query_authority_fails_closed():
    from query_authority import classify_query_authority

    class Boom:
        def invoke(self, _prompt):
            raise RuntimeError("down")

    payload = classify_query_authority(Boom(), "Quel était le volume d'exportations ?")
    assert payload["query_class"] == "uncertain"


def test_classify_query_authority_parses_structured_json():
    class Fake:
        def invoke(self, _prompt):
            return (
                '{"query_class":"statistical_fact","confidence":"high",'
                '"rationale":"volume bulletin"}'
            )

    payload = classify_query_authority(Fake(), "volume d'exportations 2024")
    assert payload["query_class"] == "statistical_fact"


def test_secondary_source_on_regulatory_query_is_flagged_not_refused():
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Bulletin_2024_fr.pdf",
            "page": 1,
            "text": "Le volume d'exportations atteint 12 milliards.",
            "score": 0.9,
            "doc_kind": "statistical",
            "authority": "secondary",
        }
    ]
    draft = {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": "Le volume d'exportations atteint 12 milliards.",
                "cites": ["E1.1"],
            }
        ],
    }
    flagged = parse_answer(
        json.dumps(draft),
        "Quel est le plafond reglementaire ?",
        evidence,
        query_class="regulatory_rule",
    )
    # A bulletin may state the fact but is not the legal authority: partial + trusted notice.
    assert flagged["status"] == "partial_answer"
    assert "12 milliards" in flagged["answer"]
    assert "pas sur un texte réglementaire" in flagged["answer"]

    allowed = parse_answer(
        json.dumps(draft),
        "Quel etait le volume d'exportations ?",
        evidence,
        query_class="statistical_fact",
    )
    assert allowed["status"] == "answered"
    assert allowed["sources"]
