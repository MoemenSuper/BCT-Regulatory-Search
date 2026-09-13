"""Regressions for Groq-compatible Graph Lite extraction fixes."""

import json

from regulatory_graph_lite.builder import _normalize_extraction_json, page_may_contain_relationship
from regulatory_graph_lite.identity import canonical_instrument_id, instrument_from_properties
from regulatory_graph_lite.validation import _recover_exact_quote, validate_extracted_relationship
from regulatory_graph_lite.identity import instrument_from_filename


def test_normalize_extraction_json_maps_source_target_and_missing_ids():
    raw = json.dumps(
        {
            "nodes": [
                {"label": "Instrument", "kind": "cir", "year": 2018, "number": 9},
                {"label": "Instrument", "kind": "cir", "year": 2017, "number": 8},
            ],
            "relationships": [
                {
                    "type": "ABROGATES",
                    "source": 0,
                    "target": 1,
                    "evidence_quote": "sont abrogées",
                }
            ],
        }
    )

    payload = _normalize_extraction_json(raw)

    assert payload["nodes"][0]["id"] == "0"
    assert payload["nodes"][0]["properties"]["year"] == 2018
    assert payload["relationships"][0]["start_node_id"] == "0"
    assert payload["relationships"][0]["end_node_id"] == "1"
    assert payload["relationships"][0]["properties"]["evidence_quote"] == "sont abrogées"


def test_two_digit_year_expands_to_gregorian_instrument_id():
    assert canonical_instrument_id("cir", 91, 24) == "cir:1991:24"
    assert instrument_from_properties({"kind": "cir", "year": 91, "number": 24}).id == "cir:1991:24"


def test_prefilter_accepts_short_year_circular_reference():
    text = (
        "Article 2 : Les dispositions de l'article 4 de la circulaire n°91-24 "
        "du 17 décembre 1991 sont abrogées et remplacées."
    )
    assert page_may_contain_relationship(text) is True


def test_ellipsis_quote_recovery_keeps_contiguous_source_span():
    source = (
        "Sont abrogées toutes dispositions contraires avec la présente circulaire "
        "et notamment la circulaire n°2001-11 du 4 mai 2001 relative au marché."
    )
    recovered = _recover_exact_quote(
        source,
        "Sont abrogées ... la circulaire n°2001-11 du 4 mai 2001",
    )
    assert recovered is not None
    assert "circulaire n°2001-11" in recovered
    assert "Décide" not in recovered


def test_two_digit_year_target_properties_validate_against_catalog():
    source = "Cir_2018_09_fr.pdf"
    source_ref = instrument_from_filename(source)
    target_ref = instrument_from_filename("Cir_2017_08_fr.pdf")
    catalog = {source_ref.id: source_ref, target_ref.id: target_ref}
    text = (
        "Les dispositions de l'article 58 de la circulaire n°2017-08 sont abrogées."
    )
    nodes = {
        "0": {
            "label": "Instrument",
            "properties": {"name": source, "kind": "cir", "year": 2018, "number": 9},
        },
        "1": {
            "label": "Instrument",
            "properties": {
                "name": "circulaire 2017-08",
                "kind": "cir",
                "year": 17,
                "number": 8,
            },
        },
    }

    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=2,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "ABROGATES",
            "start_node_id": "0",
            "end_node_id": "1",
            "properties": {"evidence_quote": text},
        },
        catalog=catalog,
    )

    assert candidate is not None
    assert candidate.target.id == "cir:2017:8"
