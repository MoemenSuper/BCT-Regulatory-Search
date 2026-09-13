from regulatory_graph_lite.identity import instrument_from_filename
from regulatory_graph_lite.models import VerificationStatus
from regulatory_graph_lite.validation import validate_extracted_relationship


def _fixture():
    source = "Cir_2025_17_fr.pdf"
    target = "Cir_2017_08_fr.pdf"
    source_ref = instrument_from_filename(source)
    target_ref = instrument_from_filename(target)
    catalog = {source_ref.id: source_ref, target_ref.id: target_ref}
    nodes = {
        "s": {
            "label": "Instrument",
            "properties": {"name": source, "kind": "cir", "year": 2025, "number": 17},
        },
        "t": {
            "label": "Instrument",
            "properties": {
                "name": "circulaire n° 2017-08",
                "kind": "cir",
                "year": 2017,
                "number": 8,
            },
        },
    }
    return source, catalog, nodes


def test_explicit_replacement_is_auto_verified_after_deterministic_evidence_checks():
    source, catalog, nodes = _fixture()
    text = "La présente circulaire remplace la circulaire n° 2017-08 à compter du 1er janvier 2026."
    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=3,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "REPLACES",
            "start_node_id": "s",
            "end_node_id": "t",
            "properties": {"evidence_quote": text, "effective_date": "2026-01-01"},
        },
        catalog=catalog,
    )
    assert candidate is not None
    assert candidate.verification_status is VerificationStatus.VERIFIED
    assert candidate.target_in_catalog is True
    assert candidate.evidence_quote == text
    assert candidate.verification_method == "AUTO_DETERMINISTIC_V1"


def test_invented_quote_is_rejected():
    source, catalog, nodes = _fixture()
    text = "La présente circulaire remplace la circulaire n° 2017-08."
    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=3,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "REPLACES",
            "start_node_id": "s",
            "end_node_id": "t",
            "properties": {"evidence_quote": "Cette citation n'existe pas dans le PDF."},
        },
        catalog=catalog,
    )
    assert candidate is None


def test_reversed_llm_edge_is_rejected():
    source, catalog, nodes = _fixture()
    text = "La présente circulaire remplace la circulaire n° 2017-08."
    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=3,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "REPLACES",
            "start_node_id": "t",
            "end_node_id": "s",
            "properties": {"evidence_quote": text},
        },
        catalog=catalog,
    )
    assert candidate is None


def test_amendment_without_action_word_is_rejected():
    source, catalog, nodes = _fixture()
    text = "Vu la circulaire n° 2017-08 relative aux opérations concernées."
    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=3,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "AMENDS",
            "start_node_id": "s",
            "end_node_id": "t",
            "properties": {"evidence_quote": text},
        },
        catalog=catalog,
    )
    assert candidate is None


def test_target_number_with_leading_zero_survives_conservative_fallback():
    source = "Cir_2025_17_fr.pdf"
    source_ref = instrument_from_filename(source)
    target_ref = instrument_from_filename("Cir_2017_03_fr.pdf")
    catalog = {source_ref.id: source_ref, target_ref.id: target_ref}
    nodes = {
        "s": {"label": "Instrument", "properties": {"name": source, "kind": "cir", "year": 2025, "number": 17}},
        "t": {"label": "Instrument", "properties": {"name": "Cir_2017_03_fr.pdf", "kind": "cir", "year": 2017, "number": 3}},
    }
    text = "La circulaire 2025-17 remplace la circulaire 2017-03."
    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=1,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "REPLACES",
            "start_node_id": "s",
            "end_node_id": "t",
            "properties": {"evidence_quote": text},
        },
        catalog=catalog,
    )
    assert candidate is not None
    assert candidate.target.id == "cir:2017:3"


def test_target_absent_from_trusted_catalog_is_discarded_instead_of_waiting_for_review():
    source, catalog, nodes = _fixture()
    catalog = {next(iter(catalog)): next(iter(catalog.values()))}
    text = "La présente circulaire remplace la circulaire n° 2017-08."
    candidate = validate_extracted_relationship(
        source_file=source,
        source_page=3,
        source_text=text,
        nodes_by_id=nodes,
        relationship={
            "type": "REPLACES",
            "start_node_id": "s",
            "end_node_id": "t",
            "properties": {"evidence_quote": text},
        },
        catalog=catalog,
    )
    assert candidate is None
