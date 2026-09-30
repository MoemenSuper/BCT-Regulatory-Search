import json
import pytest
from answer_contract import parse_answer


EVIDENCE = [{"evidence_id": "E1", "source": "Cir_2020_03_fr.pdf", "page": 1,
             "score": 0.8, "text": "Objet : Les allocations pour voyages d'affaires."}]


def draft(evidence_id="E1", unit=1):
    return {"status": "answered", "message": "UNTRUSTED FREE TEXT MUST NOT BE DISPLAYED",
            "claims": [{"text": "Elle concerne les allocations pour voyages d'affaires.",
                        "cites": [f"{evidence_id}.{unit}"]}]}


def test_only_claims_and_actual_cited_sources_are_displayed():
    result = parse_answer(json.dumps(draft()), "Quel est l'objet ?", EVIDENCE)
    assert result["status"] == "answered"
    assert "UNTRUSTED" not in result["answer"]
    assert result["sources"] == [{"file": "Cir_2020_03_fr.pdf", "page": 1, "score": 0.8,
                                 "excerpt": "Objet : Les allocations pour voyages d'affaires."}]


@pytest.mark.parametrize("value", [draft("E999"), draft(unit=9),
    {"status": "answered", "message": "A fabricated legal conclusion", "claims": []},
    {"status": "insufficient_evidence", "message": "A fabricated legal conclusion", "claims": []}])
def test_invalid_or_unsupported_output_has_no_answer_sources(value):
    result = parse_answer(json.dumps(value), "Quel est l'objet ?", EVIDENCE)
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
    assert "fabricated" not in result["answer"]


def test_conflicting_native_header_is_a_warning_not_a_discarded_page():
    from langchain_core.documents import Document
    from answer_contract import evidence_records
    text = "CIRCULAIRE AUX INTERMEDIAIRES AGREES n° 2002-03 du 04 février 2020\n" + EVIDENCE[0]["text"]
    records = evidence_records([(Document(page_content=text, metadata={"source": "Cir_2020_03_fr.pdf", "page": 1}), 0.8)])
    assert records[0]["evidence_warning"] == "source_header_conflict"
    assert "unusable_reason" not in records[0]
    # Identity is the trusted filename; the body still supports a verbatim claim.
    result = parse_answer(json.dumps(draft()), "Quel est l'objet ?", records)
    assert result["status"] == "answered"
    assert result["sources"][0]["file"] == "Cir_2020_03_fr.pdf"
    # Ingestion-detected extraction conflicts remain unusable.
    conflicted = evidence_records([(Document(page_content=text, metadata={
        "source": "Cir_2020_03_fr.pdf", "page": 1, "extraction_conflict": True}), 0.8)])
    assert conflicted[0]["unusable_reason"] == "extraction_conflict"
    assert parse_answer(json.dumps(draft()), "Quel est l'objet ?", conflicted)["status"] == "insufficient_evidence"


def test_invalid_model_response_keeps_the_arabic_question_language():
    result = parse_answer("not JSON", "ما هو تاريخها؟", EVIDENCE)
    assert result["answer"].startswith("المقاطع")
    assert result["sources"] == []


def test_partial_answer_keeps_supported_facts_and_uses_a_trusted_gap_message():
    value = draft()
    value.update(status="partial_answer", message="Je n’ai pas trouvé la date demandée dans ces passages.")
    result = parse_answer(json.dumps(value), "Quel est l'objet et la date ?", EVIDENCE)
    assert result["status"] == "partial_answer"
    assert "voyages d'affaires. [1]" in result["answer"]
    # Model-chosen partial without dropped claims: no stock incompleteness footer.
    assert "une partie de la demande" not in result["answer"]
    assert result["sources"][0]["excerpt"] == EVIDENCE[0]["text"]


@pytest.mark.parametrize("change", ["no_claims", "unknown_unit"])
def test_partial_answer_does_not_bypass_evidence_requirements(change):
    value = draft()
    value.update(status="partial_answer", message="La date n’est pas établie.")
    if change == "no_claims":
        value["claims"] = []
    elif change == "unknown_unit":
        value["claims"][0]["cites"] = ["E1.9"]
    result = parse_answer(json.dumps(value), "Quel est l'objet ?", EVIDENCE)
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_graph_relationship_note_is_surfaced_for_drafting():
    from langchain_core.documents import Document
    from answer_contract import evidence_records

    records = evidence_records([(Document(
        page_content="abroge la circulaire n° 2001-11",
        metadata={
            "source": "Cir_2018_09_fr.pdf",
            "page": 15,
            "temporal_relation": "ABROGATES",
            "temporal_source_id": "cir:2018:9",
            "temporal_target_id": "cir:2001:11",
            "temporal_verification": "VERIFIED_RELATIONSHIP_ONLY",
        },
    ), 0.9)])
    assert records[0]["relationship_note"].startswith("cir:2018:9 ABROGATES cir:2001:11")
    assert "not proof" in records[0]["relationship_note"]


def test_graph_supersession_keeps_both_sides_and_marks_roles():
    from langchain_core.documents import Document
    from answer_contract import evidence_records, _annotate_supersession

    docs = [
        (
            Document(
                page_content="Horaire: 7h00 a 13h00 pendant la seance unique.",
                metadata={"source": "Cir_2016_01_fr.pdf", "page": 2},
            ),
            0.9,
        ),
        (
            Document(
                page_content="Horaire: 8h00 a 14h00 pendant la seance unique.",
                metadata={
                    "source": "Cir_2021_03_fr.pdf",
                    "page": 2,
                    "temporal_relation": "ABROGATES",
                    "temporal_source_id": "cir:2021:3",
                    "temporal_target_id": "cir:2016:1",
                },
            ),
            0.85,
        ),
    ]
    records = _annotate_supersession(evidence_records(docs))
    by_source = {r["source"]: r for r in records}
    assert by_source["Cir_2021_03_fr.pdf"]["graph_role"] == "successor"
    assert by_source["Cir_2016_01_fr.pdf"]["graph_role"] == "superseded"
    assert "SUPERSEDED" in by_source["Cir_2016_01_fr.pdf"]["graph_guidance"]
    assert records[0]["source"] == "Cir_2021_03_fr.pdf"


def test_jsonl_amends_edge_marks_successor_for_topical_drafting():
    from langchain_core.documents import Document
    from answer_contract import evidence_records, _annotate_supersession

    docs = [
        (
            Document(
                page_content="La duree du travail est fixee a sept heures.",
                metadata={"source": "Cir_2016_08_fr.pdf", "page": 5},
            ),
            0.9,
        ),
        (
            Document(
                page_content="Les dispositions de l'article 5 de la circulaire 2016-08 sont modifiees.",
                metadata={
                    "source": "Cir_2020_03_fr.pdf",
                    "page": 2,
                    "temporal_relation": "AMENDS",
                    "temporal_source_id": "cir:2020:3",
                    "temporal_target_id": "cir:2016:8",
                },
            ),
            0.8,
        ),
    ]
    records = _annotate_supersession(evidence_records(docs))
    by_source = {r["source"]: r for r in records}
    assert by_source["Cir_2020_03_fr.pdf"]["graph_role"] == "successor"
    assert by_source["Cir_2016_08_fr.pdf"]["graph_role"] == "superseded"
    assert "AMENDS" in by_source["Cir_2016_08_fr.pdf"]["graph_guidance"]
    assert "successor" in by_source["Cir_2020_03_fr.pdf"]["graph_guidance"].casefold()


def test_unverified_temporal_scope_cannot_be_presented_as_a_complete_answer():
    result = parse_answer(json.dumps(draft()), "What does this say and is it still in force?",
                          EVIDENCE, temporal_unverified=True)
    assert result["status"] == "partial_answer"
    assert result["answer"].startswith("This is the latest value supported by the cited passages")
    assert "currently in force" in result["answer"]
    assert "could not be fully confirmed" in result["answer"]
    assert result["sources"]


def test_generation_binds_partial_answer_scope_and_literal_reference_context():
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from answer_contract import generate_grounded_answer

    def respond(prompt):
        messages = prompt.to_messages()
        if "Select evidence for" in messages[0].content:
            return AIMessage(content=json.dumps({"decision": "partial", "reason": "", "evidence_ids": ["E1"]}))
        assert "Unverified temporal scope: True" in messages[0].content
        assert '{"previous": "message"}' in messages[1].content
        return AIMessage(content=json.dumps(draft()))

    document = Document(page_content=EVIDENCE[0]["text"], metadata={
        "source": EVIDENCE[0]["source"], "page": 1, "pages": [1]})
    result = generate_grounded_answer(RunnableLambda(respond), "Is it still in force?",
        [(document, 0.8)], '{"previous": "message"}', temporal_unverified=True)
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["page"] == 1


def test_all_cited_units_on_a_shared_page_are_preserved():
    evidence = [{**EVIDENCE[0], "text": EVIDENCE[0]["text"] + "\nTunis, le 04 février 2020."}]
    value = draft()
    value["claims"].append({"text": "L’en-tête porte la date du 04 février 2020.",
        "cites": ["E1.2"]})
    result = parse_answer(json.dumps(value), "Quel est l'objet et la date de l'en-tête ?", evidence)
    assert len(result["sources"]) == 1
    assert "voyages d'affaires" in result["sources"][0]["excerpt"]
    assert "Tunis, le 04 février 2020." in result["sources"][0]["excerpt"]
    assert result["answer"].count("[1]") == 2


def _one_claim(text, cites=("E1.1",)):
    return json.dumps({"status": "answered", "message": "", "claims": [{"text": text, "cites": list(cites)}]})


def _page(text, source="Cir_2020_03_fr.pdf", page=3, language="fr"):
    return [{"evidence_id": "E1", "source": source, "page": page, "score": 0.9, "text": text, "language": language}]


@pytest.mark.parametrize("claim, accepted", [
    ("Le plafond est de 50 000 D par année civile.", True),       # French thousands with a space
    ("Le plafond est de 50.000 D par année civile.", True),       # the source's own notation
    ("Le plafond est de 50,000 D par année civile.", False),      # French comma: 50 dinars
])
def test_numbers_are_compared_as_values_in_french_notation(claim, accepted):
    evidence = _page("Le plafond est fixé à cinquante mille dinars (50.000 D) par année civile.")
    result = parse_answer(_one_claim(claim), "Quel est le plafond ?", evidence)
    assert (result["status"] == "answered") is accepted


def test_english_answer_reads_numbers_the_english_way_over_a_french_page():
    evidence = _page("Encours de la dette extérieure à long terme : 75.966 MDT à fin 2024.")
    result = parse_answer(_one_claim("Long-term external debt stood at 75,966 MDT at the end of 2024."),
                          "What was the long-term external debt at the end of 2024?", evidence)
    # Also: English words ("debt", "stood") cannot be looked up on a French page.
    assert result["status"] == "answered"


def test_picture_reading_number_needs_the_pictures_own_words():
    from answer_evidence import confirmed_numbers

    page = ("REPARTITION DES REQUETES (en %)\n"
            "[Mots de l'image] 32,3 39,1 9,7 Tunis Nabeul Autres\n"
            "[Lecture de l'image] Tunis 391\n"
            "[Lecture de l'image] Nabeul 9,7")
    assert confirmed_numbers("[Lecture de l'image] Tunis 391", page) == set()
    assert confirmed_numbers("[Lecture de l'image] Nabeul 9,7", page) == {"9.7"}
    assert confirmed_numbers("Le délai réglementaire est de 2 mois.", page) == {"2"}


def test_table_or_page_reference_is_not_an_unsupported_number():
    evidence = _page("Tableau 4-1 : Indicateurs\nPIB aux prix courants — 2025: 172.709", source="RA_2025_fr.pdf", page=120)
    for claim in ("Selon le tableau 4-1, le PIB aux prix courants en 2025 est de 172 709 MDT.",
                  "Le PIB aux prix courants en 2025 est de 172.709 MDT (page 120)."):
        assert parse_answer(_one_claim(claim, ("E1.2",)), "Quel est le PIB en 2025 ?", evidence)["status"] == "answered"
    # A table the page does not have is still an unsupported number.
    wrong = _one_claim("Selon le tableau 9-9, le PIB en 2025 est de 172 709 MDT.", ("E1.2",))
    assert parse_answer(wrong, "Quel est le PIB en 2025 ?", evidence)["status"] == "insufficient_evidence"


def test_arabic_no_longer_valid_is_not_a_claim_that_nothing_amended_the_text():
    from answer_evidence import claim_asserts_unsupported_negative_amendment

    assert not claim_asserts_unsupported_negative_amendment("لم تعد صالحة للصرف بعد 19 فيفري 2018")
    assert claim_asserts_unsupported_negative_amendment("لم يعدل أي نص لاحق هذا المنشور")


def test_a_broad_answer_may_have_more_than_eight_claims():
    lines = [f"Règle {n} : le bureau de change tient le registre {n}." for n in range(1, 11)]
    claims = [{"text": f"Le bureau de change tient le registre {n}.", "cites": [f"E1.{n}"]} for n in range(1, 11)]
    result = parse_answer(json.dumps({"status": "answered", "message": "", "claims": claims}),
                          "Quelles sont les règles des bureaux de change ?", _page("\n".join(lines)))
    assert result["status"] == "answered"
    assert result["answer"].count("[1]") == 10


@pytest.mark.parametrize("question, language", [
    ("During Ramadan single session, what are the interbank FX market hours?", "en"),
    ("Tunisia inflation rate at end of 2025?", "en"),
    ("and in English please, what about the installation allowance?", "en"),
    ("Compare l'inflation en 2023, 2024 et 2025.", "fr"),
    ("chnowa el plafond mta3 allocation voyage d'affaires l'entreprise?", "fr"),
    ("ما هو المبلغ الأقصى لمنحة الإقامة؟", "ar"),
])
def test_answer_language_is_read_from_the_whole_question(question, language):
    from answer_contract import language_of

    assert language_of(question) == language


def test_before_circular_x_does_not_require_evidence_from_x():
    from answer_evidence import direct_identity

    assert direct_identity("Avant la circulaire 2020-03, quel était le plafond ?") is None
    assert direct_identity("Selon la circulaire 2020-03, quel est le plafond ?")["number"] == 3


def test_selector_is_shown_the_line_that_answers_even_on_a_long_page():
    from answer_draft import _relevant_units

    page = "\n".join([
        "L'intermédiaire agréé qui procède à l'annulation du règlement ainsi que le titulaire de l'allocation sont tenus d'en informer l'intermédiaire agréé domiciliataire de l'allocation dans les meilleurs délais.",
        "SECTION 2 : ALLOCATION POUR VOYAGES D'AFFAIRES",
        "« AUTRES ACTIVITÉS »",
        "Article 8 : Les personnes physiques et morales résidentes ne disposant pas d'Allocations pour Voyages d'Affaires peuvent bénéficier d'une allocation pour voyages d'affaires « autres activités » ouverte auprès d'un intermédiaire agréé de leur choix, sur présentation des pièces justificatives prévues par la présente circulaire et dans les conditions fixées ci-après.",
        "Article 9 : Le montant de l'Allocation pour Voyages d'Affaires « autres activités » est fixé à huit pourcent (8%) du chiffre d'affaires avec un plafond de cinquante mille dinars (50.000D) par année civile.",
        "Article 10 : Lorsqu'à l'ouverture ou à la reconduction de cette allocation, la déclaration fiscale faisant ressortir le chiffre d'affaires de l'année précédente n'est pas encore disponible, l'allocation est ouverte sur la base de la dernière déclaration disponible, sous réserve de régularisation.",
        "Article 11 : Les intermédiaires agréés peuvent ouvrir des dossiers d'Allocations pour Voyages d'Affaires « autres activités » au profit des personnes qui en font la demande, dans la limite des montants prévus par la présente circulaire et après vérification des pièces.",
    ])
    indices, spans = _relevant_units("Quel était le plafond de l'allocation voyages d'affaires autres activités ?", page, 1000)
    shown = " ".join(page[spans[i][0]:spans[i][1]] for i in indices)
    assert "50.000D" in shown


def test_successor_order_only_when_the_replaced_text_is_in_the_evidence():
    from answer_draft import _annotate_supersession

    newest = {"evidence_id": "E1", "source": "Cir_2026_04_fr.pdf", "page": 2, "text": "règle 2026", "score": 0.9}
    successor = {"evidence_id": "E2", "source": "Cir_2018_13_fr.pdf", "page": 2, "text": "règle 2018", "score": 0.8,
                 "temporal_relation": "ABROGATES", "temporal_source_id": "cir:2018:13", "temporal_target_id": "cir:2017:9"}
    # 2017-09 is not in the evidence: nothing is reordered, 2026-04 stays first.
    assert [r["evidence_id"] for r in _annotate_supersession([newest, successor])] == ["E1", "E2"]
    replaced = {"evidence_id": "E3", "source": "Cir_2017_09_fr.pdf", "page": 2, "text": "règle 2017", "score": 0.95}
    annotated = _annotate_supersession([replaced, newest, successor])
    assert annotated[0]["evidence_id"] == "E2" and annotated[0]["graph_role"] == "successor"
    assert annotated[-1]["graph_role"] == "superseded"
