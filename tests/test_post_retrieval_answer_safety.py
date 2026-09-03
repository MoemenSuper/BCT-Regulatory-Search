from experiments.post_retrieval_answer_safety import (
    apply_answer_safety,
    compose_incomplete_answer_from_claims,
    recompose_records,
)


def _evidence(source="Cir_2021_06_fr.pdf", text="plafond 870, échéance 31 août"):
    return [
        {
            "evidence_id": "E1",
            "source": source,
            "page": 2,
            "text": text,
        }
    ]


def _response(
    *,
    answer="Le plafond est 870 et l'échéance est le 31 août.",
    claim="Le plafond est 870 et l'échéance est le 31 août.",
    source="Cir_2021_06_fr.pdf",
):
    return {
        "status": "answered",
        "answer": answer,
        "claims": [{"text": claim, "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": source, "page": 2}],
    }


def _state(**overrides):
    value = {
        "scope": "bct_regulatory_or_financial",
        "temporal_state": "not_current_or_future",
        "ambiguity": "sufficiently_specific",
        "missing_detail": "",
    }
    value.update(overrides)
    return value


def test_generic_authorization_validity_requests_clarification():
    guarded, audit = apply_answer_safety(
        question="Combien de temps mon autorisation reste-t-elle valable ?",
        language="fr",
        response={"status": "insufficient_evidence", "answer": "Inconnu.", "claims": [], "citations": []},
        evidence=_evidence(),
        query_state=_state(),
    )

    assert guarded["status"] == "clarification_needed"
    assert audit["action"] == "request_clarification_missing_operation"


def test_generic_arabic_reporting_deadline_requests_clarification():
    guarded, audit = apply_answer_safety(
        question="ما هو آخر أجل لإرسال التصريح إلى البنك المركزي؟",
        language="ar",
        response=_response(),
        evidence=_evidence(),
        query_state=_state(),
    )

    assert guarded["status"] == "clarification_needed"
    assert audit["action"] == "request_clarification_missing_operation"


def test_unsupported_current_status_abstains_even_after_model_clarification():
    guarded, audit = apply_answer_safety(
        question="Quel est aujourd'hui le plafond en vigueur pour une allocation de voyage d'affaires ?",
        language="fr",
        response={"status": "clarification_needed", "answer": "Précisez.", "claims": [], "citations": []},
        evidence=_evidence(text="Une ancienne circulaire fixe un plafond."),
        query_state=_state(),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "fail_closed_current_status_not_established"


def test_explicit_instrument_claim_must_cite_that_instrument():
    guarded, audit = apply_answer_safety(
        question="Selon la circulaire n°2021-06, quel est le plafond ?",
        language="fr",
        response=_response(source="Cir_2024_11_fr.pdf"),
        evidence=_evidence(source="Cir_2024_11_fr.pdf"),
        query_state=_state(),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "fail_closed_requested_instrument_mismatch"


def test_analogous_version_is_allowed_only_as_labeled_context_beside_primary_claim():
    evidence = [
        *_evidence(text="La circulaire n°2021-06 fixe le plafond à 870."),
        {
            "evidence_id": "E2",
            "source": "Cir_2024_11_fr.pdf",
            "page": 2,
            "text": "La circulaire n°2024-11 fixe le plafond à 1030.",
        },
    ]
    response = {
        "status": "answered",
        "answer": (
            "La circulaire 2021-06 fixe 870. À titre de comparaison, "
            "la circulaire 2024-11 fixe 1030."
        ),
        "claims": [
            {"text": "La circulaire 2021-06 fixe 870.", "evidence_ids": ["E1"]},
            {
                "text": "À titre de comparaison, la circulaire 2024-11 fixe 1030.",
                "evidence_ids": ["E2"],
            },
        ],
        "citations": [
            {"evidence_id": "E1", "source": "Cir_2021_06_fr.pdf", "page": 2},
            {"evidence_id": "E2", "source": "Cir_2024_11_fr.pdf", "page": 2},
        ],
    }

    guarded, audit = apply_answer_safety(
        question="Selon la circulaire n°2021-06, quel est le plafond ?",
        language="fr",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert audit["action"] == "keep", audit
    assert guarded is response


def test_structural_claim_link_is_not_enough_when_evidence_is_inadequate():
    guarded, audit = apply_answer_safety(
        question="Selon la circulaire n°2021-06, quel est le plafond ?",
        language="fr",
        response=_response(answer="Le plafond est 870.", claim="Le plafond est 870."),
        evidence=_evidence(text="Article 2. La circulaire entre en vigueur."),
        query_state=_state(),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "fail_closed_inadequate_linked_evidence"


def test_spaced_thousands_match_dot_thousands_with_currency_suffix():
    response = _response(
        answer="Le seuil est de 20 000 dinars et le taux maximal est de 50%.",
        claim="Le seuil est de 20 000 dinars et le taux maximal est de 50%.",
        source="Cir_2020_02_fr.pdf",
    )
    evidence = _evidence(
        source="Cir_2020_02_fr.pdf",
        text="Le taux maximal est 50% lorsque la valeur dépasse 20.000D.",
    )

    guarded, audit = apply_answer_safety(
        question="Quel est le taux maximal au-dessus du seuil ?",
        language="fr",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert audit["action"] == "keep", audit
    assert guarded is response


def test_claim_cannot_invert_the_legal_role_of_an_advisory_opinion():
    guarded, audit = apply_answer_safety(
        question="Les articles 2 et 4 ont-ils été abrogés ?",
        language="fr",
        response={
            "status": "answered",
            "answer": "Les articles 2 et 4 ont été abrogés par l'avis 2020-15.",
            "claims": [
                {
                    "text": "Les articles 2 et 4 ont été abrogés par l'avis 2020-15.",
                    "evidence_ids": ["E1"],
                }
            ],
            "citations": [
                {"evidence_id": "E1", "source": "Cir_2020_15_fr.pdf", "page": 2}
            ],
        },
        evidence=_evidence(
            source="Cir_2020_15_fr.pdf",
            text=(
                "Vu l'avis du Comité de contrôle de la conformité n°2020-15. "
                "Décide : Sont abrogées les dispositions des articles 2 et 4."
            ),
        ),
        query_state=_state(),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "fail_closed_legal_role_inversion"


def test_incomplete_rendering_is_completed_only_from_existing_linked_claims():
    response = {
        "status": "answered",
        "answer": "Les banques doivent prendre les mesures suivantes :",
        "claims": [
            {"text": "Approvisionner les distributeurs.", "evidence_ids": ["E1"]},
            {"text": "Maintenir les plateformes disponibles.", "evidence_ids": ["E1"]},
        ],
        "citations": [
            {"evidence_id": "E1", "source": "Note_2026_129_fr.pdf", "page": 1}
        ],
    }

    completed = compose_incomplete_answer_from_claims(response)

    assert completed is not response
    assert "Approvisionner les distributeurs." in completed["answer"]
    assert "Maintenir les plateformes disponibles." in completed["answer"]
    assert completed["claims"] == response["claims"]


def test_missing_material_subquestion_fails_closed_instead_of_looking_complete():
    guarded, audit = apply_answer_safety(
        question="Quels plafonds et quelles échéances s'appliquent ?",
        language="fr",
        response=_response(answer="Le plafond est 870.", claim="Le plafond est 870."),
        evidence=_evidence(text="plafond 870"),
        query_state=_state(),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "fail_closed_material_subquestion_unsupported"


def test_ambiguous_query_with_multiple_plausible_instruments_requests_detail():
    evidence = [
        *_evidence(source="Cir_2025_10_fr.pdf"),
        {
            "evidence_id": "E2",
            "source": "Cir_2018_14_fr.pdf",
            "page": 11,
            "text": "autres documents de transfert",
        },
    ]
    guarded, audit = apply_answer_safety(
        question="Quels documents dois-je fournir pour mon transfert ?",
        language="fr",
        response=_response(source="Cir_2025_10_fr.pdf"),
        evidence=evidence,
        query_state=_state(
            ambiguity="missing_discriminating_detail", missing_detail="type de transfert"
        ),
    )

    assert guarded["status"] == "clarification_needed"
    assert guarded["answer"].strip()
    assert "type de transfert" in guarded["answer"]
    assert audit["action"] == "request_clarification_multiple_instruments"


def test_classifier_detail_is_localized_in_the_clarification():
    guarded, _audit = apply_answer_safety(
        question="Quels documents dois-je fournir pour mon transfert ?",
        language="fr",
        response=_response(source="Cir_2025_10_fr.pdf"),
        evidence=[
            *_evidence(source="Cir_2025_10_fr.pdf"),
            {
                "evidence_id": "E2",
                "source": "Cir_2018_14_fr.pdf",
                "page": 11,
                "text": "autres documents de transfert",
            },
        ],
        query_state=_state(
            ambiguity="missing_discriminating_detail", missing_detail="type of transfer"
        ),
    )

    assert "type de transfert" in guarded["answer"]
    assert "type of transfer" not in guarded["answer"]


def test_current_claim_without_explicit_status_evidence_fails_closed():
    guarded, audit = apply_answer_safety(
        question="Quel est aujourd'hui le plafond en vigueur ?",
        language="fr",
        response=_response(
            answer="Le plafond en vigueur est 870.", claim="Le plafond en vigueur est 870."
        ),
        evidence=_evidence(text="Le plafond est 870. Entre en vigueur à sa publication."),
        query_state=_state(temporal_state="current_or_future"),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert guarded["answer"].strip()
    assert audit["action"] == "fail_closed_current_status_not_established"


def test_historical_in_force_question_with_date_is_not_treated_as_current():
    response = _response(
        answer="Après le 19 juin 2020, les articles 2 et 4 étaient abrogés.",
        claim="Après le 19 juin 2020, les articles 2 et 4 étaient abrogés.",
        source="Cir_2020_15_fr.pdf",
    )
    evidence = _evidence(
        source="Cir_2020_15_fr.pdf",
        text="Le 19 juin 2020, sont abrogées les dispositions des articles 2 et 4.",
    )

    guarded, audit = apply_answer_safety(
        question="Ces articles sont-ils restés en vigueur après le 19 juin 2020 ?",
        language="fr",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert audit["action"] == "keep", audit
    assert guarded is response


def test_cross_version_numeric_corroboration_is_not_treated_as_mixing():
    response = {
        "status": "answered",
        "answer": "L'horaire conventionnel est de 8h à 17h.",
        "claims": [
            {
                "text": "L'horaire conventionnel est de 8h à 17h.",
                "evidence_ids": ["E1", "E2"],
            }
        ],
        "citations": [
            {"evidence_id": "E1", "source": "Cir_2021_03_fr.pdf", "page": 2},
            {"evidence_id": "E2", "source": "Cir_2016_01_fr.pdf", "page": 2},
        ],
    }
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Cir_2021_03_fr.pdf",
            "page": 2,
            "text": "L'horaire conventionnel s'étend de 8h à 17h.",
        },
        {
            "evidence_id": "E2",
            "source": "Cir_2016_01_fr.pdf",
            "page": 2,
            "text": "L'horaire conventionnel s'étend de 8h à 17h.",
        },
    ]

    guarded, audit = apply_answer_safety(
        question="Quel est l'horaire conventionnel ?",
        language="fr",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert audit["action"] == "keep", audit
    assert guarded is response


def test_unlabelled_incompatible_versions_across_claims_fail_closed():
    response = {
        "status": "answered",
        "answer": "Le plafond d'hiver est 1030 et le plafond d'été est 1140.",
        "claims": [
            {"text": "Le plafond d'hiver est 1030.", "evidence_ids": ["E1"]},
            {"text": "Le plafond d'été est 1140.", "evidence_ids": ["E2"]},
        ],
        "citations": [
            {"evidence_id": "E1", "source": "Cir_2024_11_fr.pdf", "page": 2},
            {"evidence_id": "E2", "source": "Cir_2021_06_fr.pdf", "page": 2},
        ],
    }
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Cir_2024_11_fr.pdf",
            "page": 2,
            "text": "Le plafond d'hiver est 1030.",
        },
        {
            "evidence_id": "E2",
            "source": "Cir_2021_06_fr.pdf",
            "page": 2,
            "text": "Le plafond d'été est 1140.",
        },
    ]

    guarded, audit = apply_answer_safety(
        question="Quels plafonds s'appliquent aux fourrages d'hiver et d'été ?",
        language="fr",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "fail_closed_unlabelled_cross_claim_version_mixing"


def test_separately_labelled_versions_across_claims_remain_distinct_context():
    response = {
        "status": "answered",
        "answer": "Le taux est 2,75% selon la note 2019 et 2,5% selon la note 2020.",
        "claims": [
            {"text": "Le taux est 2,75%.", "evidence_ids": ["E1"]},
            {"text": "Le taux est 2,5%.", "evidence_ids": ["E2"]},
        ],
        "citations": [
            {"evidence_id": "E1", "source": "Note_2019_17_fr.pdf", "page": 1},
            {"evidence_id": "E2", "source": "Note_2020_01_fr.pdf", "page": 1},
        ],
    }
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Note_2019_17_fr.pdf",
            "page": 1,
            "text": "Note 2019-17 : le taux est 2,75%.",
        },
        {
            "evidence_id": "E2",
            "source": "Note_2020_01_fr.pdf",
            "page": 1,
            "text": "Note 2020-01 : le taux est 2,5%.",
        },
    ]

    guarded, audit = apply_answer_safety(
        question="Quels taux les deux notes indiquent-elles ?",
        language="fr",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert audit["action"] == "keep", audit
    assert guarded is response


def test_arabic_numeric_claim_with_inflection_and_exact_literals_is_adequate():
    response = {
        "status": "answered",
        "answer": "يحدد القرض في حدود 80% ويحل أجل سداده في 31 أوت 2018.",
        "claims": [
            {
                "text": "يحدد مبلغ القرض في حدود 80% من الكلفة المرجعية.",
                "evidence_ids": ["E1"],
            },
            {"text": "أجل سداده هو 31 أوت 2018.", "evidence_ids": ["E1"]},
        ],
        "citations": [
            {"evidence_id": "E1", "source": "Note_2018_03_ar.pdf", "page": 2}
        ],
    }
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Note_2018_03_ar.pdf",
            "page": 2,
            "text": (
                "يضبط مبلغ القروض في حدود 80% من الكلفة المرجعية "
                "ويحل أجل خلاصها في 31 أوت 2018."
            ),
        }
    ]

    guarded, audit = apply_answer_safety(
        question="ما مبلغ القرض وأجل سداده؟",
        language="ar",
        response=response,
        evidence=evidence,
        query_state=_state(),
    )

    assert audit["action"] == "keep", audit
    assert guarded is response


def test_historical_answer_is_not_blocked_by_query_state():
    response = _response()
    guarded, audit = apply_answer_safety(
        question="Quel plafond la circulaire n°2021-06 fixait-elle ?",
        language="fr",
        response=response,
        evidence=_evidence(),
        query_state=_state(scope="clearly_unrelated"),
    )

    assert guarded is response
    assert audit["action"] == "keep"


def test_false_out_of_scope_preemption_is_corrected_only_after_retrieval_abstains():
    response = {
        "status": "out_of_scope",
        "answer": "Ce sujet est hors périmètre.",
        "claims": [],
        "citations": [],
    }
    guarded, audit = apply_answer_safety(
        question="متى بدأ تداول الورقة النقدية السويسرية؟",
        language="ar",
        response=response,
        evidence=[],
        query_state=_state(scope="clearly_unrelated"),
        generated_status_before_query_state="insufficient_evidence",
        answer_path="retrieved_top5_abstention_then_query_state",
    )

    assert guarded["status"] == "insufficient_evidence"
    assert audit["action"] == "correct_false_query_state_preemption"

    untouched, untouched_audit = apply_answer_safety(
        question="Donne-moi une recette de gâteau.",
        language="fr",
        response=response,
        evidence=[],
        query_state=_state(scope="clearly_unrelated"),
        generated_status_before_query_state="insufficient_evidence",
        answer_path="retrieved_top5_abstention_then_query_state",
    )
    assert untouched is response
    assert untouched_audit["action"] == "keep_nonanswer"


def test_recomposition_preserves_untouched_record_objects():
    original = [{"id": "keep"}, {"id": "change"}]
    replacement = {"id": "change", "response": {"status": "insufficient_evidence"}}

    composed = recompose_records(original, {"change": replacement})

    assert composed[0] is original[0]
    assert composed[1] is replacement
