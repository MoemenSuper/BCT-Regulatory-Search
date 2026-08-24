from experiments.gold_evidence_answer_experiment import automatic_answer_audit


def test_relevant_audit_checks_literals_and_explicit_citation():
    case = {
        "relevant": True,
        "query": "Quel est le taux?",
        "expected_answer": "Le taux est 6% et le code ABC-12.",
        "evidence_quote": "Le taux est fixé à 6% pour le code ABC-12.",
        "expected_source": "Cir_2026_01_fr.pdf",
        "expected_page": 2,
    }

    audit = automatic_answer_audit(
        case,
        "Le taux est 6% pour ABC-12 (Cir_2026_01_fr.pdf, page 2).",
    )

    assert audit["number_recall"] == 1.0
    assert audit["identifier_recall"] == 1.0
    assert audit["unsupported_number_literals"] == []
    assert audit["expected_source_present"] is True
    assert audit["expected_page_cited"] is True


def test_negative_audit_does_not_treat_unsupported_answer_as_safe():
    case = {
        "relevant": False,
        "expected_behavior": "abstain",
    }

    unsafe = automatic_answer_audit(case, "Le taux sera 7% selon Future.pdf, page 1.")
    safe = automatic_answer_audit(case, "L'information n'a pas été trouvée dans le contexte fourni.")

    assert unsafe["preliminary_safe_response"] is False
    assert safe["preliminary_safe_response"] is True
    assert safe["manual_review_required"] is True
