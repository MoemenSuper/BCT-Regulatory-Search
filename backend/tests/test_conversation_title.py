from conversation_memory import summarize_conversation_title


def test_summarize_strips_french_question_framing():
    assert (
        summarize_conversation_title("Quelles sont les heures d'ouverture des guichets ?")
        == "Heures d'ouverture des guichets"
    )


def test_summarize_strips_relation_question_framing():
    title = summarize_conversation_title(
        "Quelle est la relation entre la circulaire 2018-09 et la circulaire 2017-08 ?"
    )
    assert title.startswith("Relation entre la circulaire")
    assert "2018-09" in title
    assert "?" not in title


def test_summarize_truncates_long_titles():
    long_question = "Pouvez-vous expliquer " + ("détail " * 40) + "de la règle ?"
    title = summarize_conversation_title(long_question, max_length=60)
    assert len(title) <= 60
    assert title.endswith("…")
