from pathlib import Path

from conversation_memory import ConversationStore, new_memory_state


def test_full_history_is_separate_from_bounded_llm_memory(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_turns=2)
    conversation_id = store.create()
    state = new_memory_state()

    for index in range(4):
        state["turns"].append(
            {
                "user_message": f"Question {index}",
                "standalone_query": f"Question {index}",
                "answer": f"Answer {index}",
                "sources": [],
            }
        )
        store.save_with_turn(
            conversation_id,
            state,
            question=f"Question {index}",
            answer=f"Answer {index}",
            sources=[],
            profile="cloud",
            answer_status="answered",
        )

    assert len(store.load(conversation_id)["turns"]) == 2
    transcript = store.transcript(conversation_id)
    assert transcript is not None
    assert [turn["question"] for turn in transcript["turns"]] == [
        "Question 0",
        "Question 1",
        "Question 2",
        "Question 3",
    ]
    history = store.list_conversations()
    assert history[0]["conversation_id"] == conversation_id
    assert history[0]["title"] == "Question 0"
    assert history[0]["last_question"] == "Question 3"
    assert history[0]["turn_count"] == 4
    assert store.transcript(conversation_id)["title"] == "Question 0"


def test_history_title_is_summarized_from_first_question(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation_id = store.create()
    state = new_memory_state()
    question = "Quelles sont les heures d'ouverture des guichets ?"
    state["turns"].append(
        {
            "user_message": question,
            "standalone_query": question,
            "answer": "Les guichets ouvrent à 8h.",
            "sources": [],
        }
    )
    store.save_with_turn(
        conversation_id,
        state,
        question=question,
        answer="Les guichets ouvrent à 8h.",
        sources=[],
        profile="cloud",
        answer_status="answered",
    )
    history = store.list_conversations()
    assert history[0]["title"] == "Heures d'ouverture des guichets"
    assert history[0]["title"] != question
    assert store.transcript(conversation_id)["title"] == "Heures d'ouverture des guichets"


def test_empty_sessions_are_not_shown_in_history(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    store.create()
    assert store.list_conversations() == []


def test_rename_and_delete_conversation(tmp_path: Path):
    import pytest

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation_id = store.create()
    store.save_with_turn(conversation_id, new_memory_state(), question="Question initiale ?", answer="A")

    store.rename(conversation_id, "  Titre choisi  ")
    assert store.list_conversations()[0]["title"] == "Titre choisi"
    assert store.transcript(conversation_id)["title"] == "Titre choisi"
    # A later turn must not overwrite the user's title.
    store.save_with_turn(conversation_id, new_memory_state(), question="Autre question ?", answer="B")
    assert store.transcript(conversation_id)["title"] == "Titre choisi"

    store.delete(conversation_id)
    assert store.load(conversation_id) is None
    assert store.transcript(conversation_id) is None
    assert store.list_conversations() == []
    with pytest.raises(KeyError):
        store.rename(conversation_id, "x")
    with pytest.raises(KeyError):
        store.delete(conversation_id)
