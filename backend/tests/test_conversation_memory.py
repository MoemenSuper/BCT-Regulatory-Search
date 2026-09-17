from conversation_memory import ConversationStore, new_memory_state

OWNER = "user-a"


def test_conversation_state_survives_store_reopen(tmp_path):
    database = tmp_path / "conversations.sqlite3"
    first_store = ConversationStore(database)
    conversation_id = first_store.create(OWNER)
    state = first_store.load(conversation_id, user_id=OWNER)
    state["current_topic"] = "Circular 2019-07"
    state["topics"] = ["Circular 2019-07"]
    state["turns"] = [
        {
            "user_message": "What does Circular 2019-07 change?",
            "standalone_query": "changes made by Circular 2019-07",
            "answer": "It changes several provisions.",
            "sources": [{"file": "Cir_2019_07_fr.pdf", "page": 3}],
            "graph_trace": {"status": "EXPANDED"},
        }
    ]
    first_store.save(conversation_id, state, user_id=OWNER)

    reopened = ConversationStore(database)

    assert reopened.load(conversation_id, user_id=OWNER) == state


def test_conversation_store_keeps_sessions_isolated(tmp_path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    first = store.create(OWNER)
    second = store.create(OWNER)
    first_state = store.load(first, user_id=OWNER)
    first_state["current_topic"] = "Circular 2019-07"
    store.save(first, first_state, user_id=OWNER)

    assert store.load(first, user_id=OWNER)["current_topic"] == "Circular 2019-07"
    assert store.load(second, user_id=OWNER) == new_memory_state()
    assert store.load("missing-session", user_id=OWNER) is None


def test_conversation_store_caps_history_to_recent_turns(tmp_path):
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_turns=3)
    conversation_id = store.create(OWNER)
    state = store.load(conversation_id, user_id=OWNER)
    state["turns"] = [
        {
            "user_message": f"question {index}",
            "standalone_query": f"query {index}",
            "answer": f"answer {index}",
            "sources": [],
            "graph_trace": {"status": "NOT_REQUESTED"},
        }
        for index in range(5)
    ]

    store.save(conversation_id, state, user_id=OWNER)

    assert [
        turn["user_message"] for turn in store.load(conversation_id, user_id=OWNER)["turns"]
    ] == ["question 2", "question 3", "question 4"]
