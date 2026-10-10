from extras.conversation_memory import ConversationStore, new_memory_state

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
        }
        for index in range(5)
    ]

    store.save(conversation_id, state, user_id=OWNER)

    assert [
        turn["user_message"] for turn in store.load(conversation_id, user_id=OWNER)["turns"]
    ] == ["question 2", "question 3", "question 4"]


def test_older_database_with_graph_trace_column_keeps_working(tmp_path):
    import sqlite3

    database = tmp_path / "conversations.sqlite3"
    ConversationStore(database)  # create the current schema, then give it the old column back
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE conversation_turns ADD COLUMN graph_trace_json TEXT NOT NULL DEFAULT '{}'")
    store = ConversationStore(database)
    conversation_id = store.create(OWNER)
    turn_id = store.save_with_turn(conversation_id, new_memory_state(), user_id=OWNER, question="Q", answer="A")
    assert store.get_turn(conversation_id, turn_id, user_id=OWNER)["answer"] == "A"
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(conversation_turns)")}
    assert "graph_trace_json" not in columns
