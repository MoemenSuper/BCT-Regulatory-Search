from experiments.conversation_routing_experiment import memory_fixture, score_case


def test_score_case_requires_both_intent_and_answer_bearing_rewrite_terms():
    case = {
        "expected_intent": "FOLLOW_UP",
        "required_rewrite_groups": [["2019"], ["07"], ["deadline", "délai"]],
    }

    assert score_case(
        case,
        {
            "intent": "FOLLOW_UP",
            "rewrite_query": "deadline in Circular 2019-07",
        },
    )["passed"] is True
    assert score_case(
        case,
        {"intent": "FOLLOW_UP", "rewrite_query": "deadline in a circular"},
    )["passed"] is False


def test_ambiguous_fixture_has_no_current_topic_but_retains_both_topics():
    memory = memory_fixture("two_no_current")

    assert memory["current_topic"] is None
    assert memory["topics"] == ["Circular 2019-07", "Circular 2025-17"]
    assert len(memory["turns"]) == 2
