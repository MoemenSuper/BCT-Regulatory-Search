import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import conversation
from graph_contract import is_temporal_rule_query


PLAFOND_QUESTION = (
    "Quel est aujourd'hui le plafond en vigueur pour une allocation de voyage d'affaires ?"
)
EXPORTER_QUOTE = (
    "25 % des recettes d'exportation rapatriées, dans la limite de 500 000 dinars par année civile"
)
OTHER_QUOTE = (
    "8 % du chiffre d'affaires hors taxes de l'année précédente, dans la limite de 50 000 dinars par année civile"
)


def _page(source, year_blob):
    return Document(
        page_content=(
            f"{year_blob}\n"
            f"Allocation de voyage d'affaires Exportateurs : {EXPORTER_QUOTE}.\n"
            f"Allocation de voyage d'affaires Autres activités : {OTHER_QUOTE}."
        ),
        metadata={"source": source, "page": 2, "pages": [2]},
    )


def _claims_draft():
    return {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": (
                    "Le plafond pour l'allocation de voyage d'affaires Exportateurs "
                    "est de 500 000 dinars par année civile."
                ),
                "quotes": [{"evidence_id": "E1", "quote": EXPORTER_QUOTE}],
            },
            {
                "text": (
                    "Le plafond pour l'allocation de voyage d'affaires Autres activités "
                    "est de 50 000 dinars par année civile."
                ),
                "quotes": [{"evidence_id": "E1", "quote": OTHER_QUOTE}],
            },
        ],
    }


def test_business_travel_plafond_question_is_treated_as_currentness():
    assert is_temporal_rule_query(PLAFOND_QUESTION) is True
    assert is_temporal_rule_query(
        "Quel est le plafond pour une allocation de voyage d'affaires ?"
    ) is False


def test_business_travel_plafond_gives_latest_supported_value_without_claiming_it_is_current(
    monkeypatch,
):
    older = _page("Cir_2016_01_fr.pdf", "Circulaire 2016-01")
    later = _page("Cir_2020_03_fr.pdf", "Circulaire 2020-03")
    captured = {}

    class Backend:
        def retrieve(self, query):
            captured["retrieval_query"] = query
            return [(older, 0.96), (later, 0.91)]

    abstain = {
        "status": "insufficient_evidence",
        "message": "",
        "claims": [],
    }
    monkeypatch.setattr(conversation, "create_llm", lambda *_args, **_kwargs: FakeListChatModel(
        responses=[json.dumps({"decision": "partial", "reason": "Latest same-scope document is E1", "evidence_ids": ["E1"]}),
                   json.dumps(abstain), json.dumps(_claims_draft())]
    ))
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "plafond allocation voyage affaires",
            "new_topic": "allocation voyage affaires",
            "current_topic": "allocation voyage affaires",
        },
    )

    result = conversation.chat(
        PLAFOND_QUESTION,
        {"topics": [], "turns": []},
        retrieval_backend=Backend(),
    )

    answer = result["answer"]
    assert captured["retrieval_query"] == "plafond allocation voyage affaires"
    assert result["status"] == "partial_answer"
    assert "500 000" in answer
    assert "50 000" in answer
    assert "actuellement en vigueur" in answer.casefold() or "validité présente" in answer.casefold()
    assert "n’a pas pu être pleinement confirmée" in answer or "pas pu être pleinement confirmée" in answer
    assert "ne permettent pas une réponse suffisamment étayée" not in answer
    sources = [source["file"] for source in result["sources"]]
    assert sources[0] == "Cir_2020_03_fr.pdf"
