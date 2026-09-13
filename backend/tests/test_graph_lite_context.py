from langchain_core.documents import Document

from conversation import _answer_results
from retrieval_selection import diversify_ranked_pages


def _doc(name, **metadata):
    return Document(page_content=name, metadata=metadata)


def test_graph_lite_reserves_two_slots_without_reducing_normal_top_five():
    ordinary = [(_doc(f"ordinary-{i}", source=f"o{i}.pdf"), 10.0 - i) for i in range(8)]
    graph = [
        (_doc(f"graph-{i}", temporal_verification="VERIFIED_RELATIONSHIP_ONLY"), 1.0 - i / 10)
        for i in range(3)
    ]

    selected = _answer_results(ordinary, graph_results=graph)

    assert [document.page_content for document, _score in selected] == [
        "graph-0",
        "graph-1",
        "ordinary-0",
        "ordinary-1",
        "ordinary-2",
        "ordinary-3",
        "ordinary-4",
    ]


def test_answer_budget_keeps_two_same_page_graph_quotes():
    ordinary = [(_doc(f"ordinary-{i}", source=f"o{i}.pdf", page=i + 1), 10.0 - i) for i in range(5)]
    graph = [
        (
            _doc(
                "amends-quote",
                source="Cir_2018_09_fr.pdf",
                page=15,
                page_label=15,
                temporal_relation="AMENDS",
                temporal_verification="VERIFIED_RELATIONSHIP_ONLY",
            ),
            2.0,
        ),
        (
            _doc(
                "abrogates-quote",
                source="Cir_2018_09_fr.pdf",
                page=15,
                page_label=15,
                temporal_relation="ABROGATES",
                temporal_verification="VERIFIED_RELATIONSHIP_ONLY",
            ),
            1.0,
        ),
    ]

    selected = _answer_results(ordinary, graph_results=graph)

    assert [document.page_content for document, _score in selected[:2]] == [
        "amends-quote",
        "abrogates-quote",
    ]


def test_diversify_uses_page_label_when_zero_based_page_key_is_absent():
    ranked = [
        (_doc("first", source="Cir_2018_09_fr.pdf", page_label=2), 1.0),
        (_doc("second", source="Cir_2018_09_fr.pdf", page_label=15), 0.9),
        (_doc("dup-page", source="Cir_2018_09_fr.pdf", page_label=2), 0.8),
    ]

    kept = diversify_ranked_pages(ranked)

    assert [document.page_content for document, _score in kept] == ["first", "second"]
