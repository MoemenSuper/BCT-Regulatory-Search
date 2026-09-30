"""Synthetic answer-layer regressions; no benchmark IDs or expected answers."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from answer_contract import generate_grounded_answer, parse_answer


def record(source="Cir_2022_41_fr.pdf", text="Le plafond est de 320 dinars.", eid="E1"):
    return dict(evidence_id=eid, source=source, page=2, score=0.9, text=text)


def draft(text="Le plafond est de 320 dinars.", unit=1, eid="E1", **changes):
    """One claim citing unit `unit` of evidence `eid` (pointer citation)."""
    return dict(status="answered", message="", claims=[dict(
        text=text, cites=[f"{eid}.{unit}"]
    )], **changes)


def parse(value, evidence, question="Quel est le plafond ?", **kwargs):
    return parse_answer(json.dumps(value, ensure_ascii=False), question, evidence, **kwargs)


def test_requested_circular_cannot_be_replaced_by_a_similar_document():
    evidence = [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]
    result = parse(draft("Le plafond est de 640 dinars.", eid="E2"), evidence,
                   "Selon la circulaire 2022-41, quel est le plafond ?")
    assert result["status"] == "insufficient_evidence"


@pytest.mark.parametrize("claim,quote", [
    ("Le plafond est de 640 dinars.", "Le plafond est de 320 dinars."),
    ("Le code est 0912.", "Le code est 0192."),
    ("La durée est de deux mois (60 jours).", "La durée est de deux mois."),
])
def test_claim_numbers_must_be_supported_by_its_own_quote(claim, quote):
    result = parse(draft(claim), [record(text=quote), record(text=claim, eid="E2")])
    assert result["status"] == "insufficient_evidence"


def test_partial_message_cannot_smuggle_an_unquoted_legal_answer():
    value = draft()
    value.update(status="partial_answer", message="Le plafond actuel est de 990 dinars.")
    result = parse(value, [record()])
    assert "990" not in result["answer"]


def test_current_query_is_qualified_even_when_called_without_graph_router():
    result = parse(draft(), [record()], "What is the latest supported ceiling?")
    assert result["status"] == "partial_answer"


def test_long_contiguous_quote_and_empty_partial_message_are_not_rejected():
    text = "Conditions complémentaires. " * 65 + "Le plafond est de 320 dinars."
    value = draft(unit=66)  # a long page is split into sentences; cite the one that states it
    value["status"] = "partial_answer"
    result = parse(value, [record(text=text)])
    assert result["status"] == "partial_answer"
    assert result["sources"]


def test_small_word_number_equivalence():
    result = parse(draft("La durée est de 7 ans."), [record(text="La durée est de sept ans.")])
    assert result["status"] == "answered"


def test_cited_document_name_is_metadata_not_an_unsupported_rule_number():
    result = parse(draft("Le document Cir_2022_41_fr fixe le plafond à 320 dinars."), [record()])
    assert result["status"] == "answered"
    wrong = parse(draft("Le document Cir_2022_41_fr fixe le plafond à 41 dinars."), [record()])
    assert wrong["status"] == "insufficient_evidence"


def test_arabic_corrupt_header_warns_words_may_support_claims_digits_may_not():
    from answer_contract import evidence_records
    text = "مذكرة إلى البنوك عدد 41 لسنة 2202\nلون الورقة أخضر. الرمز هو 709. المدة ثلاث سنوات."
    records = evidence_records([(Document(page_content=text, metadata={"source": "Note_2022_41_ar.pdf", "page": 2}), 0.9)])
    assert records[0]["evidence_warning"] == "source_header_conflict"
    assert "unusable_reason" not in records[0]
    # Non-numeric fact from the body: supported.
    assert parse(draft("لون الورقة أخضر.", unit=2), records, "ما اللون؟")["status"] == "answered"
    # Digits on a page with reversed/garbled header digits are not reliable, even verbatim.
    assert parse(draft("الرمز هو 709.", unit=2), records, "ما الرمز؟")["status"] == "insufficient_evidence"
    assert parse(draft("الرمز هو 907.", unit=2), records, "ما الرمز؟")["status"] == "insufficient_evidence"
    # Numbers written in words remain usable.
    assert parse(draft("المدة 3 سنوات.", unit=2), records, "ما المدة؟")["status"] == "answered"
    # Spelling a garbled digit out in words does not launder it.
    assert parse(draft("الرمز هو سبعة.", unit=2), records, "ما الرمز؟")["status"] == "insufficient_evidence"


def test_claim_may_name_the_cited_instrument_year_without_quoting_it():
    table = "| d'hiver | d'été |\n| 630 | 1005 |"
    result = parse(dict(draft("En 2018, le plafond pour les fourrages d'hiver en sec était de 630 dinars par hectare."), claims=[dict(text="En 2018, le plafond pour les fourrages d'hiver en sec était de 630 dinars par hectare.", cites=["E1.1", "E1.2"])]),
                   [record(source="Cir_2018_11_fr.pdf", text=table)], "Quel était le plafond pour les fourrages d'hiver ?")
    assert result["status"] == "answered"
    other_year = parse(dict(draft(), claims=[dict(text="En 2016, le plafond pour les fourrages d'hiver était de 630 dinars.", cites=["E1.1", "E1.2"])]),
                       [record(source="Cir_2018_11_fr.pdf", text=table)], "Quel était le plafond pour les fourrages d'hiver ?")
    assert other_year["status"] == "insufficient_evidence"
    asked_year = parse(dict(draft(), claims=[dict(text="En 2016, le plafond pour les fourrages d'hiver était de 630 dinars.", cites=["E1.1", "E1.2"])]),
                       [record(source="Cir_2018_11_fr.pdf", text=table)], "Quel était le plafond en 2016 ?")
    assert asked_year["status"] == "answered"


def test_evidence_warning_targets_garbled_digits_not_body_cross_references():
    from answer_evidence import evidence_warning
    # A body reference to another instrument is not this page's header.
    body = "الفصل 23 : يجب ألا يكون المستفيد مخلّ بتعهدات على معنى المنشور عدد 6 لسنة 2008 المؤرخ في 10 مارس 2008."
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": body}) is None
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "منشور إلى البنوك عدد 04 لسنة 2016\n" + body}) is None
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "منشور إلى البنوك عدد 40 لسنة 2016"}) == "source_header_conflict"
    # Font-map garbling ("6112" for 2016) is caught anywhere on the page, spaced or not.
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": body + "\nقرض المبرم بتاريخ 18 مارس6112"}) == "implausible_gregorian_year"
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "القانون عدد 84 لسنة 6112"}) == "implausible_gregorian_year"
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "أجل أقصاه 31 ديسمبر 2020"}) is None


def test_claim_may_scope_itself_to_the_cited_instrument_identifier():
    page = "L'horaire conventionnel s'étend de 8h00 à 17h00 heure locale."
    ok = parse(draft("Selon la circulaire 2016-01, l'horaire s'étend de 8h00 à 17h00."),
               [record(source="Cir_2016_01_fr.pdf", text=page)], "Quelles sont les heures d'ouverture ?")
    assert ok["status"] == "answered"
    ok_ar = parse(draft("حسب المنشور عدد 1 لسنة 2016، يمتد التوقيت من 8h00 إلى 17h00."),
                  [record(source="Cir_2016_01_fr.pdf", text=page)], "ما هو التوقيت؟")
    assert ok_ar["status"] == "answered"
    # Another instrument's identifier is not trusted metadata for this claim.
    other = parse(draft("Selon la circulaire 2021-03, l'horaire s'étend de 8h00 à 17h00."),
                  [record(source="Cir_2016_01_fr.pdf", text=page)], "Quelles sont les heures d'ouverture ?")
    assert other["status"] == "insufficient_evidence"


def test_question_numbers_present_on_the_cited_page_may_be_restated():
    page = "Objet : billets de 100 et 500 couronnes.\nLes anciens billets restent acceptés jusqu'au 10 mai 2017 inclus."
    quote = "Les anciens billets restent acceptés jusqu'au 10 mai 2017 inclus."
    question = "Jusqu'à quand les billets de 100 et 500 couronnes restaient-ils acceptés ?"
    ok = parse(draft("Les billets de 100 et 500 couronnes restaient acceptés jusqu'au 10 mai 2017.", unit=2),
               [record(source="Note_2017_22_fr.pdf", text=page)], question)
    assert ok["status"] == "answered"
    # A number asked about but absent from the page is not laundered into a claim.
    leading = parse(draft("Le plafond est de 500 dinars."), [record()],
                    "Le plafond est-il de 500 dinars ?")
    assert leading["status"] == "insufficient_evidence"


def test_answer_layer_reads_the_whole_retrieved_page(monkeypatch):
    import conversation
    from retrieval_selection import expand_ranked_pages, page_chunks
    chunks = [Document(page_content=text, metadata={"source": "Cir_2016_02_fr.pdf", "page": 2, "pages": [2], "chunk_index": i})
              for i, text in enumerate(["Article 1. Objet du crédit auto et conditions générales applicables aux emprunteurs.",
                                        "conditions générales applicables aux emprunteurs. Article 2. La durée est de 7 ans.",
                                        "Article 3. Dispositions finales."])]
    pages = page_chunks(chunks)
    expanded = expand_ranked_pages([(chunks[0], 0.9)], pages)
    assert expanded[0][0].page_content == chunks[0].page_content + " Article 2. La durée est de 7 ans.\nArticle 3. Dispositions finales."
    assert expanded[0][0].metadata["source"] == "Cir_2016_02_fr.pdf" and expanded[0][0].metadata["page"] == 2
    # Long pages stay bounded around the retrieved chunk.
    assert expand_ranked_pages([(chunks[1], 0.9)], pages, max_chars=len(chunks[1].page_content) + 40)[0][0].page_content \
        == chunks[1].page_content + "\nArticle 3. Dispositions finales."
    # The chat flow expands answer evidence but keeps search fallback on original chunks.
    class Backend:
        def retrieve(self, query, other_queries=()):
            return [(chunks[0], 0.9)]
        def expand_pages(self, ranked):
            return expand_ranked_pages(ranked, pages)
    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(intent="NEW_TOPIC", rewrite_query="", new_topic="t", current_topic="t"))
    def answer(_llm, _question, evidence, *_args, **_kwargs):
        assert "7 ans" in evidence[0][0].page_content
        return dict(status="search_results", answer="fallback", sources=[])
    monkeypatch.setattr(conversation, "generate_grounded_answer", answer)
    result = conversation.chat("Quelle est la durée ?", {"topics": [], "turns": []}, retrieval_backend=Backend())
    assert result["sources"][0]["excerpt"] == chunks[0].page_content


def test_fallback_carries_rejection_diagnostics_for_offline_evaluation_only(monkeypatch):
    import conversation
    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        return AIMessage(content=json.dumps(draft(unit=9)))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    # Both drafts cite a unit that does not exist: no model is pushed further, the pages are listed.
    assert result["status"] == "search_results"
    assert result["diagnostics"][:2] == ["unknown_citation", "unknown_citation"]
    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(intent="NEW_TOPIC", rewrite_query="", new_topic="t", current_topic="t"))
    monkeypatch.setattr(conversation, "generate_grounded_answer", lambda *a, **k: result)
    class Backend:
        def retrieve(self, query, other_queries=()):
            return [(doc, .9)]
    chat_result = conversation.chat("Quel est le plafond ?", {"topics": [], "turns": []}, retrieval_backend=Backend())
    assert "diagnostics" not in chat_result
    assert chat_result["status"] == "search_results"


def test_invalid_citation_gets_a_bounded_retry_with_reason():
    responses = [draft(unit=9), draft()]
    seen = []
    def respond(prompt):
        seen.append(prompt.to_messages())
        if "Select evidence for" in seen[-1][0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        return AIMessage(content=json.dumps(responses.pop(0)))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    assert result["status"] == "answered"
    assert any("unknown_citation" in m[-1].content for m in seen)


def test_broad_summary_first_pass_multi_page_partial_skips_forced_fallback():
    """B: broad/summary asks can accept a multi-page quoted partial on the first draft."""
    calls = []

    def respond(prompt):
        messages = prompt.to_messages()
        calls.append(messages)
        system = messages[0].content
        if "Select evidence for" in system:
            assert "answer_intent is summary" in system or "broad topic briefing" in system
            return AIMessage(content=json.dumps(dict(
                decision="partial",
                answer_intent="summary",
                reason="complementary facts across pages",
                evidence_ids=["E1", "E2"],
            )))
        # First writer draft succeeds.
        return AIMessage(content=json.dumps(dict(
            status="partial_answer",
            message="",
            claims=[
                dict(
                    text="Selon la circulaire 2016-01, les cours au comptant doivent être affichés.",
                    cites=["E1.1"],
                ),
                dict(
                    text="Selon la circulaire 2021-03, les intermédiaires agréés peuvent négocier devises/dinar au comptant.",
                    cites=["E2.1"],
                ),
            ],
        )))

    docs = [
        Document(
            page_content="Les cours au comptant acheteur et vendeur des devises contre dinar tunisien doivent être portés à la connaissance du marché, de façon continue, par affichage électronique.",
            metadata={"source": "Cir_2016_01_fr.pdf", "page": 3, "pages": [3]},
        ),
        Document(
            page_content="Les Intermédiaires Agréés peuvent effectuer librement sur le marché des changes interbancaires des transactions de change devises/dinar au comptant, dans le respect des règles prévues.",
            metadata={"source": "Cir_2021_03_fr.pdf", "page": 3, "pages": [3]},
        ),
    ]
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Quelles sont les règles du marché des changes ?",
        [(docs[0], 0.9), (docs[1], 0.8)],
    )
    assert result["status"] == "partial_answer"
    assert "2016-01" in result["answer"]
    assert "2021-03" in result["answer"]
    assert len(calls) == 2  # select + one draft


def test_empty_draft_gets_a_retry_not_a_fake_abstention():
    """Blank model content is a provider hiccup: the second draft is asked for, not a refusal."""
    drafts = []
    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="summary", reason="ok", evidence_ids=["E1"],
            )))
        drafts.append(1)
        return AIMessage(content="" if len(drafts) == 1 else json.dumps(draft()))

    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quelles sont les règles du plafond ?", [(doc, 0.9)])
    assert len(drafts) == 2
    assert result["status"] == "answered"
    assert "320 dinars" in result["answer"]


def test_truncated_answer_json_salvages_complete_claims():
    """Broad drafts often hit the token wall mid-claim; keep finished claims without another LLM call."""
    from answer_contract import _load_answer_payload

    truncated = (
        '{"status":"partial_answer","message":"","claims":['
        '{"text":"Selon la circulaire 2022-41, le plafond est de 320 dinars.",'
        '"cites":["E1.1"]},'
        '{"text":"Selon la circulaire 2022-41, la marge est plafonn'
    )
    payload = _load_answer_payload(truncated)
    assert payload["status"] == "partial_answer"
    assert len(payload["claims"]) == 1
    result = parse_answer(truncated, "Quel est le plafond ?", [record()])
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]


def test_selection_keeps_valid_ids_when_model_adds_junk():
    """One invented evidence ID must not discard an otherwise usable selection."""
    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="answer", reason="ok", evidence_ids=["E1", "E99", "e1"],
            )))
        return AIMessage(content=json.dumps(draft()))

    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, 0.9)])
    assert result["status"] == "answered"
    assert "320 dinars" in result["answer"]


def test_question_anchors_keep_arabic_content_tokens():
    from answer_contract import _question_anchors

    anchors = _question_anchors("هل تذكر المنشورات سعر الذهب؟")
    assert "سعر" in anchors
    assert "الذهب" in anchors or "ذهب" in anchors
    assert "هل" not in anchors
    assert "المنشورات" not in anchors


def test_question_year_prefers_matching_instrument_over_older_facility():
    seen = []
    def respond(prompt):
        messages = prompt.to_messages()
        seen.append(messages)
        system = messages[0].content
        human = messages[-1].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="conditions", reason="mixed", evidence_ids=["E1", "E2"])))
        if "Answer a question about" in system:
            # Same-year evidence leads, but the selector's other pick is not discarded.
            assert human.index("Note_2024_163_fr.pdf") < human.index("Note_2020_01_fr.pdf")
            return AIMessage(content=json.dumps(dict(
                status="partial_answer", message="",
                claims=[dict(text="Selon la note 2024-163, le volume d'investissement ne dépasse pas 15 millions de dinars.",
                             cites=["E2.1"])],
            )))
        return AIMessage(content=json.dumps(dict(status="insufficient_evidence", message="", claims=[])))
    docs = [
        (Document(page_content="Taux d'intérêt : 6,5% l'an au maximum. Durée de remboursement : 12 ans.",
                  metadata={"source": "Note_2020_01_fr.pdf", "page": 4, "pages": [4]}), 0.95),
        (Document(page_content="On entend par PME toute entreprise dont le volume d'investissement ne dépasse pas quinze (15) millions de dinars.",
                  metadata={"source": "Note_2024_163_fr.pdf", "page": 2, "pages": [2]}), 0.9),
    ]
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Une petite entreprise obtient un crédit d'investissement en 2024. Quelles conditions lui sont applicables ?",
        docs,
    )
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["file"] == "Note_2024_163_fr.pdf"


def test_unresolved_scope_requests_clarification_without_drafting():
    def respond(prompt):
        assert "Select evidence for" in prompt.to_messages()[0].content
        return AIMessage(content=json.dumps(dict(decision="clarification_needed", reason="Same scope, different ceilings; no requested instrument or precedence.", evidence_ids=[])))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", docs)
    assert result["status"] == "clarification_needed"
    assert "320" not in result["answer"]
    assert result["sources"] == []


def test_mixed_claims_keep_only_literally_supported_parts():
    value = dict(
        status="answered",
        message="",
        claims=[
            dict(text="Le plafond est de 320 dinars.", cites=["E1.1"]),
            dict(text="Le taux est de 7,5 %.", cites=["E1.1"]),
        ],
    )
    result = parse(value, [record()])
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]
    assert "7,5" not in result["answer"]
    assert "partie de la demande" in result["answer"]
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


def test_claim_cannot_remap_question_actor_onto_a_different_regime_page():
    """Reject claims that restate a question actor absent from the cited page."""
    page = (
        "Les prix des ventes peuvent être réglés par n'importe quel moyen de règlement, "
        "lorsque les contrats y afférents prévoient des délais de règlement allant jusqu'à "
        "60 jours à compter de la date d'expédition des marchandises."
    )
    value = dict(
        status="answered",
        message="",
        claims=[
            dict(
                text=(
                    "Une entreprise peut régler un fournisseur étranger par n'importe quel "
                    "moyen de paiement lorsque le contrat prévoit un délai de 60 jours."
                ),
                cites=["E1.1"],
            ),
        ],
    )
    evidence = [record(source="Cir_2020_02_fr.pdf", text=page, eid="E1")]
    result = parse(
        value,
        evidence,
        question="Est-ce qu’une entreprise peut payer un fournisseur à l’étranger ?",
    )
    assert result["status"] == "insufficient_evidence"
    assert "fournisseur" not in result["answer"].casefold()


def test_question_verb_matches_accented_participle_on_page():
    page = (
        "Article premier : Est abrogée la circulaire n° 2017-09 en date du 27 octobre 2017, "
        "relative aux conditions de financement de l'importation de produits non prioritaires."
    )
    value = draft(
        "La circulaire abroge la circulaire n° 2017-09 relative aux conditions de financement "
        "de l'importation de produits non prioritaires.",
    )
    result = parse(
        value,
        [record(source="Cir_2018_13_fr.pdf", text=page)],
        question="La circulaire 2018-13 abroge-t-elle une circulaire antérieure, et laquelle ?",
    )
    assert result["status"] == "answered"


def test_evidence_view_keeps_units_with_most_question_words():
    from answer_draft import _evidence_view

    page = (
        "Les prix des produits de base ont progressé en juin.\n" + "Texte neutre sans rapport.\n" * 80
        + "Le cours du baril de Brent a augmenté de 19,4% en juin 2026 par rapport à juin 2025."
    )
    view = _evidence_view("Le prix du Brent a-t-il augmenté en juin 2026 ?", [record(text=page)],
                          max_chars=400, labels=True)
    assert "[E1.82] Le cours du baril de Brent a augmenté de 19,4%" in view[0]["text"]
    assert len(view[0]["text"]) < 600


def test_citations_resolve_to_exact_page_units_and_unknown_ids_fail():
    page = "Titre\nLe plafond est de 320 dinars.\nLa durée est de 12 ans."
    ok = parse(dict(draft("La durée est de 12 ans."), claims=[dict(text="La durée est de 12 ans.", cites=["E1.3"])]),
               [record(text=page)], "Quelle est la durée ?")
    assert ok["status"] == "answered"
    assert ok["sources"][0]["excerpt"] == "La durée est de 12 ans."
    assert parse(draft(unit=7), [record(text=page)])["status"] == "insufficient_evidence"
    assert parse(draft(eid="E4"), [record(text=page)])["status"] == "insufficient_evidence"


def test_chart_words_are_never_citable():
    from answer_draft import _evidence_view
    from answer_evidence import IMAGE_WORDS

    # Words inside a chart box come in drawing order: 67 sits next to the wrong age label.
    page = f"Répartition du personnel par âge\n{IMAGE_WORDS} Hommes Femmes de 30 à 34 ans 67 de 25 à 29 ans 100\nTotal: 834 agents."
    view = _evidence_view("Combien de femmes de 30 à 34 ans ?", [record(text=page)], max_chars=2500, labels=True)
    assert "67" not in view[0]["text"] and "[E1.2] Total: 834 agents." in view[0]["text"]
    ok = parse(draft("La BCT compte 834 agents.", unit=2), [record(text=page)], "Combien d'agents ?")
    assert ok["status"] == "answered" and ok["sources"][0]["excerpt"] == "Total: 834 agents."
    assert parse(draft("67 femmes de 30 à 34 ans.", unit=3), [record(text=page)])["status"] == "insufficient_evidence"


def test_fenced_json_and_extra_fields_still_parse():
    payload = draft()
    payload["commentary"] = "ignore me"
    payload["claims"][0]["notes"] = "also ignore"
    wrapped = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    result = parse_answer(wrapped, "Quel est le plafond ?", [record()])
    assert result["status"] == "answered"
    assert "320 dinars" in result["answer"]


def test_all_unsupported_claims_still_fail_closed():
    value = dict(
        status="answered",
        message="",
        claims=[
            dict(text="Le taux est de 7,5 %.", cites=["E1.1"]),
            dict(text="La durée est de 12 ans.", cites=["E1.9"]),
        ],
    )
    result = parse(value, [record()])
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_selector_insufficiency_gets_one_literal_checked_partial_attempt():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        if "Select evidence for" in calls[-1][0].content:
            return AIMessage(content=json.dumps(dict(
                decision="insufficient_evidence", reason="uncertain scope", evidence_ids=[])))
        assert "uncertain scope" in calls[-1][-1].content
        return AIMessage(content=json.dumps(draft()))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    # The selector doubted the pack, so a validated draft cannot claim a full answer.
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"
    assert len(calls) == 2


def test_best_effort_attempt_never_bypasses_named_instrument_identity():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        if "Select evidence for" in calls[-1][0].content:
            return AIMessage(content=json.dumps(dict(
                decision="insufficient_evidence", reason="uncertain", evidence_ids=[])))
        assert "Cir_2024_52_fr.pdf" not in calls[-1][-1].content
        return AIMessage(content=json.dumps(draft()))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond),
        "Selon la circulaire 2022-41, quel est le plafond ?", docs)
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


def test_broad_question_synthesizes_selected_parts_and_carries_answer_intent():
    seen = []
    def respond(prompt):
        messages = prompt.to_messages()
        seen.append(messages)
        if "Select evidence for" in messages[0].content:
            assert "answer_intent" in messages[0].content
            assert "Do not confuse topical evidence" in messages[0].content
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="conditions", reason="E1 and E2 support distinct conditions",
                evidence_ids=["E1", "E2"])))
        assert '"answer_intent": "conditions"' in messages[-1].content
        return AIMessage(content=json.dumps(dict(status="partial_answer", message="", claims=[
            dict(text="La condition A s'applique.", cites=["E1.1"]),
            dict(text="La condition B s'applique.", cites=["E2.1"]),
        ])))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(text="La condition A s'applique."),
                      record("Cir_2022_41_fr.pdf", "La condition B s'applique.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond),
        "Quelles sont les conditions A, B et C ?", docs)
    assert result["status"] == "partial_answer"
    assert "La condition A s'applique." in result["answer"]
    assert "La condition B s'applique." in result["answer"]
    assert len(seen) == 2


def test_draft_cannot_cite_evidence_excluded_by_selection():
    """Drafts only see selected evidence; citing an excluded page fails, and nothing reopens it."""
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        system = calls[-1][0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="E2 concerns another operation", evidence_ids=["E1"])))
        assert '"evidence_id": "E2"' not in calls[-1][-1].content
        return AIMessage(content=json.dumps(draft("Le plafond est de 640 dinars.", eid="E2")))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", docs)
    assert result["status"] == "search_results"
    assert len(calls) == 3  # selection + two drafts


def test_named_document_is_the_only_candidate_for_direct_contents_question():
    def respond(prompt):
        messages = prompt.to_messages()
        assert "Cir_2024_52_fr.pdf" not in messages[-1].content
        if "Select evidence for" in messages[0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E2"])))
        return AIMessage(content=json.dumps(draft(eid="E2")))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars."), record()]]
    result = generate_grounded_answer(RunnableLambda(respond), "Selon la circulaire 2022-41, quel est le plafond ?", docs)
    assert result["status"] == "answered"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


@pytest.mark.parametrize("question", [
    "Quel est le dernier plafond applicable ?", "ما أحدث سقف؟", "What is the latest ceiling?",
])
def test_currentness_detection_in_both_answer_entrypoints(question):
    from query_currentness import is_temporal_rule_query
    assert is_temporal_rule_query(question)
    assert parse(draft(), [record()], question)["status"] == "partial_answer"


def test_historical_uncertainty_is_not_described_as_todays_latest_value():
    result = parse(draft(), [record()], "Au 1er janvier 2023, quel plafond était applicable ?")
    assert result["status"] == "partial_answer"
    assert "date demandée" in result["answer"]
    assert "dernière valeur" not in result["answer"]


def test_resolved_applicability_can_be_explicitly_passed_by_the_caller():
    result = parse(draft(), [record()], "What is currently in force?", temporal_unverified=False)
    assert result["status"] == "answered"


@pytest.mark.parametrize("claim", ["Le plafond actuel est de 320 dinars.", "The current ceiling is 320 dinars.", "السقف الحالي هو 320 دينار."])
def test_a_disclaimer_does_not_validate_an_unverified_currentness_assertion(claim):
    result = parse(draft(claim), [record()],
                   "What is currently applicable?", temporal_unverified=True)
    assert result["status"] == "insufficient_evidence"
    assert not result["sources"]


def test_negative_force_wording_is_allowed_when_quote_supports_abrogation():
    evidence = [{
        "evidence_id": "E1",
        "source": "Cir_2019_07_fr.pdf",
        "page": 2,
        "text": (
            "Les dispositions de l'article 2 de la circulaire n°2018-07 "
            "sont abrogées."
        ),
        "score": 1.0,
        "temporal_relation": "ABROGATES",
        "temporal_source_id": "cir:2019:7",
        "temporal_target_id": "cir:2018:7",
    }]
    claim = (
        "L'article 2 de la circulaire 2018-07 n'est plus en vigueur : "
        "il est abrogé par la circulaire 2019-07."
    )
    quote = "Les dispositions de l'article 2 de la circulaire n°2018-07 sont abrogées."
    result = parse(
        draft(claim),
        evidence,
        "L'article 2 de la circulaire 2018-07 est-il encore en vigueur ?",
        temporal_unverified=True,
    )
    assert result["status"] == "partial_answer"
    assert result["sources"]
    assert "abrog" in result["answer"].casefold()


def test_unsupported_no_later_text_claim_is_rejected():
    """Absence of a retrieved amendment is not proof that none exists."""
    claim = (
        "Selon la circulaire 2025-13, le délai libre est de 120 jours et "
        "aucun texte ultérieur ne modifie ces dispositions."
    )
    quote = (
        "Les ventes dont les contrats prévoient des délais de règlement allant "
        "jusqu'à 120 jours sont effectuées librement et sans autorisation."
    )
    result = parse(
        draft(claim),
        [record(text=quote)],
        "Quelles dispositions de 2025-13 restent pertinentes aujourd'hui ?",
    )
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_draft_evidence_cap_prefers_distinct_pages():
    from answer_draft import _cap_draft_evidence

    records = [
        {"evidence_id": "E1", "source": "Cir_2025_13_fr.pdf", "page": 2, "text": "a"},
        {"evidence_id": "E2", "source": "Cir_2025_13_fr.pdf", "page": 2, "text": "b"},
        {"evidence_id": "E3", "source": "Cir_2025_13_fr.pdf", "page": 3, "text": "c"},
        {"evidence_id": "E4", "source": "Cir_2020_02_fr.pdf", "page": 1, "text": "d"},
    ]
    capped = _cap_draft_evidence(records, limit=3)
    pages = [(r["source"], r["page"]) for r in capped]
    assert pages == [
        ("Cir_2025_13_fr.pdf", 2),
        ("Cir_2025_13_fr.pdf", 3),
        ("Cir_2020_02_fr.pdf", 1),
    ]


def test_temporal_counterpart_ids_are_trusted_like_cited_filenames():
    from answer_evidence import strip_instrument_references, trusted_years

    records = [{
        "source": "Cir_2019_07_fr.pdf",
        "temporal_source_id": "cir:2019:7",
        "temporal_target_id": "cir:2018:7",
    }]
    stripped = strip_instrument_references(
        "La circulaire 2019-07 abroge la circulaire 2018-07.",
        records,
    )
    assert "2018" not in stripped
    assert "2019" not in stripped
    assert "2018" in trusted_years("encore en vigueur ?", records)


def test_supersession_partial_from_pinned_evidence_without_llm():
    from answer_contract import try_supersession_partial_answer

    evidence = [{
        "evidence_id": "E1",
        "source": "Cir_2019_07_fr.pdf",
        "page": 2,
        "text": (
            "Les dispositions de l'article 2 de la circulaire n°2018-07 "
            "sont abrogées."
        ),
        "score": 9.0,
        "temporal_relation": "ABROGATES",
        "temporal_source_id": "cir:2019:7",
        "temporal_target_id": "cir:2018:7",
    }]
    result = try_supersession_partial_answer(
        "L'article 2 de la circulaire 2018-07 est-il encore en vigueur ?",
        evidence,
    )
    assert result is not None
    assert result["status"] == "partial_answer"
    assert result["sources"]
    assert "abrog" in result["answer"].casefold() or "2018" in result["answer"]


def test_supersession_partial_states_the_edge_without_filler_claims():
    from answer_contract import try_supersession_partial_answer

    evidence = [
        {
            "evidence_id": "E1",
            "source": "Cir_2019_07_fr.pdf",
            "page": 2,
            "text": (
                "Les dispositions de l'article 2 de la circulaire n°2018-07 "
                "sont abrogées."
            ),
            "score": 9.0,
            "temporal_relation": "ABROGATES",
            "temporal_source_id": "cir:2019:7",
            "temporal_target_id": "cir:2018:7",
        },
        {
            "evidence_id": "E2",
            "source": "Cir_2019_07_fr.pdf",
            "page": 4,
            "text": (
                "La duree quotidienne du travail est fixee a huit heures "
                "pendant la seance unique des etablissements de credit."
            ),
            "score": 8.0,
        },
    ]
    result = try_supersession_partial_answer(
        "Quels sont les horaires de travail des banques ?",
        evidence,
    )
    assert result is not None
    assert result["status"] == "partial_answer"
    assert "abroge" in result["answer"].casefold()
    assert "passage cité énonce" not in result["answer"]


def test_search_fallback_preserves_original_top5_not_currentness_answer_order(monkeypatch):
    import conversation
    docs = [(Document(page_content=f"Passage original {n}", metadata={
        "source": f"Cir_{2020+n}_41_fr.pdf", "page": 2, "pages": [2]}), 1 - n / 10)
        for n in range(6)]
    class Backend:
        def retrieve(self, query, other_queries=()):
            return docs
    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(
        intent="NEW_TOPIC", rewrite_query="", new_topic="plafond", current_topic="plafond"))
    def answer(_llm, _question, evidence, *_args, **_kwargs):
        assert evidence[0][0].metadata["source"] == "Cir_2024_41_fr.pdf"
        return dict(status="search_results", answer="fallback", sources=[])
    monkeypatch.setattr(conversation, "generate_grounded_answer", answer)
    result = conversation.chat("Quel est le plafond actuel ?", {"topics": [], "turns": []},
                               retrieval_backend=Backend())
    assert result["status"] == "search_results"
    assert [s["file"] for s in result["sources"]] == [d.metadata["source"] for d, _ in docs[:5]]
    assert [s["excerpt"] for s in result["sources"]] == [d.page_content for d, _ in docs[:5]]


def test_no_retrieval_does_not_invent_search_results():
    def unexpected(_):
        raise AssertionError("Model must not be called without evidence")
    result = generate_grounded_answer(RunnableLambda(unexpected), "Quel est le plafond ?", [])
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_search_results_survive_api_serialization_and_history(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    from conversation_memory import ConversationStore
    from answer_contract import search_response
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: ConversationStore(tmp_path / "history.sqlite3"))
    result = search_response("Quel est le plafond ?", [record(eid=f"E{n}", source=f"Cir_2022_{n:02}_fr.pdf") for n in range(1, 6)])
    monkeypatch.setattr(app_module, "chat", lambda *args, **kwargs: {**result, "memory_state": {}})
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "Quel est le plafond ?"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "search_results"
        assert body["sources"] == result["sources"]
        saved = client.get(f"/conversations/{body['conversation_id']}").json()["turns"][-1]
        assert saved["answer_status"] == "search_results"
        assert saved["sources"] == result["sources"]


def test_supersession_partial_needs_the_relationship_verb_in_the_quote():
    from answer_contract import try_supersession_partial_answer

    evidence = [{
        "evidence_id": "E1",
        "source": "Cir_2019_07_fr.pdf",
        "page": 2,
        "text": "Objet : horaires de travail des etablissements de credit pendant la seance unique d'ete.",
        "score": 9.0,
        "temporal_relation": "ABROGATES",
        "temporal_source_id": "cir:2019:7",
        "temporal_target_id": "cir:2018:7",
    }]
    history = []
    assert try_supersession_partial_answer("Horaires des banques ?", evidence, diagnostics=history) is None
    assert "supersession_partial:no_declaring_unit" in history


def test_selector_and_draft_agreeing_on_insufficient_stop_the_ladder():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        if "Select evidence for" in calls[-1][0].content:
            return AIMessage(content=json.dumps(dict(
                decision="insufficient_evidence", reason="banknote fee, not a transfer fee", evidence_ids=[])))
        if len(calls) == 2:
            return AIMessage(content=json.dumps(dict(status="insufficient_evidence", message="", claims=[])))
        return AIMessage(content=json.dumps(draft()))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    # Both found nothing that answers: a short refusal, not a list of unrelated pages.
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
    assert "draft_abstained:insufficient_evidence" in result["diagnostics"]
    assert len(calls) == 2


def test_draft_prompt_names_the_question_language():
    seen = []
    def respond(prompt):
        messages = prompt.to_messages()
        seen.append(messages)
        if "Select evidence for" in messages[0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="E1", evidence_ids=["E1"])))
        return AIMessage(content=json.dumps(draft()))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    generate_grounded_answer(RunnableLambda(respond), "What is the ceiling?", [(doc, .9)])
    assert "Write the claims in English." in seen[1][-1].content


def test_replacement_fallback_answers_only_a_question_about_replacement():
    declaring = Document(
        page_content="Article 7 : La présente circulaire abroge et remplace les dispositions de la circulaire n°2016-08.",
        metadata={"source": "Cir_2020_03_fr.pdf", "page": 7, "pages": [7], "temporal_relation": "ABROGATES",
                  "temporal_source_id": "cir:2020:3", "temporal_target_id": "cir:2016:8"},
    )

    def respond(prompt):
        if "Select evidence for" in prompt.to_messages()[0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", evidence_ids=["E1"])))
        return AIMessage(content="not json")  # both drafts fail

    plafond = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond de l'allocation ?", [(declaring, 9000.0)])
    assert plafond["status"] == "search_results"  # not "2020-03 abroge 2016-08"
    replaced = generate_grounded_answer(RunnableLambda(respond), "Quelle circulaire a remplacé la circulaire 2016-08 ?",
                                        [(declaring, 9000.0)])
    assert replaced["status"] == "partial_answer"
